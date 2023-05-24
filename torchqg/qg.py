import math
import tqdm

import h5py

import torch
import torch.fft

import matplotlib
import matplotlib.pyplot as plt

from torchqg.grid import TwoGrid
from torchqg.timestepper import ForwardEuler, RungeKutta2, RungeKutta4, PureMLStepper
from torchqg.pde import Pde, Eq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', device)

def to_spectral(y): return torch.fft. rfftn(y, norm='forward')
def to_physical(y): return torch.fft.irfftn(y, norm='forward')

class QgModel:
  def __init__(self, name, Nx, Ny, Lx, Ly, dt, t0, B, mu, nu, nv, eta, source=None, kernel=None, sgs=None):
    """
    Notation: 
    - q, $\omega$ is potential vorticity
    - p, $\psi$ is streamfunction
    - u, $u$ is x-axis velocity
    - v, $v$ is y-axis velocity
    - Variables in spectral space contain an h
    - Spatially filtered variables contain an _

    Args:
    """
    self.name = name
    
    # See pce-pinns/solver/qgturb.py
    self.B = B # Planetary vorticity y-gradient, aka, Rossby parameter?
    self.mu = mu # linear drag coefficient
    self.nu = nu # Viscosity coefficient
    self.nv = nv # Hyperviscous order (nv=1 is viscosity)
    self.eta = eta.to(device)
    
    self.grid = TwoGrid(device, Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)

    if sgs:
      # use 3/2 rule
      self.eq = Eq(grid=self.grid, linear_term=self.linear_term(self.grid), nonlinear_term=self.nonlinear_les)
      self.da = TwoGrid(device, Nx=int((3./2.)*Nx), Ny=int((3./2.)*Ny), Lx=Lx, Ly=Ly, dealias=1/3)
    else:
      # use 2/3 rule
      self.eq = Eq(grid=self.grid, linear_term=self.linear_term(self.grid), nonlinear_term=self.nonlinear_dns)
    self.stepper = RungeKutta4(eq=self.eq)

    self.pde = Pde(dt=dt, t0=t0, eq=self.eq, stepper=self.stepper)
    self.source = source
    self.kernel = kernel
    self.sgs = sgs

  def __str__(self):
    return """Qg model
       Grid: [{nx},{ny}] in [{lx},{ly}]
       μ: {mu}
       ν: {nu}
       β: {beta}
       dt: {dt}
       """.format(
      nx=self.grid.Nx, 
      ny=self.grid.Ny, 
      lx=self.grid.Lx,
      ly=self.grid.Ly,
      mu=self.mu,
      nu=self.nu,
      beta=self.B,
      dt=self.pde.cur.dt)

  def nonlinear_dns(self, i, S, sol, dt, t, grid):
    """
    Calculates nonlinear terms in eq. (19) top of Frezat 
    et al 22. Specifically, the nonlinear Jacobian, 
    J(psi,omega), and source, F.
    """

    # Calculate streamfunction and velocities from 
    # vorticity
    qh = sol.clone()
    ph = -qh * grid.irsq
    uh = -1j * grid.ky * ph # eq (10) in Frezat et al., 22
    vh =  1j * grid.kr * ph # eq (10) in Frezat et al., 22

    q = to_physical(qh)
    u = to_physical(uh)
    v = to_physical(vh)

    # Calculate nonlinear Jacobian, eq (11.5): 
    ## - J(p, q) = - dx(p) dy(q) + dy(p) dx(q)
    ## - J(p, q) = - v dy(q) - u dx(q)
    qe = q + self.eta # eta = 0 in default setting.
    uq = u * qe
    vq = v * qe

    uqh = to_spectral(uq)
    vqh = to_spectral(vq)
    S[:] = -1j * grid.kr * uqh - 1j * grid.ky * vqh

    grid.dealias(S[:])

    if (self.source):
      # Add external forcing, e.g., wind-forcing
      S[:] += self.source(i, sol, dt, t, grid)

  def nonlinear_les(self, i, S, sol, dt, t, grid):
    """
    Calculates eq (19)-bottom from Frezat et al 22.
    These are similar calculations to nonlinear_dns, with
    the addition of the subgrid (SGS) term from eq (20). 
    """
    qh = sol.clone() # e.g., shape (128, 65)
    ph = -qh * grid.irsq
    uh = -1j * grid.ky * ph
    vh =  1j * grid.kr * ph
    eh = to_spectral(self.eta)

    qhh = self.da.increase(qh) # e.g., shape (192, 97)
    uhh = self.da.increase(uh)
    vhh = self.da.increase(vh)
    ehh = self.da.increase(eh)

    q = to_physical(qhh) # e.g., shape (192, 192)
    u = to_physical(uhh)
    v = to_physical(vhh)
    e = to_physical(ehh)

    qe = q + e
    uq = u * qe
    vq = v * qe

    uqhh = to_spectral(uq) # e.g., shape (192, 97)
    vqhh = to_spectral(vq)

    uqh = grid.reduce(uqhh) # e.g., shape (128, 65)
    vqh = grid.reduce(vqhh)

    S[:] = -1j * grid.kr * uqh - 1j * grid.ky * vqh # e.g., shape (128, 65)

    if (self.sgs):
      # Calculate and add SGS term as forcing in spectral space
      S[:] += self.sgs.predict(self, i, sol, grid)

    if (self.source):
      S[:] += self.source(i, sol, dt, t, grid)

  def linear_term(self, grid):
    """
    Linear terms on right hand-side (RHS) in eq. (19) of Frezat et al., 22
    https://doi.org/10.1029/2022MS003124
    """
    Lc = -self.mu - self.nu * grid.krsq**self.nv - 1j * self.B * grid.kr * grid.irsq
    Lc[0, 0] = 0
    return Lc
    
  # Flow with random gaussian energy only in the wavenumbers range
  def init_randn(self, energy, wavenumbers):
    K = torch.sqrt(self.grid.krsq) # Wavenumber of each point in frequency space
    k = self.grid.kr.repeat(self.grid.Ny, 1)

    qih = torch.randn(self.pde.sol.size(), dtype=torch.complex128).to(device)
    qih[K < wavenumbers[0]] = 0.0
    qih[K > wavenumbers[1]] = 0.0
    qih[k == 0.0] = 0.0

    E0 = energy
    Ei = 0.5 * (self.grid.int_sq(self.grid.kr * self.grid.irsq * qih) + self.grid.int_sq(self.grid.ky * self.grid.irsq * qih)) / (self.grid.Lx * self.grid.Ly)
    qih *= torch.sqrt(E0 / Ei)
    self.pde.sol = qih
    
  def update(self):
    """
    Returns all state variables in physical space by calculating 
    streamfunction and velocities from vorticity
    
    Returns:
      q torch.Tensor([self.grid.Ny, self.grid.Nx]): Potential vorticity
      p torch.Tensor([self.grid.Ny, self.grid.Nx]): Streamfunction
      u torch.Tensor([self.grid.Ny, self.grid.Nx]): x-axis velocity
      v torch.Tensor([self.grid.Ny, self.grid.Nx]): y-axis velocity
    """
    qh = self.pde.sol.clone() # PDE solution only stores pot. vorticity
    ph = -qh * self.grid.irsq
    uh = -1j * self.grid.ky * ph
    vh =  1j * self.grid.kr * ph

    q = to_physical(qh)
    p = to_physical(ph)
    u = to_physical(uh)
    v = to_physical(vh)
    return q, p, u, v

  def J(self, grid, qh):
    ph = -qh * grid.irsq
    uh = -1j * grid.ky * ph
    vh =  1j * grid.kr * ph

    q = to_physical(qh)
    u = to_physical(uh)
    v = to_physical(vh)

    uq = u * q
    vq = v * q

    uqh = to_spectral(uq)
    vqh = to_spectral(vq)

    J = 1j * grid.kr * uqh + 1j * grid.ky * vqh
    return J

  def R(self, grid, scale):
    """
    Returns the exact SGS parametrization, R, for the large-scale
    grid <grid> and with reduced scale by factor <scale> based
    on filtering the DNS solution that is stored in self.pde.sol

    Args:
      grid src.grid.TwoGrid
      scale int
    Returns:
      R_field torch.Tensor([]): SGS param in spectral space
    """
    return self.R_field(grid, scale, self.pde.sol)

  def R_field(self, grid, scale, yh):
    """
    See eq (20) in Frezat et al., 22
    https://doi.org/10.1029/2022MS003124
    """
    return grid.div(torch.stack(self.R_flux(grid, scale, yh), dim=0))

  def R_flux(self, grid, scale, yh):
    """
    See eq (20) in Frezat et al., 22
    https://doi.org/10.1029/2022MS003124
    """    
    # Calc. streamfn and velocities from vorticity
    qh = yh.clone()
    ph = -qh * self.grid.irsq # \psi = \nabla^2 \omega
    uh = -1j * self.grid.ky * ph # u = -\partial_y \psi
    vh =  1j * self.grid.kr * ph # v = \partial_x \psi

    q = to_physical(qh)
    u = to_physical(uh)
    v = to_physical(vh)

    # Calc. \overline{\vec{u} \omega}
    uq = u * q
    vq = v * q
    uqh = to_spectral(uq)
    vqh = to_spectral(vq)
    uqh_ = self.kernel(scale * self.grid.delta(), uqh)
    vqh_ = self.kernel(scale * self.grid.delta(), vqh)

    # Calc. \bar{\vec{u}} \bar\omega
    uh_  = self.kernel(scale * self.grid.delta(), uh)
    vh_  = self.kernel(scale * self.grid.delta(), vh)
    qh_  = self.kernel(scale * self.grid.delta(), qh)
    u_ = to_physical(uh_)
    v_ = to_physical(vh_)
    q_ = to_physical(qh_)
    u_q_ = u_ * q_
    v_q_ = v_ * q_
    u_q_h = to_spectral(u_q_)
    v_q_h = to_spectral(v_q_)

    tu = u_q_h - uqh_
    tv = v_q_h - vqh_
    return grid.reduce(tu), grid.reduce(tv)

  # Returns filtered spectral variable, y. 
  def filter(self, grid, scale, y):
    yh = y.clone()
    return grid.reduce(self.kernel(scale * self.grid.delta(), yh))

  def filter_physical(self, grid, scale, y):
    yh = to_spectral(y)
    yl = grid.reduce(self.kernel(scale * self.grid.delta(), yh))
    yl = to_physical(yl)
    return yl

  def run(self, iters, visit, update=False, invisible=False):
    for it in tqdm.tqdm(range(iters), disable=invisible):
      self.pde.step(self)
      visit(self, self.pde.cur, it)
    if update:
      return self.update()

  # Diagnostics
  def energy(self, u, v):
    return 0.5 * torch.mean(u**2 + v**2)

  def enstrophy(self, q):
    return 0.5 * torch.mean(q**2)

  def cfl(self):
    _, _, u, v = self.update()
    return (u.abs().max() * self.pde.cur.dt) / self.grid.dx + (v.abs().max() * self.pde.cur.dt) / self.grid.dy

  def spectrum(self, y):
    K = torch.sqrt(self.grid.krsq)
    d = 0.5
    k = torch.arange(1, int(self.grid.kcut + 1))
    m = torch.zeros(k.size())

    e = [torch.zeros(k.size()) for _ in range(len(y))]
    for ik in range(len(k)):
      n = k[ik]
      i = torch.nonzero((K < (n + d)) & (K > (n - d)), as_tuple=True)
      m[ik] = i[0].numel()
      for j, yj in enumerate(y):
        e[j][ik] = torch.sum(yj[i]) * k[ik] * math.pi / (m[ik] - d)
    return k, e

  def invariants(self, qh):
    ph = -qh * self.grid.irsq
    uh = -1j * self.grid.ky * ph
    vh =  1j * self.grid.kr * ph

    # kinetic energy
    e = torch.abs(uh)**2 + torch.abs(vh)**2
    # enstrophy
    z = torch.abs(qh)**2

    k, [ek, zk] = self.spectrum([e, z])
    return k, ek, zk

  def fluxes(self, R, qh):
    # resolved rate
    sh = -torch.conj(qh) * self.J(self.grid, qh)
    # modeled rate
    lh =  torch.conj(qh) * R

    k, [sk, lk] = self.spectrum([torch.real(sh), torch.real(lh)])
    return k, sk, lk

  # Data
  def save(self, name):
    hf = h5py.File(name, 'w')
    hf.create_dataset('q', data=to_physical(self.p_.sol).cpu().detach())
    hf.close()

  def load(self, name):
    hf = h5py.File(name, 'r')
    fq = hf.get('q')
    sq = to_spectral(torch.from_numpy(fq[:]).to(device))

    # Copy first wavenumbers
    self.pde.sol = self.grid.increase(sq)
    hf.close()

  def zero_grad(self):
    self.stepper.zero_grad()

class QgModelPureML(QgModel):
  def __init__(self, name, Nx, Ny, Lx, Ly, dt, t0, B=None, mu=None, nu=None, nv=None, eta=None, source=None, kernel=None, sgs=None, model=None):
    """
    Args:
      model: torch.model Takes in current solution q and predict solution at next time step qnext
      dtype: dtype at which model is integrating
    """
    # ML model is subclass s.t. it can inherit most of the analysis functions
    super(QgModel, self).__init__()

    self.name = name

    self.grid = TwoGrid(device, Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
    self.eq = Eq(grid=self.grid, linear_term=self.zeros(self.grid), nonlinear_term=self.nonlinear_pure_ml)

    self.stepper = PureMLStepper(eq=self.eq)

    self.pde = Pde(dt=dt, t0=t0, eq=self.eq, stepper=self.stepper)
    self.kernel = kernel
    self.sgs = None
    
    self.model = model
    self.model.eval()
    self.model.test(model, mode=True) # Activates normalization during test

  def nonlinear_pure_ml(self, i, S, sol, dt=None, t=None, grid=None):
    """
    Uses ML model to forecast the next step given the current.
    Only forecasts q.

    Args:
      S torch((grid.irsq.shape)): Placeholder for the solution at t+1
      sol torch((grid.irsq.shape)): Current solution at t
      not used: i, dt, t, grid
    """
    qh = sol.clone() # Current potential vorticity

    # Convert to expected model input
    q = to_physical(qh)
    q = q[None,...,None].type(torch.float32)

    # Forecast next step with model
    qnext = self.model(q).squeeze()

    # Project back into spectral space, s.t., all analysis functions work
    S[:] = to_spectral(qnext)

  def zeros(self, grid):
    "These zeros are used as linear term and are queried to get the solution shape."
    # return torch.zeros((grid.Nx, grid.Ny), dtype=self.dtype)
    return torch.zeros((grid.irsq.shape), dtype=torch.complex128)

  def __str__(self):
    return """Qg model forecasting large-scale dynamics with Pure ML
       Grid: [{nx},{ny}] in [{lx},{ly}]
       dt: {dt}
       """.format(
      nx=self.grid.Nx, 
      ny=self.grid.Ny, 
      lx=self.grid.Lx,
      ly=self.grid.Ly,
      dt=self.pde.cur.dt)