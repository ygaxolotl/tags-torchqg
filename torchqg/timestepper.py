import math
import torch

class ForwardEuler:
  def __init__(self, eq):
    self.n = 1
    self.S = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)

  def zero_grad(self):
    self.S.detach_()

  def step(self, m, sol, cur, eq, grid):
    dt = cur.dt
    t  = cur.t

    eq.nonlinear_term(0, self.S, sol, dt, t, grid)
    self.S += eq.linear_term*sol.clone()
    sol += dt*self.S
    cur.step()

class RungeKutta2:
  def __init__(self, eq):
    self.n = 2
    self.S    = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)
    self.rhs1 = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)
    self.rhs2 = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)

  def zero_grad(self):
    self.S.detach_()
    self.rhs1.detach_()
    self.rhs2.detach_()

  def step(self, m, sol, cur, eq, grid):
    dt = cur.dt
    t  = cur.t

    # substep 1
    eq.nonlinear_term(0, self.rhs1, sol, dt, t, grid)
    self.rhs1 += eq.linear_term*sol

    # substep 2
    self.S = sol + self.rhs1 * dt*0.5
    eq.nonlinear_term(1, self.rhs2, self.S, dt*0.5, t + dt*0.5, grid)
    self.rhs2 += eq.linear_term*self.S

    sol += dt*self.rhs2
    cur.step()

class RungeKutta4:
  def __init__(self, eq):
    self.n    = 4
    self.S    = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)
    self.rhs1 = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)
    self.rhs2 = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)
    self.rhs3 = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)
    self.rhs4 = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)

  def zero_grad(self):
    self.S.detach_()
    self.rhs1.detach_()
    self.rhs2.detach_()
    self.rhs3.detach_()
    self.rhs4.detach_()

  def step(self, m, sol, cur, eq, grid):
    dt = cur.dt
    t  = cur.t

    # substep 1  
    eq.nonlinear_term(i=0, S=self.rhs1, sol=sol, dt=dt, t=t, grid=grid)
    self.rhs1 += eq.linear_term*sol

    # substep 2
    self.S = sol + self.rhs1 * dt*0.5
    eq.nonlinear_term(1, self.rhs2, self.S, dt*0.5, t + dt*0.5, grid)
    self.rhs2 += eq.linear_term*self.S

    # substep 3
    self.S = sol + self.rhs2 * dt*0.5
    eq.nonlinear_term(2, self.rhs3, self.S, dt*0.5, t + dt*0.5, grid)
    self.rhs3 += eq.linear_term*self.S

    # substep 4
    self.S = sol + self.rhs3 * dt
    eq.nonlinear_term(3, self.rhs4, self.S, dt, t + dt, grid)
    self.rhs4 += eq.linear_term*self.S

    sol += dt*(self.rhs1/6.0 + self.rhs2/3.0 + self.rhs3/3.0 + self.rhs4/6.0)
    cur.step()

class PureMLStepper:
  def __init__(self, eq):
    """
    Uses a pure ML model to predict next time step given the present.
    u(t+1) = ML(u(t)). This ML model assumes fixed time step between train
    and test dataset.
    """
    self.n    = 1
    self.S    = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)

  def zero_grad(self):
    self.S.detach_()

  def step(self, m, sol, cur, eq, grid):
    eq.nonlinear_term(i=0, S=self.S, sol=sol, dt=None, t=None, grid=None)
    sol[:] = self.S
    cur.step()