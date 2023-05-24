import math
import numpy as np
import torch

import torchqg.qg as qg

class Constant:
  def __init__(self, c=0.0):
    self.c = c

  def predict(self, m, it, sol, grid, verbose=False):
    div = torch.full_like(sol, self.c)
    return div

class MLdiv:
  def __init__(self, model):
    self.model = model
    self.model.eval()
    #print(self.model)

  def predict(self, m, it, sol, grid, verbose=False):
    """
    Uses machine learning (ML) model to predict SGS parametrization
    in physical space and then converts to spectral space
    """
    qh = sol.clone()
    ph = -qh * grid.irsq # calc. streamfn from vorticity

    q = qg.to_physical(qh)
    p = qg.to_physical(ph)

    # M(q, p) = M({i})
    i = torch.stack((q, p), dim=0) 
    # M({i}) ~ r
    r = self.model(i.unsqueeze(0).to(torch.float32)).view(grid.Ny, grid.Nx)
    return qg.to_spectral(r)

class MNOparam:
  def __init__(self, model):
    self.model = model # Deep Learning model
    self.model.eval()
    self.model.test(model, mode=True) # Activates normalization during test

  def predict(self, m, it, sol, grid, verbose=False):
    # Prepare MNO vorticity input
    qh = sol.clone()
    q = qg.to_physical(qh)
    
    q = q.to(torch.float32).unsqueeze(-1).unsqueeze(0) # convert to ML shape
    r = self.model(q)
    print('r min,max, mean, std: ', r.min(), r.max(), r.mean(), r.std())
    r = r.view(grid.Ny, grid.Nx) # convert to diffeq shape
    return qg.to_spectral(r)

class Testparam:
    def __init__(self, rs_tst, fdns_tst, time_tst):
        """  
        Tests if ground-truth parametrizations can be 
        plugged back into equation and achieve 100% accuracy 
        Args:
            rs np.array([steps, Nyl, Nyx]): Ground-truth parametrization
            fdns np.array([steps, Nyl, Nyx, n_dims]): Ground-truh filtered DNS solution 
            time np.array([steps]): Timestamps of recorded data
        # todo: init by loading raw data
        """
        self.rs = torch.from_numpy(rs_tst)
        self.fdns = torch.from_numpy(fdns_tst)
        self.time = torch.from_numpy(time_tst)

    def predict(self, m, it, sol, grid, verbose=False):
        # Lookup closest time in self.time
        """
        time = self.time.detach().cpu().numpy()
        t = np.array([it]).astype(np.float64)
        dt = np.array([time[1]-time[0]]).astype(np.float64)
        if t < (dt/4 + dt):
            t_idx = 0
        else:
            import pdb;pdb.set_trace()
            t_rounded = t - ((t-dt/4)%dt)
            t_idxs = np.where(time.astype(np.float32)==t_rounded.astype(np.float32),1,0)
            t_idx = np.nonzero(t_idxs)[0][0]
        """
        # TODO: Fix issue when qg.nonlinear_les.predict() gets called in qg
        t_idx = m.pde.cur.n
        r = self.rs[t_idx,...]

        qh = sol.clone()
        q = qg.to_physical(qh)
        if verbose:
            print('iter it       \t', t_idx)
            print('vorticity(t): \t', q[:4,0]) # 128, 128
            print('r(it)         \t', r[:4,0]) # 128, 128
            #print('m.pde.cur.n   \t', m.pde.cur.n)
            print('fdns(it)      \t', self.fdns[t_idx,:4,0,0]) # 100, 128, 128, 1
            # print('fdns(t)', self.fdns[it,...].shape)
            print('time(it)      \t', self.time[t_idx])

        return qg.to_spectral(r)