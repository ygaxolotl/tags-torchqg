import math
import os
import tqdm
import h5py

import torch
import numpy as np

from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torchqg.qg as qg

plt.rcParams.update({'mathtext.fontset':'cm'})
plt.rcParams.update({'xtick.minor.visible':True})
plt.rcParams.update({'ytick.minor.visible':True})

def workflow(
  dir,
  name,
  iters, 
  steps,
  scale,
  diags,
  system,
  models,
  dump=False,
):
  """
  Solves the equation and stores the results

  Args:
    dir string: Relative path for storing data and figures
    name:
    iters int: Maximum number of LES iterations
    steps int: Total number of steps that will be stored
    scale int: Scale of coarse-graining kernel (DNS -> LES), e.g., 4
    diags [fn]: List of evaluation, diagnostics, plotting functions
    system qg.QgModel: DNS model
    models [qg.QgModel,...]: LES models with different parametrizations
  """
  t0 = system.pde.cur.t
  store_les = int(iters / steps) # Data will be stored every <store_les> time steps
  store_dns = store_les * scale
  
  Nx = system.grid.Nx
  Ny = system.grid.Ny
  Nxl = int(Nx / scale)
  Nyl = int(Ny / scale)

  if models:
    sgs_grid = models[-1].grid

  ## Init data loggers
  # Filtered DNS
  fdns = torch.zeros([steps, 5, Nyl, Nxl], dtype=torch.float64)
  # DNS
  dns  = torch.zeros([steps, 4, Ny,  Nx ], dtype=torch.float64)
  # LES
  les = {}
  for m in models:
    les[m.name] = torch.zeros([steps, 5, Nyl, Nxl], dtype=torch.float64)
  # Time
  time = torch.zeros([steps])

  def visitor_dns(m, cur, it):
    """
    Logs DNS and filtered DNS data
    """
    # High res
    if it % store_dns == 0:
      i = int(it / store_dns)
      q, p, u, v = m.update()

      # Calculate filtered DNS from DNS; see Frezat et al., eq. (16)
      if models:
        r = m.R(sgs_grid, scale) # Exact ground-truth SGS param # (128,33)
        fdns[i, 0] = qg.to_physical(r) # used for ML train set
        fdns[i, 1] = m.filter_physical(sgs_grid, scale, q).view(1, Nyl, Nxl) # used for ML train and testy set
        fdns[i, 2] = m.filter_physical(sgs_grid, scale, p).view(1, Nyl, Nxl)
        fdns[i, 3] = m.filter_physical(sgs_grid, scale, u).view(1, Nyl, Nxl)
        fdns[i, 4] = m.filter_physical(sgs_grid, scale, v).view(1, Nyl, Nxl)
        #print(f'Ground-truth fdns at iter {cur.n} and t={cur.t}, but iter it={it} and index {i}')
        #print('vorticity(t): \t', fdns[i,1,:4,0])
        #print('r(it)         \t', fdns[i,0,:4,0])

      dns[i] = torch.stack((q, p, u, v))

      # step time
      time[i] = cur.t - t0
    return None

  def visitor_les(m, cur, it):
    # Low res
    if it % store_les == 0:
      i = int(it / store_les)
      q, p, u, v = m.update()

      # Predicted sgs
      if m.sgs:
        # print(f'SGS {type(m.sgs).__name__} at iter {m.pde.cur.n} and t={m.pde.cur.t}, but iter i={i}')
        r = m.sgs.predict(m, it, m.pde.sol, m.grid, verbose=True)
      else:
        # r = torch.zeros([Nyl, Nxl], dtype=torch.float64)
        r = torch.zeros(sgs_grid.irsq.shape, dtype=torch.complex128)
      les[m.name][i] = torch.stack((qg.to_physical(r), q, p, u, v))
    return None
 
  if not os.path.exists(dir):
    os.mkdir(dir)

  # Integrate the dynamics.
  with torch.no_grad():
    for it in tqdm.tqdm(range(iters * scale)):
      system.pde.step(system)
      # Step dns on every iter, but only store every <store_dns - 4> iters
      visitor_dns(system, system.pde.cur, it)
      for m in models:
        # Step les only every <scale - 4> iters, but store every <store_les - 1> step
        if it % scale == 0: # Only steps LES every <scale> iters
          m.pde.step(m)
          visitor_les(m, m.pde.cur, it / scale)

    for diag in diags:
      diag(
        dir, 
        name, 
        scale, 
        time, 
        system, 
        models, 
        dns=dns, 
        fdns=fdns, 
        les=les
      )
    if dump:
      hf = h5py.File(os.path.join(dir, name + '_dump.h5'), 'w')
      hf.create_dataset('time', data=time.detach().numpy())
      # Store DNS data for initializing val model
      hf.create_dataset('dns_q', data=dns[:, 0].detach().numpy())
      # Store filtered DNS
      hf.create_dataset(system.name + '_r', data=fdns[:, 0].detach().numpy()) # used for ML train set
      hf.create_dataset(system.name + '_q', data=fdns[:, 1].detach().numpy()) # used for ML train and test set
      hf.create_dataset(system.name + '_p', data=fdns[:, 2].detach().numpy())
      hf.create_dataset(system.name + '_u', data=fdns[:, 3].detach().numpy())
      hf.create_dataset(system.name + '_v', data=fdns[:, 4].detach().numpy())
      for m in models:
        # Store LES models
        hf.create_dataset(m.name + '_r', data=les[m.name][:, 0].detach().numpy())
        hf.create_dataset(m.name + '_q', data=les[m.name][:, 1].detach().numpy())
        hf.create_dataset(m.name + '_p', data=les[m.name][:, 2].detach().numpy())
        hf.create_dataset(m.name + '_u', data=les[m.name][:, 3].detach().numpy())
        hf.create_dataset(m.name + '_v', data=les[m.name][:, 4].detach().numpy())
      hf.close()

def diag_pred_vs_target_param(dir, name, scale, time, system, models, dns, fdns, les):
  """
  Creates 2D plot with input (vorticity) and ground truth parametrization vs. predicted params
  Args:
    dir str: Directory to save plot
    name str: Filename of plot
    scale: --
    time: --
    system: DNS (not used here)
    models: list of LES model objects
    dns: --
    fdns: ground-truth filtered DNS data
    les: predicted LES data
  """

  cols = len(models) + 3
  rows = 1
  width_ratios = np.concatenate((np.array([1, 0.1]), np.repeat(rows, cols-2), np.array([0.1])))
  
  m_fig, m_axs = plt.subplots(
    nrows=rows,
    ncols=cols + 1,
    figsize=((cols-1) * 2.4 + 2*0.5, rows * 2.5),
    constrained_layout=True,
    gridspec_kw={"width_ratios": width_ratios} # np.append(np.repeat(rows, cols), 0.1)}
  )
  r_idx = 0
  q_idx = 1
  last_tstep = les[models[0].name].shape[0]-1
  eval_tstep =  last_tstep # 0 # Time step that will be plotted

  # Plot large-scale vorticity, which is used as input
  data = les[models[0].name][eval_tstep, q_idx]
  c0 = m_axs[0].contourf(models[0].grid.x.cpu().detach().numpy(), models[0].grid.y.cpu().detach().numpy(), data.cpu().detach().numpy(), cmap='bwr', levels=100) # LR vorticity
  m_fig.colorbar(c0,cax=m_axs[1])
  m_axs[0].set_ylabel(r'$\omega$', fontsize=20)

  # Plot parametrizations
  def plot_field(i, label, grid, data, span_r):
    x = grid.x.cpu().detach().numpy()
    y = grid.y.cpu().detach().numpy()
    data_np = data.cpu().detach().numpy()

    c0 = m_axs[i].contourf(x, y, data_np, vmax=span_r, vmin=-span_r, cmap='bwr', levels=100) # parametrization

    if i == 2:
      m_fig.colorbar(c0,cax=m_axs[-1])
    m_axs[i].set_xlabel(label, fontsize=20)

  ## Init min, max ranges
  span_r = max(fdns[eval_tstep, r_idx].max(), abs(fdns[eval_tstep, r_idx].min())).cpu().detach().numpy()
  # Projected DNS
  plot_field(2, r'$\overline{\mathcal{M}' + system.name + '}$', models[-1].grid, fdns[eval_tstep, r_idx], span_r)
  # LES
  rmse = {}
  for i, m in enumerate(models):
    data = les[m.name][eval_tstep, r_idx]
    # Calculate RMSE
    rmse[m.name] = torch.mean((data - fdns[eval_tstep, r_idx]) ** 2)
    
    plot_field(i + 3, r'$\mathcal{M}_{' + m.name + '}$', m.grid, data, span_r)

  print('RMSE of predicted vs. ground-truth parametrization:')
  pprint(rmse)

  filepath = os.path.join(dir, name + '_params.png')
  m_fig.savefig(filepath, dpi=300)
  plt.show()
  plt.close(m_fig)

  print('Plotted ground truth vs. target parametrization at: ', filepath)

def diag_pred_vs_target_sol_err(dir, name, scale, time, system, models, dns, fdns, les):
  """
  Creates 2D plot with input (vorticity at t), ground truth vorticity at t+1, and 
    error of LES with null param, and LES with predicted solution
  Args:
    dir str: Directory to save plot
    name str: Filename of plot
    scale: --
    time: --
    system: DNS (not used here)
    models: list of LES model objects
    dns: --
    fdns: ground-truth filtered DNS data
    les: predicted LES data
  """
  cols = len(models)*2 + 2
  rows = 1
  width_ratios = np.concatenate((np.array([1, 0.1]), np.array([1., 0.1]*len(models))))

  m_fig, m_axs = plt.subplots(
    nrows=rows,
    ncols=cols,
    figsize=(cols * 1.5, rows * 2.5),
    constrained_layout=True,
    gridspec_kw={"width_ratios": width_ratios} # np.append(np.repeat(rows, cols), 0.1)}
  )
  q_idx = 1 # index of vorticity
  last_tstep = les[models[0].name].shape[0]-1
  eval_tstep = last_tstep - 1 # 0 # Time step that will be plotted

  # Plot large-scale vorticity, which is used as input
  data = les[models[0].name][eval_tstep, q_idx]
  c0 = m_axs[0].contourf(models[0].grid.x.cpu().detach().numpy(), models[0].grid.y.cpu().detach().numpy(), data.cpu().detach().numpy(), cmap='bwr', levels=100) # LR vorticity
  m_fig.colorbar(c0,cax=m_axs[1])
  m_axs[0].set_xlabel(r'input: $\omega$', fontsize=20)

  # Plot predicted vorticities
  def plot_field(i, label, grid, data, span_qerr):
    x = grid.x.cpu().detach().numpy()
    y = grid.y.cpu().detach().numpy()
    data_np = data.cpu().detach().numpy()
    c0 = m_axs[i].contourf(x, y, data_np, cmap='bwr', levels=100) # vmax=span_qerr, vmin=-span_qerr, )
    m_fig.colorbar(c0,cax=m_axs[i+1])
    m_axs[i].set_xlabel(label, fontsize=20)

  ## Init min, max ranges with range of null parametrization error. Assumes null parametrization is first input
  # null_qerr = les[models[0].name][eval_tstep+1, q_idx] - fdns[eval_tstep+1, q_idx]
  # span_qerr = max(null_qerr.max(), abs(null_qerr.min())).cpu().detach().numpy()
  span_qerr = None

  print('fdns, les', fdns.shape, les[models[0].name].shape)
  import pdb;pdb.set_trace()
  # Predicted LES at next time step
  rmse = {}
  for i, m in enumerate(models):
    data = les[m.name][eval_tstep+1, q_idx]
    err = data - fdns[eval_tstep+1, q_idx]
    # Calculate RMSE
    rmse[m.name] = torch.mean(err ** 2)
    
    plot_field(i*2 + 2, r'err: $\mathcal{M}_{' + m.name + '}$', m.grid, err, span_qerr)

  print('RMSE of predicted - ground-truth solution:')
  pprint(rmse)

  filepath = os.path.join(dir, name + '_sol_err.png')
  m_fig.savefig(filepath, dpi=300)
  plt.show()
  plt.close(m_fig)

  print('Plotted ground truth vs. target solution at: ', filepath)

def diag_fields(dir, name, scale, time, system, models, dns, fdns, les):
  # Plotting

  ## Create 2D plot of DNS vorticity, streamfunction, vel_x and vel_y at the final timestep
  cols = 1
  rows = 4
  m_fig, m_axs = plt.subplots(
    nrows=rows,
    ncols=cols + 1,
    figsize=(cols * 2.5 + 0.5, rows * 2.5),
    constrained_layout=True,
    gridspec_kw={"width_ratios": np.append(np.repeat(rows, cols), 0.1)}
  )

  # DNS
  m_fig.colorbar(m_axs[0, 0].contourf(system.grid.x.cpu().detach(), system.grid.y.cpu().detach(), dns[-1, 0], cmap='bwr', levels=100), cax=m_axs[0, 1])
  m_fig.colorbar(m_axs[1, 0].contourf(system.grid.x.cpu().detach(), system.grid.y.cpu().detach(), dns[-1, 1], cmap='bwr', levels=100), cax=m_axs[1, 1])
  m_fig.colorbar(m_axs[2, 0].contourf(system.grid.x.cpu().detach(), system.grid.y.cpu().detach(), dns[-1, 2], cmap='bwr', levels=100), cax=m_axs[2, 1])
  m_fig.colorbar(m_axs[3, 0].contourf(system.grid.x.cpu().detach(), system.grid.y.cpu().detach(), dns[-1, 3], cmap='bwr', levels=100), cax=m_axs[3, 1])
  
  m_axs[0, 0].set_ylabel(r'$\omega$', fontsize=20)
  m_axs[1, 0].set_ylabel(r'$\psi$', fontsize=20)
  m_axs[2, 0].set_ylabel(r'$u_{x}$', fontsize=20)
  m_axs[3, 0].set_ylabel(r'$u_{y}$', fontsize=20)
  m_axs[3, 0].set_xlabel(r'$\mathcal{M}' + system.name + '$', fontsize=20)

  m_fig.savefig(os.path.join(dir, name + '_dns.png'), dpi=300)
  plt.show()
  plt.close(m_fig)

  if not models:
    return

  ## Create 2D plot of DNS vs. Parametrizations, plotting vorticity, streamfunction, vel_x, vel_y, and parametrization at the final timestep
  cols = len(models) + 1
  rows = 5
  #rep = torch.repeat_interleave(torch.Tensor([rows]), cols)
  #width_ratios = torch.cat((rep, torch.Tensor([0.1])))
  width_ratios = np.append(np.repeat(rows, cols), 0.1)
  m_fig, m_axs = plt.subplots(
    nrows=rows,
    ncols=cols + 1,
    figsize=(cols * 2.5 + 0.5, rows * 2.5),
    constrained_layout=True,
    gridspec_kw={"width_ratios": width_ratios} # np.append(np.repeat(rows, cols), 0.1)}
  )
  # m_fig, m_axs = plt.subplots(nrows=rows,ncols=cols + 1,figsize=(cols * 2.5 + 0.5, rows * 2.5),constrained_layout=True,gridspec_kw={"width_ratios": width_ratios})

  span_r = max(fdns[-1, 0].max(), abs(fdns[-1, 0].min())).cpu().detach().numpy()
  span_q = max(fdns[-1, 1].max(), abs(fdns[-1, 1].min())).cpu().detach().numpy()
  span_p = max(fdns[-1, 2].max(), abs(fdns[-1, 2].min())).cpu().detach().numpy()
  span_u = max(fdns[-1, 3].max(), abs(fdns[-1, 3].min())).cpu().detach().numpy()
  span_v = max(fdns[-1, 4].max(), abs(fdns[-1, 4].min())).cpu().detach().numpy()

  def plot_fields(i, label, grid, data):
    x = grid.x.cpu().detach().numpy()
    y = grid.y.cpu().detach().numpy()
    data_np = data.cpu().detach().numpy()

    c0 = m_axs[0, i].contourf(x, y, data_np[-1, 1], vmax=span_q, vmin=-span_q, cmap='bwr', levels=100) # vorticity
    c1 = m_axs[1, i].contourf(x, y, data_np[-1, 2], vmax=span_p, vmin=-span_p, cmap='bwr', levels=100) # streamfunction
    c2 = m_axs[2, i].contourf(x, y, data_np[-1, 3], vmax=span_u, vmin=-span_u, cmap='bwr', levels=100) # vel x
    c3 = m_axs[3, i].contourf(x, y, data_np[-1, 4], vmax=span_v, vmin=-span_v, cmap='bwr', levels=100) # vel y
    c4 = m_axs[4, i].contourf(x, y, data_np[-1, 0], vmax=span_r, vmin=-span_r, cmap='bwr', levels=100) # parametrization
    if i == 0:
      m_fig.colorbar(c0, cax=m_axs[0, cols])
      m_fig.colorbar(c1, cax=m_axs[1, cols])
      m_fig.colorbar(c2, cax=m_axs[2, cols])
      m_fig.colorbar(c3, cax=m_axs[3, cols])
      m_fig.colorbar(c4, cax=m_axs[4, cols])
    m_axs[4, i].set_xlabel(label, fontsize=20)
  
  # Projected DNS
  plot_fields(0, r'$\overline{\mathcal{M}' + system.name + '}$', models[-1].grid, fdns)
  # LES
  for i, m in enumerate(models):
    data = les[m.name]
    plot_fields(i + 1, r'$\mathcal{M}_{' + m.name + '}$', m.grid, data)

  m_axs[0, 0].set_ylabel(r'$\omega$', fontsize=20)
  m_axs[1, 0].set_ylabel(r'$\psi$', fontsize=20)
  m_axs[2, 0].set_ylabel(r'$u_{x}$', fontsize=20)
  m_axs[3, 0].set_ylabel(r'$u_{y}$', fontsize=20)
  m_axs[4, 0].set_ylabel(r'$R(q)$', fontsize=20)

  m_fig.savefig(os.path.join(dir, name + '_fields.png'), dpi=300)
  plt.show()
  plt.close(m_fig)

