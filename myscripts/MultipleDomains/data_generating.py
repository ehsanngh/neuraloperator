"""
Creates multiple rectangle domains (various L1 and L2 values) and returns
the solution to equation:
ddu/dx1^2 + ddu/dx2^2 - 6 = 0
u_exact = 1 + x1^2 + 2*x2^2
"""
import torch
from utilities import GridEmbedding2D

def u_exact(x):
    return 1 + x[:, 0:1] ** 2 + 2 * x[:, 1:2] ** 2

def src_func(x):
    return torch.tensor(-6.)

def create_drchltbool(coor, drchltBCs):
    drchlt_bool = torch.zeros_like(coor[:, 0:1], dtype=torch.float)
        
    drchltBCx1 = drchltBCs['x1']
    drchltBCx1 = [torch.tensor(t, dtype=torch.float) for t in drchltBCx1]
    mask = torch.any(torch.isclose(coor[:, 0:1].view((-1, 1)), torch.stack(drchltBCx1)), dim=1).view(drchlt_bool.shape)
    drchlt_bool[mask] = 1.
        
    drchltBCx2 = drchltBCs['x2']
    drchltBCx2 = [torch.tensor(t, dtype=torch.float) for t in drchltBCx2]
    mask = torch.any(torch.isclose(coor[:, 1:2].view((-1, 1)), torch.stack(drchltBCx2)), dim=1).view(drchlt_bool.shape)
    drchlt_bool[mask] = 1.
    return drchlt_bool

def create_one_domain(L1=1., L2=1., resolution=21,
                      src_func: callable=src_func,
                      target_func: callable=u_exact,
                      requires_grad: bool=False):
    grid_boundaries = [[0., L1], [0., L2]]
    pos_encoding = GridEmbedding2D(grid_boundaries=grid_boundaries)
    init_src_field = torch.ones([1, 1, resolution, resolution])
    x = pos_encoding(init_src_field)
    x[:, [0, 1, 2], :, :] = x[:, [1, 2, 0], :, :]
    if src_func is not None:
        x[:, 2:] = src_func(x[:, :2])
    else:
        x = x[:, :2]
    u = target_func(x)
    drichltBCs = {'x1': tuple({0., L1}), 'x2': tuple({0., L2})}
    drchltbool = create_drchltbool(x[:, :2], drichltBCs)
    x = torch.cat((x, drchltbool), axis=1)
    x.requires_grad = requires_grad
    return x, u

def create_dataset(num_samples=25, resolution=21, seed=0, grad=False):
    torch.manual_seed(seed)
    L1s = 20 * torch.rand(num_samples).numpy()
    L2s = 20 * torch.rand(num_samples).numpy()
    data = [create_one_domain(L1=L1s[i], L2=L2s[i], resolution=resolution,
                              src_func=None,
                              target_func=u_exact,
                              requires_grad=grad) for i in range(num_samples)]
    X, Y = zip(*data)
    X = torch.cat(X, axis=0)
    Y = torch.cat(Y, axis=0)
    return X, Y