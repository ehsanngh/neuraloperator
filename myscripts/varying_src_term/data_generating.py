"""
Creating data instances for various rectangle domains (various L1 and L2 values)
to solve the following PDE:
ddu/dx1^2 + ddu/dx2^2 - (2 * x2 + 4 * x1) = 0
u_exact = 1 + x1^2 * x2 + 2 * x1 * x2^2
"""
import torch
from utilities import GridEmbedding2D

from torch_geometric.data import Data

def u_exact(x):
    return 1 + x[:, 0:1] * x[:, 1:2] * (x[:, 0:1] + 2 * x[:, 1:2])

def src_func(x):
    return -(4 * x[:, 0:1] + 2 * x[:, 1:2])

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

def calculate_edge_ftrs(pos, edge_index):

        if len(pos.shape) != 2:
            num_graphs, _, n1, n2 = pos.shape
            pos = pos.view(num_graphs, 2, -1).permute(0, 2, 1).reshape(-1, 2)
        # pos = torch.vstack((pos[0, 0, :, :].flatten(), pos[0, 1, :, :].flatten())).T
        diffs = pos[edge_index[1]] - pos[edge_index[0]]
        distances = torch.linalg.norm(diffs, dim=1, keepdim=True)
        return torch.concat((diffs / distances, distances), axis=1)

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
    y = target_func(x)
    drichltBCs = {'x1': tuple({0., L1}), 'x2': tuple({0., L2})}
    drchltbool = create_drchltbool(x[:, :2], drichltBCs)
    x = torch.cat((x, drchltbool), axis=1)
    x.requires_grad = requires_grad
    return x, y

def create_one_domain_graph(L1=1., L2=1., resolution=21,
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
    y = target_func(x)
    drichltBCs = {'x1': tuple({0., L1}), 'x2': tuple({0., L2})}
    drchltbool = create_drchltbool(x[:, :2], drichltBCs)
    x = torch.cat((x, drchltbool), axis=1)

    edges = []
    n1 = len(x[0, 0, :, 0])
    n2 = len(x[0, 0, 0, :])
    for i in range(n1):
        for j in range(n2):
            index = i * n2 + j
            if j != n2 - 1:
                edges.append((index, index + 1))
                edges.append((index + 1, index))
            if i != n1 - 1:
                edges.append((index, index + n2))
                edges.append((index + n2, index))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = calculate_edge_ftrs(x[:, :2], edge_index)
    x.requires_grad = requires_grad

    _, num_channels, _, _ = x.shape
    x = x.permute(0, 2, 3, 1).contiguous().view(-1, num_channels)
    y = y.permute(0, 2, 3, 1).contiguous().view(-1, 1)
    graph = Data(x=x[:, 2:],
                 edge_index=edge_index,
                 pos=x[:, :2],
                 edge_attr=edge_attr,
                 y = y)

    return graph

def create_dataset(num_samples=25, resolution=21, seed=0, grad=False):
    torch.manual_seed(seed)
    L1s = 20 * torch.rand(num_samples).numpy()
    L2s = 20 * torch.rand(num_samples).numpy()
    data = [create_one_domain(L1=L1s[i], L2=L2s[i], resolution=resolution,
                              src_func=src_func,
                              target_func=u_exact,
                              requires_grad=grad) for i in range(num_samples)]
    X, Y = zip(*data)
    X = torch.cat(X, axis=0)
    Y = torch.cat(Y, axis=0)
    return X, Y

def create_dataset_graph(num_samples=25, resolution=21, seed=0, grad=False):
    torch.manual_seed(seed)
    L1s = 20 * torch.rand(num_samples).numpy()
    L2s = 20 * torch.rand(num_samples).numpy()
    data = [create_one_domain_graph(L1=L1s[i], L2=L2s[i], resolution=resolution,
                                    src_func=src_func,
                                    target_func=u_exact,
                                    requires_grad=grad) for i in range(num_samples)]
    return data