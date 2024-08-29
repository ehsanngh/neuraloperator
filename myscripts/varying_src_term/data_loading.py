import torch
# torch.set_default_dtype(torch.float64)

from neuralop.data.datasets.tensor_dataset import TensorDataset
from data_generating import create_dataset, create_dataset_graph
from data_processing import UnitGaussianNormalizer, RangeNormalizer
from data_processing import CustomDataProcessor, CustomDataProcessorGraph


DataScaler = UnitGaussianNormalizer  # RangeNormalizer

import torch_geometric

def load_dataset(
    n_trains=[25],
    n_tests=[25],
    train_resolutions=[32],
    test_resolutions=[32],
    train_batch_sizes=[1],
    test_batch_sizes=[1],
    grad=False,
    seed=0
):
    X_train, Y_train = create_dataset(num_samples=n_trains[0],
                                      resolution=train_resolutions[0],
                                      grad=grad, seed=seed)
    train_db = TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=train_batch_sizes[0],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    train_loaders = {train_resolutions[0]: train_loader}
    if len(train_resolutions) > 1:
        for i in range(1,len(train_resolutions)):
            X_train, Y_train = create_dataset(
                num_samples=n_trains[i],
                resolution=train_resolutions[i],
                grad=grad, seed=seed+3*i)
            train_db = TensorDataset(X_train, Y_train)
            train_loader = torch.utils.data.DataLoader(
                train_db,
                batch_size=train_batch_sizes[i],
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
            )
            train_loaders[train_resolutions[i]] = train_loader

    X_test, Y_test = create_dataset(num_samples=n_tests[0],
                                    resolution=test_resolutions[0],
                                    grad=grad, seed=seed+1)
    test_db = TensorDataset(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=test_batch_sizes[0],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loaders = {test_resolutions[0]: test_loader}
    if len(test_resolutions) > 1:
        for i in range(1,len(test_resolutions)):
            X_test, Y_test = create_dataset(
                num_samples=n_tests[i],
                resolution=test_resolutions[i],
                grad=grad, seed=seed+2*i)
            test_db = TensorDataset(X_test, Y_test)
            test_loader = torch.utils.data.DataLoader(
                test_db,
                batch_size=test_batch_sizes[i],
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
            )
            test_loaders[test_resolutions[i]] = test_loader

    input_encoder = DataScaler(dim=[0, 2, 3])
    input_encoder.fit(X_train)

    output_encoder = DataScaler(dim=[0, 2, 3])
    output_encoder.fit(Y_train)

    data_processor = CustomDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder
    )
    return train_loaders, test_loaders, data_processor


def load_dataset_graph(
    n_trains=[25],
    n_tests=[25],
    train_resolutions=[32],
    test_resolutions=[32],
    train_batch_sizes=[1],
    test_batch_sizes=[1],
    grad=False,
    seed=0
):
    graph_data = create_dataset_graph(num_samples=n_trains[0],
                                      resolution=train_resolutions[0],
                                      grad=grad, seed=seed)
    
    x = torch.cat([data.x for data in graph_data], dim=0) 
    pos = torch.cat([data.pos for data in graph_data], dim=0)
    y = torch.cat([data.y for data in graph_data], dim=0)
    edge_attr = torch.cat([data.edge_attr for data in graph_data], dim=0)
    
    input_encoder = DataScaler(dim=[0])
    input_encoder.fit(x)

    pos_encoder = DataScaler(dim=[0])
    pos_encoder.fit(pos)

    edge_attr_encoder = DataScaler(dim=[0])
    edge_attr_encoder.fit(edge_attr)

    output_encoder = DataScaler(dim=[0])
    output_encoder.fit(y)

    train_loader = torch_geometric.loader.DataLoader(
        graph_data, batch_size=train_batch_sizes[0], shuffle=True)
    train_loaders = {train_resolutions[0]: train_loader}
    if len(train_resolutions) > 1:
        for i in range(1,len(train_resolutions)):
            graph_data = create_dataset_graph(
                num_samples=n_trains[0],
                resolution=train_resolutions[0],
                grad=grad, seed=seed)
            train_loader = torch_geometric.loader.DataLoader(
                graph_data, batch_size=train_batch_sizes[i], shuffle=True)
            train_loaders[train_resolutions[i]] = train_loader

    graph_data = create_dataset_graph(
        num_samples=n_tests[0],
        resolution=test_resolutions[0],
        grad=grad, seed=seed+1)
    test_loader = torch_geometric.loader.DataLoader(
        graph_data, batch_size=test_batch_sizes[0], shuffle=False)
    test_loaders = {test_resolutions[0]: test_loader}
    if len(test_resolutions) > 1:
        for i in range(1,len(test_resolutions)):
            graph_data = create_dataset_graph(
                num_samples=n_tests[i],
                resolution=test_resolutions[i],
                grad=grad, seed=seed+2*i)
            test_loader = torch_geometric.loader.DataLoader(
                graph_data, batch_size=test_batch_sizes[i], shuffle=False)
            test_loaders[test_resolutions[i]] = test_loader

    data_processor = CustomDataProcessorGraph(
        in_normalizer=input_encoder,
        edge_attr_normalizer=edge_attr_encoder,
        pos_normalizer=pos_encoder,
        out_normalizer=output_encoder

    )
    return train_loaders, test_loaders, data_processor

