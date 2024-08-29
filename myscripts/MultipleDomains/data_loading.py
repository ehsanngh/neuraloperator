import torch
# torch.set_default_dtype(torch.float64)

from neuralop.data.datasets.tensor_dataset import TensorDataset
from data_generating import create_dataset
from data_processing import CustomDataProcessor, UnitGaussianNormalizer

def load_dataset(
    n_trains=[25],
    n_tests=[25],
    train_resolutions=[32],
    test_resolutions=[32],
    train_batch_sizes=[1],
    test_batch_sizes=[1],
    encode_input=True,
    encode_output=True,
    encoding="channel-wise",
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
            X_train, Y_train = create_dataset(num_samples=n_trains[i],
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

    X_test, Y_test = create_dataset(num_samples=n_trains[0],
                                    resolution=train_resolutions[0],
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
            X_test, Y_test = create_dataset(num_samples=n_tests[i],
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
    
    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(X_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        # input_encoder = RangeNormalizer()
        input_encoder.fit(X_train)
    else:
        input_encoder = None

    if encode_output:
        if encoding == "channel-wise":
            reduce_dims = list(range(Y_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        # output_encoder = RangeNormalizer()
        output_encoder.fit(Y_train)
    else:
        output_encoder = None

    data_processor = CustomDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
        positional_encoding=None
    )
    return train_loaders, test_loaders, data_processor
