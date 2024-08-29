from neuralop.data.transforms.data_processors import DataProcessor
import torch
from neuralop.data.transforms.base_transforms import Transform
from neuralop.utils import count_tensor_params
    
class RangeNormalizer(Transform):
    """
    RangeNormalizer scales data to a specified range [low, high].
    """

    def __init__(self, low=0.0, high=1.0):
        super().__init__()
        self.low = low
        self.high = high
        self.register_buffer("a", None)
        self.register_buffer("b", None)

    def fit(self, data):
        min_val = torch.min(data, dim=0)[0].view(-1)
        max_val = torch.max(data, dim=0)[0].view(-1)
        self.a = (self.high - self.low) / (max_val - min_val)
        self.b = -self.a * max_val + self.high

    def transform(self, data):
        s = data.size()
        data = data.view(s[0], -1)
        data = self.a * data + self.b
        return data.view(s)

    def inverse_transform(self, data):
        s = data.size()
        data = data.view(s[0], -1)
        data = (data - self.b) / self.a
        return data.view(s)

    def forward(self, data):
        return self.transform(data)

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()
        return self

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()
        return self

    def to(self, device):
        self.a = self.a.to(device)
        self.b = self.b.to(device)
        return self
    

class UnitGaussianNormalizer(Transform):
    """
    UnitGaussianNormalizer normalizes data to be zero mean and unit std.
    """

    def __init__(self, mean=None, std=None, eps=1e-7, dim=None, mask=None):
        """
        mean : torch.tensor or None
            has to include batch-size as a dim of 1
            e.g. for tensors of shape ``(batch_size, channels, height, width)``,
            the mean over height and width should have shape ``(1, channels, 1, 1)``
        std : torch.tensor or None
        eps : float, default is 0
            for safe division by the std
        dim : int list, default is None
            if not None, dimensions of the data to reduce over to compute the mean and std.

            .. important::

                Has to include the batch-size (typically 0).
                For instance, to normalize data of shape ``(batch_size, channels, height, width)``
                along batch-size, height and width, pass ``dim=[0, 2, 3]``

        mask : torch.Tensor or None, default is None
            If not None, a tensor with the same size as a sample,
            with value 0 where the data should be ignored and 1 everywhere else

        Notes
        -----
        The resulting mean will have the same size as the input MINUS the specified dims.
        If you do not specify any dims, the mean and std will both be scalars.

        Returns
        -------
        UnitGaussianNormalizer instance
        """
        super().__init__()

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("mask", mask)

        self.eps = eps
        if mean is not None:
            self.ndim = mean.ndim
        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim
        self.n_elements = 0

    def fit(self, data_batch):
        self.update_mean_std(data_batch)

    def partial_fit(self, data_batch, batch_size=1):
        if 0 in list(data_batch.shape):
            return
        count = 0
        n_samples = len(data_batch)
        while count < n_samples:
            samples = data_batch[count : count + batch_size]
            # print(samples.shape)
            # if batch_size == 1:
            #     samples = samples.unsqueeze(0)
            if self.n_elements:
                self.incremental_update_mean_std(samples)
            else:
                self.update_mean_std(samples)
            count += batch_size

    def update_mean_std(self, data_batch):
        self.ndim = data_batch.ndim  # Note this includes batch-size
        if self.mask is None:
            self.n_elements = count_tensor_params(data_batch, self.dim)
            self.mean = torch.mean(data_batch, dim=self.dim, keepdim=True).detach()
            self.squared_mean = torch.mean(data_batch**2, dim=self.dim, keepdim=True).detach()
            self.std = torch.std(data_batch, dim=self.dim, keepdim=True).detach()
        else:
            batch_size = data_batch.shape[0]
            dim = [i - 1 for i in self.dim if i]
            shape = [s for i, s in enumerate(self.mask.shape) if i not in dim]
            self.n_elements = torch.count_nonzero(self.mask, dim=dim) * batch_size
            self.mean = torch.zeros(shape)
            self.std = torch.zeros(shape)
            self.squared_mean = torch.zeros(shape)
            data_batch[:, self.mask == 1] = 0
            self.mean[self.mask == 1] = (
                torch.sum(data_batch, dim=dim, keepdim=True).detach() / self.n_elements
            )
            self.squared_mean = (
                torch.sum(data_batch**2, dim=dim, keepdim=True).detach() / self.n_elements
            )
            self.std = torch.std(data_batch, dim=self.dim, keepdim=True).detach()

    def incremental_update_mean_std(self, data_batch):
        if self.mask is None:
            n_elements = count_tensor_params(data_batch, self.dim)
            dim = self.dim
        else:
            dim = [i - 1 for i in self.dim if i]
            n_elements = torch.count_nonzero(self.mask, dim=dim) * data_batch.shape[0]
            data_batch[:, self.mask == 1] = 0

        self.mean = (1.0 / (self.n_elements + n_elements)) * (
            self.n_elements * self.mean + torch.sum(data_batch, dim=dim, keepdim=True).detach()
        )
        self.squared_mean = (1.0 / (self.n_elements + n_elements)) * (
            self.n_elements * self.squared_mean
            + torch.sum(data_batch**2, dim=dim, keepdim=True).detach()
        )
        self.n_elements += n_elements
        self.std = torch.sqrt(self.squared_mean - self.mean**2) * self.n_elements / (self.n_elements - 1)

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x):
        return x * (self.std + self.eps) + self.mean

    def forward(self, x):
        return self.transform(x)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class CustomDataProcessor(DataProcessor):
    def __init__(
        self, in_normalizer=None, out_normalizer=None, positional_encoding=None
    ):
        """A simple processor to pre/post process data before training/inferencing a model.

        Parameters
        ----------
        in_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the input samples
        out_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the target and predicted samples
        positional_encoding : Processor, optional, default is None
            class that appends a positional encoding to the input
        """
        super().__init__()
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.positional_encoding = positional_encoding
        self.device = "cpu"
        self.model = None

    def to(self, device):
        if self.in_normalizer is not None:
            self.in_normalizer = self.in_normalizer.to(device)
        if self.out_normalizer is not None:
            self.out_normalizer = self.out_normalizer.to(device)
        self.device = device
        return self

    def preprocess(self, data_dict, batched=True):
        data_dict = data_dict.copy()
        x = data_dict["x"].to(self.device)
        y = data_dict["y"].to(self.device)

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x, batched=batched)
        if self.out_normalizer is not None: # and self.training:
            y = self.out_normalizer.transform(y)

        data_dict["x"] = x
        data_dict["y"] = y

        return data_dict

    def postprocess(self, output, data_dict):
        data_dict = data_dict.copy()
        x = data_dict["x"].to(self.device)
        y = data_dict["y"].to(self.device)
        if self.in_normalizer is not None:
            x = self.in_normalizer.inverse_transform(x)
        if self.out_normalizer is not None: #  and not self.training:
            output = self.out_normalizer.inverse_transform(output)
            y = self.out_normalizer.inverse_transform(y)
        data_dict["x"] = x
        data_dict["y"] = y
        output = output * (1 - x[:, -1:]) + y * x[:, -1:]
        return output, data_dict

    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(data_dict["x"])
        output = self.postprocess(output)
        return output, data_dict