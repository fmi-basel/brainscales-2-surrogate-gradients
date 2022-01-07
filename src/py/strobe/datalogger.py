import numpy as np
import h5py

class DataLogger:
    def __init__(self, *dims, initializer=np.zeros):
        self.dims = dims

        self.data = {}
        self._averaging_axes = {}

        self._initializer = initializer

        # optional arguments passed to HDF5 when creating the dataset
        self._args = {}

    def store(self, key, data, *indices, average=None, h5_args=dict()):
        if len(indices) > len(self.dims):
            raise IndexError("The number of indices exceeds the configured dimensionalilty.")

        if isinstance(data, np.ndarray):
            shape = data.shape
            dtype = data.dtype
        else:
            shape = ()
            dtype = type(data)

        if key not in self.data:
            self.data[key] = self._initializer(self.dims[:len(indices)] + shape, dtype=dtype)
        if key not in self._averaging_axes:
            self._averaging_axes[key] = average

        self.data[key][tuple(indices)] = data
        self._args[key] = h5_args

    def dump(self, target=None):
        processed = {}
        for key in self.data.keys():
            data = self.data[key]
            averaging_axes = self._averaging_axes[key]
            if averaging_axes is None:
                processed[key] = data
            else:
                processed[key] = np.mean(data, averaging_axes)

        if target is None:
            return processed
        elif isinstance(target, h5py.Group):
            for key, value in processed.items():
                target.create_dataset(key, data=value, **self._args[key])


if __name__ == "__main__":
    n_epochs = 5
    n_batches = 3

    data_logger = DataLogger(n_epochs, n_batches)

    for e in range(n_epochs):
        for b in range(n_batches):
            data_logger.store("loss", 1, e, b, average=(1))

    print(data_logger.data)
    print(data_logger.dump())
    f = h5py.File("/tmp/asd.h5", "w")
    data_logger.dump(f)
