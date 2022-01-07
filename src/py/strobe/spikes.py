import numpy as np

import torch


class PixelsToSpikeTimes(torch.nn.Module):

    def __init__(self, tau=20, threshold=0.2, t_max=1.0, epsilon=1e-7):
        super().__init__()

        self._tau = tau
        self._threshold = threshold
        self._t_max = t_max
        self._epsilon = epsilon

    def forward(self, x):
        device = x.device
        x = x[:, 0]  # we only use the first (in most cases only) color channel
        non_spiking = x < self._threshold

        # TODO: move this to torch

        x = np.clip(x.cpu(), self._threshold + self._epsilon, 1e9)
        times = self._tau * np.log(x / (x - self._threshold))

        times[non_spiking] = self._t_max
        times = torch.clamp(times, 0, self._t_max)

        times = times.to(device)

        return times


class SpikeTimesToDense(torch.nn.Module):
    """Convert spike times to a dense matrix of zeros and ones."""

    def __init__(self, time_step, size=None):
        """Initialize the conversion of spike times to a dense matrix of zeros and ones.

        time_step -- binning interval in seconds
        size      -- number of bins along time axis (calculate from data, if `size is None`)
        """

        super().__init__()

        self._time_step = time_step
        self._size = size

    def forward(self, x):
        """Convert spike times to dense matrix of zeros and ones.

        x -- spike times of shape `(batch_size, x0[, x1, …])`.

        Returns a dense matrix of shape `(batch_size, n_time_steps, x0[, x1, …])`.
        """

        bins = (x / self._time_step).long()

        if self._size is not None:
            n_time_steps = self._size
            n_time_steps_tmp = max(self._size, int(bins.max()) + 1)
        else:
            n_time_steps = n_time_steps_tmp = int(bins.max()) + 1

        dense = torch.zeros((x.shape[0], n_time_steps_tmp, *x.shape[1:]), device=x.device)
        mesh = torch.meshgrid([torch.arange(s) for s in x.shape])
        slc = (mesh[0], ) + (bins, ) + mesh[1:]
        dense[slc] = 1
        return dense[:, :n_time_steps]
