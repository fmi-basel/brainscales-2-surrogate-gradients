from typing import Dict

import numpy as np
import torch

from .base import StrobeLayer
from .activations import SuperSpike
from .unterjubel import unterjubel


class LILayer(StrobeLayer):
    def __init__(self, size: int, params: Dict) -> None:
        super(LILayer, self).__init__()
        self.size = size
        self.params = params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_steps = x.shape[1]
        if self.on_hx and not self.training:
            return self.traces
        elif not self.on_hx:
            # when running without hardware, we can't derive time step from membrane samples
            # unfortunately, only quick fix.
            self.time_step = 1.7e-6
            
            self.traces = torch.empty((x.shape[0], n_steps, self.size), device=x.device)
            self.traces[:, 0, :] = 0

        currents = torch.empty_like(self.traces, device=x.device)
        currents[:, 0, :] = 0

        assert (self.traces[:, 0, :] == 0).all()

        alpha = np.exp(-self.time_step/self.params["tau_syn"])
        beta = np.exp(-self.time_step/self.params["tau_mem"])

        for t in range(n_steps - 1):
            currents[:, t+1, :] = alpha*currents[:, t, :] + x[:, t]
            membrane_model = beta*self.traces[:, t, :] + currents[:, t+1, :]
            self.traces[:, t+1, :] = unterjubel(membrane_model, self.traces[:, t+1, :], self.on_hx)

        return self.traces


class LIFLayer(StrobeLayer):
    def __init__(
            self,
            size: int,
            params: Dict,
            activation: torch.autograd.Function = SuperSpike,
            activation_kwargs: Dict = {}) -> None:
        """
        A feedforward layer if leaky integrate-and-fire neurons.

        :param size: Size of the layer.
        :param activation: The activation function to be used in the forward and backward pass.
        :param activation_kwargs: Parameters to be given to the activation function.
        """

        super(LIFLayer, self).__init__()
        self.size = size
        self.params = params

        self.activation_function = activation.apply
        self.activation_args = activation.process_arguments(activation_kwargs)

    def spike(self, *args):
        """
        This function wraps the activation function supplied as an argument to the class
        and inserts any arguments given for that activation function.
        """

        return self.activation_function(*args, *self.activation_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of a feedforward layer of leaky integrate-and-fire neurons.
        Takes a dense tensor of spikes as input and again returns a tensor of spikes.

        :param x: Input spikes as a dense tensor of shape `(batch_size, time_steps, input_units)`.
        """

        n_steps = x.shape[1]

        # check if we need the forward intergration (i.e. for backward or if not on hardware)
        if self.on_hx and not self.training:
            return self.spikes
        elif not self.on_hx:
            # when running without hardware, we can't derive time step from membrane samples
            # unfortunately, only quick fix.
            self.time_step = 1.7e-6
            
            # initialize trace and spike data structures if they were not populated by a hardware run
            self.traces = torch.empty((x.shape[0], n_steps, self.size), device=x.device)
            self.traces[:, 0, :] = 0

            self.spikes = torch.zeros_like(self.traces, device=x.device)

        currents = torch.empty_like(self.traces, device=x.device)
        currents[:, 0, :] = 0

        # all units should have a zeroed first timestep
        assert (self.traces[:, 0, :] == 0).all()

        # calculate synaptic and membrane decay factors
        alpha = np.exp(-self.time_step/self.params["tau_syn"])
        beta = np.exp(-self.time_step/self.params["tau_mem"])

        # Euler integration
        for t in range(n_steps - 1):
            # update synaptic currents
            currents[:, t+1, :] = alpha*currents[:, t, :] + x[:, t, :]

            # calculate membrane update
            model_membrane = beta*self.traces[:, t, :] + currents[:, t+1, :]
            if self.on_hx:
                self.traces[:, t+1, :] = unterjubel(model_membrane, self.traces[:, t+1, :])
            else:
                self.traces[:, t+1, :] = model_membrane

            # apply reset in case we do not operate on measured traces
            if not self.on_hx:
                spike_mask = self.spikes[:, t, :] == 1
                self.traces[:, t+1, :][spike_mask] -= 1.0

            # calculate/apply spikes
            model_spikes = self.spike(self.traces[:, t+1, :] - 1.0)
            if self.on_hx:
                self.spikes[:, t+1, :] = unterjubel(model_spikes, self.spikes[:, t+1, :])
            else:
                self.spikes[:, t+1, :] = model_spikes

        return self.spikes


class RecurrentLIFLayer(StrobeLayer):
    def __init__(
            self,
            size: int,
            params: Dict,
            recurrent_projection: StrobeLayer,
            activation: torch.autograd.Function = SuperSpike,
            activation_kwargs: Dict = {}) -> None:
        """
        A feedforward layer if leaky integrate-and-fire neurons.

        :param size: Size of the layer.
        :param recurrent_projection: Weight layer of shape `(size, size)` for recurrency.
        :param activation: The activation function to be used in the forward and backward pass.
        :param activation_kwargs: Parameters to be given to the activation function.
        """

        super(RecurrentLIFLayer, self).__init__()
        self.size = size
        self.params = params
        self.recurrent_projection = recurrent_projection

        self.activation_function = activation.apply
        self.activation_args = activation.process_arguments(activation_kwargs)

    def spike(self, *args):
        """
        This function wraps the activation function supplied as an argument to the class
        and inserts any arguments given for that activation function.
        """

        return self.activation_function(*args, *self.activation_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of a layer of leaky integrate-and-fire neurons.
        Takes a dense tensor of spikes as input and again returns a tensor of spikes.

        :param x: Input spikes as a dense tensor of shape `(batch_size, time_steps, input_units)`.
        """

        n_steps = x.shape[1]

        # check if we need the forward intergration (i.e. for backward or if not on hardware)
        if self.on_hx and not self.training:
            return self.spikes
        elif not self.on_hx:
            # when running without hardware, we can't derive time step from membrane samples
            # unfortunately, only quick fix.
            self.time_step = 1.7e-6
            
            # initialize trace and spike data structures if they were not populated by a hardware run
            self.traces = torch.empty((x.shape[0], n_steps, self.size), device=x.device)
            self.traces[:, 0, :] = 0

            self.spikes = torch.zeros_like(self.traces, device=x.device)

        # all units should have a zeroed first timestep
        assert (self.traces[:, 0, :] == 0).all()

        # calculate synaptic and membrane decay factors
        alpha = np.exp(-self.time_step/self.params["tau_syn"])
        beta = np.exp(-self.time_step/self.params["tau_mem"])

        spikes = [self.spikes[:, 0]]
        current = 0
        for t in range(1, n_steps):
            current = alpha*current + x[:, t - 1, :]
            current += self.recurrent_projection(spikes[-1])
            model_trace = beta*self.traces[:, t - 1, :] + current
            if self.on_hx:
                self.traces[:, t, :] = unterjubel(model_trace, self.traces[:, t, :])
            else:
                self.traces[:, t, :] = model_trace

            # apply reset in case we do not operate on measured traces
            if not self.on_hx:
                spike_mask = self.spikes[:, t-1, :] == 1
                self.traces[:, t, :][spike_mask] -= 1.0

            # calculate/apply spikes
            model_spikes = self.spike(self.traces[:, t, :] - 1.0)
            if self.on_hx:
                s = unterjubel(model_spikes, self.spikes[:, t, :])
            else:
                s = model_spikes
            spikes.append(s)
        self.spikes = torch.stack(spikes).permute(1, 0, 2)

        return self.spikes
