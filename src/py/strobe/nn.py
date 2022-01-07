import os.path
import numpy as np
import torch

try:
    import pyhxcomm_vx as hxcomm
except ImportError:
    print("Cannot import BrainScaleS-2 software stack. Please make sure to run in software-only mode.")

from .base import StrobeLayer
from .projections import Linear, SigmoidalWeights, Dropout
from .lif import LILayer, LIFLayer, RecurrentLIFLayer


class Network(torch.nn.Sequential):
    def __init__(
            self,
            *layers: StrobeLayer,
            interpolation: int = 1):
        """
        A network of sequential layers of spiking neurons, trained with the STROBE framework.

        :param layers: Layers of the network.
        :param interpolation: Interpolate between CADC samples to achieve a finer grained integration.
            In case of hardware-less this simply results in a smaller integration time step.
        """

        super(Network, self).__init__(*layers)

        for name, layer in self.named_children():
            if not isinstance(layer, StrobeLayer):
                raise TypeError("The layer you are trying to register is not a StrobeLayer. Good luck!")

        self.backend = None
        self.neuron_parameters = None

        # parameters for hardware execution
        self._interpolation = interpolation
        self.time_step = 1.7e-6 / self._interpolation
        self._trace_offset = 0.38
        self._trace_scale = 1 / 0.33

        # alignment of traces and spikes from chip
        self._spike_shift = 1.7e-6 / self._interpolation
        self._weights = []

        self._record_madc = False

    def connect(
            self,
            connection,
            calibration: str = None,
            synapse_bias: int = 1000,
            sample_separation: float = 500e-6,
            inference_mode: bool = False):
        """
        A network of sequential layers of spiking neurons, trained with the STROBE framework.

        :param calibration: Path to the calibration file generated via calix.
        :param synapse_bias: Bias setting for the synapse circuits to module their overall strength.
        :param sample_separation: Separation of samples in a harware batch.
        """

        self.inference_mode = inference_mode

        from .backend import StrobeBackend, LayerSize, FPGA_MEMORY_SIZE

        self.fpga_memory_size = FPGA_MEMORY_SIZE
        self.neuron_parameters = np.load(calibration, allow_pickle=True)["targets"].item()

        weight_layers, self.neuron_layers = self.squash()
        structure = [weight_layers[0].shape[1]]
        if isinstance(self.neuron_layers[0], RecurrentLIFLayer):
            structure[0] -= self.neuron_layers[0].size
        for w, n in zip(weight_layers, self.neuron_layers):
            recurrent = isinstance(n, RecurrentLIFLayer)
            spiking = not isinstance(n, LILayer)
            structure.append(LayerSize(w.shape[0], recurrent=recurrent, spiking=spiking))

        self.backend = StrobeBackend(connection, structure, calibration, synapse_bias, sample_separation)

        self.backend.configure()
        self.backend.load_ppu_program(os.path.join(os.path.dirname(__file__), "../../bin/strobe.bin"))

    def squash(self):
        weight_layers = []
        neuron_layers = []

        assembly = None
        for l, layer in enumerate(self):
            if isinstance(layer, (Linear, SigmoidalWeights)):
                w = layer.weight.cpu() * layer.scale
                if assembly is None:
                    assembly = w
                else:
                    assembly = torch.matmul(w, assembly)
            if isinstance(layer, Dropout):
                if assembly is None:
                    raise TypeError("Currently, dropout layers are only allowed to be placed after a weight layer.")
                assembly = assembly * layer.mask[:, np.newaxis]
            elif isinstance(layer, (LILayer, LIFLayer, RecurrentLIFLayer)):
                if assembly is None:
                    raise TypeError("Your model must always interleave (multiple) weights and neuronal layers!")

                if layer.size != assembly.shape[0]:
                    raise ValueError(f"Shape mismatch for layers {l - 1} and {l}.")

                if isinstance(layer, RecurrentLIFLayer):
                    assembly = torch.cat([
                            assembly,
                            layer.recurrent_projection.weight.cpu() * layer.recurrent_projection.scale
                            ], dim=1)

                weight_layers.append(assembly)
                neuron_layers.append(layer)
                assembly = None

        return weight_layers, neuron_layers

    def synchronize_hardware(self, force=False):
        weights, _ = self.squash()
        weights = [np.round(w.T.detach().numpy().copy()) for w in weights]

        update = force
        update = update or len(self._weights) != len(weights)
        if not update:
            for old, new in zip(self._weights, weights):
                update = update or (old != new).any()

        if update:
            self._weights = weights
            self.backend.write_weights(*weights)

    def forward(self, x):
        if self.backend is not None:
            # extract number of samples from input tensor
            batch_size = x.shape[0]
            n_steps = x.shape[1]

            # calculate maximum batch size where traces fit into FPGA memory
            max_hw_batch_size = int(np.floor(self.fpga_memory_size / n_steps / self.backend._n_vectors / 128))
            hw_batch_size = min(batch_size, max_hw_batch_size)

            self.synchronize_hardware()

            self.batch_durations = np.zeros((batch_size, 2))

            layered_traces = []
            layered_spikes = []
            for l, layer in enumerate(self.neuron_layers):
                layered_traces.append(torch.zeros((batch_size, n_steps, layer.size), device=x.device))
                layered_spikes.append(torch.zeros((batch_size, n_steps, layer.size), device=x.device))

            hw_batch_bounds = np.arange(0, batch_size, hw_batch_size)
            for s in [slice(i, min(batch_size + 1, i + hw_batch_size)) for i in hw_batch_bounds]:
                hw_x = x[s, :, :]

                input_spikes = []
                for b in range(hw_x.shape[0]):
                    spike_bins = np.where(hw_x[b].T.cpu())
                    labels = spike_bins[0] + 256
                    times = spike_bins[1].astype(np.float) * self.time_step + self._spike_shift

                    # sort spike train according to injection times
                    order = np.argsort(times)
                    input_spikes.append(np.vstack([times[order], labels[order]]).T)

                for trial in range(5):
                    spikes, traces, durations = self.backend.run(
                            input_spikes,
                            n_samples=n_steps // self._interpolation,
                            record_madc=self._record_madc,
                            trigger_reset=self.inference_mode)
                    self.batch_durations[s, :] = np.array(durations)
                    if not (np.array(durations) > 85200).any():
                        break
                    else:
                        pass

                # normalize membrane traces
                for t in traces:
                    t -= self._trace_offset
                    t *= self._trace_scale
                    t -= t[:, 0, None]

                for l, layer in enumerate(self.neuron_layers):
                    for i in range(self._interpolation):
                        layered_traces[l][s, i::self._interpolation, :] = torch.from_numpy(traces[l])

                for b in range(hw_x.shape[0]):
                    for l, layer in enumerate(self.neuron_layers):
                        if spikes[l][b].size:
                            hist = np.zeros((n_steps, layer.size))
                            spike_times = spikes[l][b][:, 0] - self._spike_shift
                            mask = spike_times < self.time_step * n_steps
                            units = spikes[l][b][:, 1].astype(np.int)
                            hist[(spike_times[mask] // self.time_step).astype(np.int), units[mask]] = 1
                            layered_spikes[l][s, :, :][b, :, :] = torch.from_numpy(hist)

            for l, layer in enumerate(self.neuron_layers):
                layer.inject(layered_spikes[l], layered_traces[l], self.neuron_parameters, self.time_step)

        # execute forward paths of child layers
        y = torch.nn.Sequential.forward(self, x)
        return y
