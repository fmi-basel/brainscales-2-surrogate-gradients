from typing import Dict

import torch


class StrobeLayer(torch.nn.Module):
    def __init__(self):
        super(StrobeLayer, self).__init__()
        self.on_hx = False

    def inject(
            self,
            spikes: torch.Tensor = None,
            traces: torch.Tensor = None,
            calibration_targets: Dict = None,
            time_step: float = None):
        self.on_hx = True

        self.spikes = spikes
        self.traces = traces
        self.time_step = time_step

        if calibration_targets["tau_mem"] != self.params["tau_mem"]:
            raise ValueError(f"Neuron parameter 'tau_mem' does not match calibration target ({calibration_targets['tau_mem']})!")

        if calibration_targets["tau_syn"] != self.params["tau_syn"]:
            raise ValueError(f"Neuron parameter 'tau_syn' does not match calibration target ({calibration_targets['tau_syn']})!")
