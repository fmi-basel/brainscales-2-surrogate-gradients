import numpy as np

import pyhaldls_vx_v2 as haldls
import pylola_vx_v2 as lola
import pystadls_vx_v2 as stadls
import pyfisch_vx as fisch
import pyhalco_hicann_dls_vx_v2 as halco
import gonzales


class RoutingGenerator(stadls.PlaybackGenerator):
    def __init__(self, neuron_size=1, signed_synapses=False):
        super().__init__()

        self._neuron_size = neuron_size
        self._signed_synapses = signed_synapses

        sources = np.arange(1024)
        blocks = (sources // (32 // self._neuron_size)) % 8
        sources_on_block = (sources - blocks * (32 // self._neuron_size)) % 256
        self._neuron_buses = blocks % 4
        unshifted_addresses = (sources_on_block % (32 // self._neuron_size) + (32 // self._neuron_size)
                               * ((sources % 256) // (128 // self._neuron_size))) % 256
        self._neuron_addresses = ((unshifted_addresses >> 2) + ((unshifted_addresses & 0b11) << 4)) << 2
        self._neuron_addresses += 1*(sources // 256)

        self._neuron_lookup = np.zeros((self._neuron_addresses.max() + 1, 4), dtype=int)
        for a in np.unique(self._neuron_addresses):
            for b in np.unique(self._neuron_buses):
                self._neuron_lookup[a, b] = np.where((self._neuron_addresses == a) & (self._neuron_buses == b))[0][0]

        # specification of synapse row assignment
        rows = np.arange(256)
        drivers = rows // 2
        odds = rows % 2

        padi_buses = drivers % 4

        self._driver_masks = (drivers // 4) % 4
        labels = (2*(drivers // 16)) << 2

        if not signed_synapses:
            labels += odds << 2

        new = ((labels >> 5) & 0b1) << 2
        new += ((labels >> 4) & 0b1) << 3
        new += ((labels >> 3) & 0b1) << 4
        new += ((labels >> 2) & 0b1) << 5

        self._synapse_labels = new

        addresses = (self._driver_masks << 6) + self._synapse_labels

        self._lookup = np.empty_like(addresses)
        for i, (address, bus) in enumerate(zip(addresses, padi_buses)):
            self._lookup[i] = np.where((self._neuron_addresses == address) & (self._neuron_buses == bus))[0][0]

        ########################################
        # neuron backends                      #
        ########################################
        self.neuron_backend_configs = dict([
            (c, haldls.NeuronBackendConfig()) for c in halco.iter_all(halco.NeuronBackendConfigOnDLS)])
        for c in halco.iter_all(halco.NeuronBackendConfigOnDLS):
            config = self.neuron_backend_configs[c]

            index = int(c.toAtomicNeuronOnDLS().toEnum())
            source = index // self._neuron_size

            # look up address from specification above
            config.address_out = int(self._neuron_addresses[source])

            if int(c.toEnum()) % self._neuron_size == 0:
                config.enable_spike_out = True
            else:
                config.enable_spike_out = False

        ########################################
        # routing crossbar                     #
        ########################################
        active_crossbar_node = haldls.CrossbarNode()
        active_crossbar_node.mask = 0
        active_crossbar_node.target = 0

        silent_crossbar_node = haldls.CrossbarNode()
        silent_crossbar_node.mask = 0
        silent_crossbar_node.target = 2**14 - 1

        # initialize all nodes to be silent
        self.crossbar_nodes = dict([
            (c, haldls.CrossbarNode(silent_crossbar_node)) for c in halco.iter_all(halco.CrossbarNodeOnDLS)])

        # enable recurrent connections
        for i in range(8):
            self.crossbar_nodes[
                halco.CrossbarNodeOnDLS(
                    halco.CrossbarOutputOnDLS(i % 4),
                    halco.CrossbarInputOnDLS(i)
                )] = active_crossbar_node
            self.crossbar_nodes[
                halco.CrossbarNodeOnDLS(
                    halco.CrossbarOutputOnDLS(4 + (i % 4)),
                    halco.CrossbarInputOnDLS(i)
                )] = active_crossbar_node

        # enable external spike input
        for o in range(8):
            self.crossbar_nodes[
                halco.CrossbarNodeOnDLS(
                    halco.CrossbarOutputOnDLS(o),
                    halco.CrossbarInputOnDLS(8 + (o % 4))
                )] = active_crossbar_node

        # enable spike output
        for i in range(8):
            self.crossbar_nodes[
                halco.CrossbarNodeOnDLS(
                    halco.CrossbarOutputOnDLS(8 + i % 4),
                    halco.CrossbarInputOnDLS(i)
                )] = active_crossbar_node

        ########################################
        # PADI bus config                      #
        ########################################
        padi_config = haldls.CommonPADIBusConfig()
        for bus in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_config.enable_spl1[bus] = True
            padi_config.dacen_pulse_extension[bus] = 0

        self.common_padi_bus_configs = dict([
            (c, haldls.CommonPADIBusConfig(padi_config)) for c in halco.iter_all(halco.CommonPADIBusConfigOnDLS)])

        ########################################
        # synapse drivers                      #
        ########################################
        driver_config = haldls.SynapseDriverConfig()
        driver_config.enable_receiver = True
        driver_config.row_address_compare_mask = 0b00011
        driver_config.enable_address_out = True
        if self._signed_synapses:
            driver_config.row_mode_top = haldls.SynapseDriverConfig.RowMode.inhibitory
            driver_config.row_mode_bottom = haldls.SynapseDriverConfig.RowMode.excitatory
        else:
            driver_config.row_mode_top = haldls.SynapseDriverConfig.RowMode.excitatory
            driver_config.row_mode_bottom = haldls.SynapseDriverConfig.RowMode.excitatory

        self.synapse_driver_configs = dict([
            (c, haldls.SynapseDriverConfig(driver_config)) for c in halco.iter_all(halco.SynapseDriverOnDLS)])

        ########################################
        # synapse current switches             #
        ########################################
        current_quad = haldls.ColumnCurrentQuad()
        switch = current_quad.ColumnCurrentSwitch()
        switch.enable_synaptic_current_excitatory = True
        switch.enable_synaptic_current_inhibitory = True
        for s in halco.iter_all(halco.EntryOnQuad):
            current_quad.set_switch(s, switch)

        self.column_current_quads = dict([
            (c, haldls.ColumnCurrentQuad(current_quad)) for c in halco.iter_all(halco.ColumnCurrentQuadOnDLS)])

    def generate(self):
        builder = stadls.PlaybackProgramBuilder()

        for coord, config in self.neuron_backend_configs.items():
            builder.write(coord, config)

        for coord, config in self.crossbar_nodes.items():
            builder.write(coord, config)

        for coord, config in self.common_padi_bus_configs.items():
            builder.write(coord, config)

        for coord, config in self.synapse_driver_configs.items():
            builder.write(coord, config)

        for coord, config in self.column_current_quads.items():
            builder.write(coord, config)

        return builder

    def transform_weights(self, weights, sources=None):
        if len(weights.shape) < 3:
            weights = weights.reshape((1, ) + weights.shape)
        if len(sources.shape) < 3:
            sources = sources.reshape((1, ) + sources.shape)

        shape = (self._neuron_size,
                 halco.SynapseRowOnSynram.size // (self._signed_synapses + 1),
                 halco.NeuronConfigOnDLS.size // self._neuron_size)

        if sources is None:
            sources = np.zeros(shape, dtype=np.uint8) + 2

        assert weights.shape == shape
        assert sources.shape == shape

        label_matrix = np.tile(self._synapse_labels, (halco.NeuronColumnOnDLS.size, 1)).T

        sources_flat = np.swapaxes(sources, 0, 1).reshape(shape[1], shape[0] * shape[2], order="F")
        weights_flat = np.swapaxes(weights, 0, 1).reshape(shape[1], shape[0] * shape[2], order="F")
        weights_assigned = weights_flat[self._lookup, :]

        weights_assigned[0::2, :] = +np.clip(weights_assigned[0::2, :], 0, 63)
        weights_assigned[1::2, :] = -np.clip(weights_assigned[1::2, :], -63, 0)

        label_matrix_top = label_matrix + sources_flat[self._lookup, 0:256]
        label_matrix_bottom = label_matrix + sources_flat[self._lookup, 256:512]

        synapse_matrix_top = lola.SynapseMatrix()
        synapse_matrix_top.labels.from_numpy(label_matrix_top)
        synapse_matrix_top.weights.from_numpy(weights_assigned[:, 0:256])

        synapse_matrix_bottom = lola.SynapseMatrix()
        synapse_matrix_bottom.labels.from_numpy(label_matrix_bottom)
        synapse_matrix_bottom.weights.from_numpy(weights_assigned[:, 256:512])

        return synapse_matrix_top, synapse_matrix_bottom

    def transform_events_from_chip(self, spikes):
        spike_times = spikes["chip_time"] / fisch.fpga_clock_cycles_per_us * 1e-6
        spike_labels = spikes["label"]

        event_outputs = (spike_labels >> 8) & 0b11
        event_addresses = spike_labels & 0b11111111

        sources = self._neuron_lookup[event_addresses, event_outputs]

        return spike_times, sources

    def generate_spike_train(self, times, sources):
        builder = stadls.PlaybackProgramBuilder()
        neuron_labels = self._neuron_addresses[sources & 0xff] + (sources >> 8)
        spl1_addresses = self._neuron_buses[sources & 0xff]
        gonzales.generate_spiketrain(builder, times, neuron_labels, spl1_addresses)

        return builder
