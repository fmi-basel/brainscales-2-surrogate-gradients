#include <array>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "fisch/vx/constants.h"
#include "fisch/vx/playback_program_builder.h"
#include "haldls/vx/timer.h"
#include "haldls/vx/v2/neuron.h"
#include "stadls/vx/v2/init_generator.h"
#include "stadls/vx/v2/playback_program.h"
#include "stadls/vx/v2/playback_program_builder.h"

using namespace halco::common;
using namespace halco::hicann_dls::vx::v2;
using namespace haldls::vx::v2;
using namespace stadls::vx::v2;

namespace py = pybind11;

void generate_spiketrain(
	PlaybackProgramBuilder& builder,
	py::array_t<double> times,
	py::array_t<uint32_t> neuron_labels,
	py::array_t<uint32_t> spl1_addresses
	) {

	for(size_t i=0; i<times.size(); ++i) {
                builder.wait_until(
			TimerOnDLS(),
			Timer::Value(times.at(i) * 1e6 * fisch::vx::fpga_clock_cycles_per_us));
		
		std::array<SpikeLabel, 1> labels;
		labels.at(0).set_neuron_label(NeuronLabel(neuron_labels.at(i)));
		labels.at(0).set_spl1_address(SPL1Address(spl1_addresses.at(i)));

		SpikePack1ToChip pack;
		pack.set_labels(labels);
		builder.write(SpikePack1ToChipOnDLS(), pack);
	}
}

py::array_t<uint8_t> parse_ppu_memory_u8(PPUMemoryBlock const& memory_block) {
	py::array_t<uint8_t> data(memory_block.size() * 4);
	py::buffer_info buf = data.request();
	uint8_t *ptr = (uint8_t *) buf.ptr;
	for(size_t i=0; i<memory_block.size(); ++i) {
		uint32_t word = memory_block.at(i).get_value();
		ptr[4*i + 3] = word & (0xff << 0);
		ptr[4*i + 2] = (word & (0xff << 8)) >> 8;
		ptr[4*i + 1] = (word & (0xff << 16)) >> 16;
		ptr[4*i + 0] = (word & (0xff << 24)) >> 24;
	}
	return data;
}

typedef fisch::vx::ContainerTicket<fisch::vx::Omnibus> fpga_memory_ticket_type;
typedef std::vector<halco::hicann_dls::vx::OmnibusAddress> fpga_addresses_type;

constexpr uint32_t fpga_memory_base_address{0x8e00'0000};
fpga_memory_ticket_type get_fpga_memory_ticket(
		PlaybackProgramBuilder& builder,
		size_t n_vectors
		)
{
	fpga_addresses_type addresses;
	addresses.reserve(n_vectors * 128 / 4);

	for(size_t i=0; i < n_vectors * 128 / 4; ++i)
	{
		addresses.push_back(
			halco::hicann_dls::vx::OmnibusAddress(fpga_memory_base_address + i));
	}
	
	fisch::vx::PlaybackProgramBuilder fbuilder;
	auto ticket = fbuilder.read(addresses);
	builder.merge_back(fbuilder);
	return ticket;
}

py::array_t<uint8_t> parse_fpga_memory_u8(fpga_memory_ticket_type const& ticket) {
	auto words = ticket.get();
	py::array_t<uint8_t> data(words.size() * 4);
	py::buffer_info buf = data.request();
	uint8_t *ptr = (uint8_t *) buf.ptr;
	uint8_t tmp = 0;
	for(size_t i=0; i<words.size(); ++i) {
		uint32_t word = words.at(i).get();
		for(size_t j=0; j<4; ++j) {
			tmp = (word & (0xff << (8*j))) >> (8*j);
			ptr[4*i + j] = tmp;
		}
	}
	return data;
}

PYBIND11_MODULE(gonzales, m) {
	py::module::import("pystadls_vx_v2");
	m.def("generate_spiketrain", &generate_spiketrain, "Generate a playback program builder for inserting spikes.");
	m.def("parse_ppu_memory_u8", &parse_ppu_memory_u8, "Parse PPUMemoryBlock into individual words of type uint8_t.");
	m.def("parse_fpga_memory_u8", &parse_fpga_memory_u8, "Parse FPGA memoryPPUMemoryBlock into individual words of type uint8_t.");
	m.def("get_fpga_memory_ticket", &get_fpga_memory_ticket, "Parse FPGA memoryPPUMemoryBlock into individual words of type uint8_t.");
}
