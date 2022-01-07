#include <array>
#include "libnux/vx/v2/dls.h"
#include "libnux/vx/v2/mailbox.h"
#include "libnux/vx/v2/spr.h"
#include "libnux/vx/v2/time.h"
#include "libnux/vx/v2/vector.h"

using namespace libnux::vx::v2;

volatile size_t ppu_id = 0;
volatile size_t n_ppus = 1;

volatile size_t n_samples = 20;
volatile size_t batch_offset = 0;
volatile size_t duration;

enum Command {RUN, NONE, HALT, RESET_BATCH, RUN_AND_RESET};
volatile Command command = NONE;

#define SYN_DEBUG_CUT 0
#define SYN_DEBUG_EXC 0x1
#define SYN_DEBUG_INH 0x2
#define SYN_DEBUG_BOTH 0x3

inline void reset() {
    asm volatile(
        // attach synapse debug line
        "fxvoutx %[attach], %[debug_switch_base], %[index_0]\n"  
        "fxvoutx %[attach], %[debug_switch_base], %[index_1]\n"  
        "fxvoutx %[zero], %[neuron_reset_base], %[index_odd]\n"
        // trigger neuron reset
        :
        : [detach] "qv" (vec_splat_u8(SYN_DEBUG_CUT)),
          [attach] "qv" (vec_splat_u8(SYN_DEBUG_BOTH)),
          [zero] "qv" (vec_splat_u8(0)),
          [debug_switch_base] "b" (dls_config_odd_base),
          [neuron_reset_base] "b" (dls_neuron_reset_base),       
          [index_0] "r" (512 + 0),
          [index_1] "r" (512 + 1),
          [index_odd] "r" (1)
        :
    );
   
    sleep_cycles(550);
   
    asm volatile(
        // detach synapse debug line
        "fxvoutx %[detach], %[debug_switch_base], %[index_0]\n"  
        "fxvoutx %[detach], %[debug_switch_base], %[index_1]\n"  
        :
        : [detach] "qv" (vec_splat_u8(SYN_DEBUG_CUT)),
          [attach] "qv" (vec_splat_u8(SYN_DEBUG_BOTH)),
          [zero] "qv" (vec_splat_u8(0)),
          [debug_switch_base] "b" (dls_config_odd_base),
          [index_0] "r" (512 + 0),
          [index_1] "r" (512 + 1)
        :
    );
}

void cadc_sampling_fast(size_t n_samples, size_t offset, bool trigger_reset) {
    uint32_t row = 0;
    for(size_t sample=0; sample < n_samples; ++sample){
        asm volatile(
            "fxvinx %1, %[cadc_trigger_read], %[row_part_0]\n"
            "fxvoutx %1, %[external_base], %[sample_index]\n"
            :
            : [cadc_trigger_read] "b" (dls_causal_base),
              [external_base] "b" (dls_extmem_base),
              [row_part_0] "r" (2*row + 0),
              [sample_index] "r" ((n_ppus*offset*n_samples + n_ppus*sample + (n_ppus - 1 - ppu_id)) * 128 / 16)
            :
        );
    }
 
    if(trigger_reset) reset();

    asm volatile("sync");
}

int start(void) {
    time_base_t start;

    while(command != HALT){
        if(command == RESET_BATCH) {
	    batch_offset = 0;
	}
	else if(command == RUN) {
            command = NONE;
            start = get_time_base();
	    cadc_sampling_fast(n_samples, batch_offset, false);
	    batch_offset++;
            duration = get_time_base() - start;
        }
	else if(command == RUN_AND_RESET) {
            command = NONE;
            start = get_time_base();
	    cadc_sampling_fast(n_samples, batch_offset, true);
	    batch_offset++;
            duration = get_time_base() - start;
	}
    }
    
    return 0;
}
