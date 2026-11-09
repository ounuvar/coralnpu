// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Smoke test for the AWS F2 custom logic: console output on UART1, then word, halfword and byte
// accesses to DDR at offsets that land in both 256-bit halves of a 512-bit DDR beat. The
// testbench (fpga/f2/tb) reads f2_result over SPI and checks the DDR contents directly.

#include <stdint.h>

#include "fpga/sw/uart.h"

#define DDR_BASE 0x80000000u
#define F2_PASS  0x600DF00Du

volatile uint32_t f2_result = 0;

// Word offsets into DDR: both halves of the first 64-byte line, the next lines, and 4 KB.
static const uint32_t kWordOffsets[] = {0, 1, 7, 8, 9, 15, 16, 31, 1024};
#define NUM_WORDS (sizeof(kWordOffsets) / sizeof(kWordOffsets[0]))

static uint32_t pattern(uint32_t i) { return 0xA5000000u ^ (i * 0x01010101u); }

int main(void) {
  uart_init();
  uart_puts("F2\r\n");

  volatile uint32_t *ddr32 = (volatile uint32_t *)DDR_BASE;
  volatile uint16_t *ddr16 = (volatile uint16_t *)DDR_BASE;
  volatile uint8_t *ddr8   = (volatile uint8_t *)DDR_BASE;
  uint32_t errors          = 0;

  for (uint32_t i = 0; i < NUM_WORDS; i++)
    ddr32[kWordOffsets[i]] = pattern(i);
  for (uint32_t i = 0; i < NUM_WORDS; i++) {
    if (ddr32[kWordOffsets[i]] != pattern(i))
      errors++;
  }

  // Narrow writes into the upper half of line 0 (byte offset 0x30): 0x30-0x33 = 0x11223344.
  ddr16[0x30 / 2] = 0x3344;
  ddr8[0x32]      = 0x22;
  ddr8[0x33]      = 0x11;
  if (ddr32[0x30 / 4] != 0x11223344u)
    errors++;
  if (ddr8[0x31] != 0x33)
    errors++;

  if (errors == 0) {
    f2_result = F2_PASS;
    uart_puts("PASS\r\n");
  } else {
    f2_result = 0xBAD00000u | errors;
    uart_puts("FAIL\r\n");
  }
  return 0;
}
