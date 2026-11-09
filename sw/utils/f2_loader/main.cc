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

// f2_loader: loads and runs programs on the CoralNPU custom logic on AWS F2
// (#15). Same flags and order of operations as //sw/utils/nexus_loader, so
// the same scripts can drive either; the transport is the CL's OCL registers
// instead of an FTDI SPI chip.

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "sw/utils/f2_loader/elf_image.h"
#include "sw/utils/f2_loader/f2_host.h"
#include "sw/utils/f2_loader/f2_regs.h"
#include "sw/utils/f2_loader/ocl_pci.h"
#include "sw/utils/f2_loader/ocl_socket.h"

// Where the CL is (exactly one).
ABSL_FLAG(int, slot, -1, "F2 FPGA slot (needs root and a loaded AFI)");
ABSL_FLAG(std::string, pci_bdf, "", "PCI address of the FPGA's application PF, e.g. 0000:34:00.0");
ABSL_FLAG(std::string, sim_socket, "",
          "Unix socket of cl_coralnpu_core_tb running with +host_socket (local simulation)");

// The nexus_loader flags.
ABSL_FLAG(std::string, load_elf, "", "ELF file to load");
ABSL_FLAG(bool, verify, false, "Verify memory after loading");
ABSL_FLAG(bool, highmem, false, "Use the highmem CSR base (0x200000, the default on F2)");
ABSL_FLAG(bool, lowmem, false, "Use the lowmem CSR base (0x30000)");
ABSL_FLAG(uint32_t, csr_base, f2::kCsrBaseHighmem, "CSR base address for core control");
ABSL_FLAG(uint32_t, read_line, 0xFFFFFFFF, "Address to read a single 128-bit line");
ABSL_FLAG(uint32_t, read_lines_addr, 0xFFFFFFFF, "Base address to read multiple lines");
ABSL_FLAG(uint32_t, read_lines_count, 1, "Number of 128-bit lines to read");
ABSL_FLAG(bool, soft_reset, false, "Reset the SoC through CTRL[0] before anything else");
ABSL_FLAG(std::string, load_data, "", "Binary data file to load");
ABSL_FLAG(uint32_t, load_data_addr, 0, "Address to load binary data");
ABSL_FLAG(uint32_t, read_data_addr, 0xFFFFFFFF, "Address to read binary data from");
ABSL_FLAG(uint32_t, read_data_size, 0, "Size of binary data to read (written raw to stdout)");
ABSL_FLAG(uint32_t, write_word_addr, 0xFFFFFFFF, "Address to write a 32-bit word");
ABSL_FLAG(uint32_t, write_word_val, 0, "Value to write to the word address");
ABSL_FLAG(uint32_t, read_word_addr, 0xFFFFFFFF, "Address to read a 32-bit word");
ABSL_FLAG(bool, start_core, false,
          "Start the core (at --set_entry_point, or the ELF's entry point with --load_elf)");
ABSL_FLAG(uint32_t, set_entry_point, 0xFFFFFFFF, "Entry point address to set");
ABSL_FLAG(double, poll_halt, 0.0, "Timeout in seconds to poll for core halt");
ABSL_FLAG(uint32_t, poll_status_addr, 0xFFFFFFFF, "Address of the status message buffer to poll");
ABSL_FLAG(uint32_t, poll_status_size, 0, "Size of the status message buffer");

// F2 only.
ABSL_FLAG(uint32_t, spi_half, 0, "SPI clock half period in clk_main_a0 cycles (0: keep the CL's)");
ABSL_FLAG(uint32_t, poll_interval_us, 10000, "Time between halt polls");
ABSL_FLAG(bool, console, false,
          "After the other steps, print the UART console as CONSOLE: lines on stdout, waiting "
          "until it stops growing");
ABSL_FLAG(uint32_t, console_idle_us, 200000, "Console quiet time that ends --console");
ABSL_FLAG(double, sim_connect_timeout, 600.0, "Seconds to wait for the testbench socket");

namespace {

using f2::F2Host;

int Fail(F2Host &host, const char *what) {
  fprintf(stderr, "f2_loader: %s failed: %s\n", what, host.error().c_str());
  return 1;
}

std::unique_ptr<f2::Ocl> OpenOcl() {
  std::string error;
  const int slot         = absl::GetFlag(FLAGS_slot);
  std::string bdf        = absl::GetFlag(FLAGS_pci_bdf);
  const std::string sock = absl::GetFlag(FLAGS_sim_socket);
  if ((slot >= 0) + !bdf.empty() + !sock.empty() != 1) {
    fprintf(stderr, "f2_loader: give exactly one of --slot, --pci_bdf, --sim_socket\n");
    return nullptr;
  }
  std::unique_ptr<f2::Ocl> ocl;
  if (!sock.empty()) {
    ocl = f2::SocketOcl::Connect(sock, absl::GetFlag(FLAGS_sim_connect_timeout), &error);
  } else {
    if (slot >= 0)
      bdf = f2::PciOcl::SlotToBdf(slot, &error);
    if (!bdf.empty())
      ocl = f2::PciOcl::Open(bdf, &error);
  }
  if (!ocl)
    fprintf(stderr, "f2_loader: %s\n", error.c_str());
  return ocl;
}

bool ReadFile(const std::string &path, std::vector<uint8_t> *data) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;
  data->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}

bool LoadElf(F2Host &host, const f2::ElfImage &image, bool verify) {
  size_t total     = 0;
  const auto start = std::chrono::steady_clock::now();
  for (const f2::ElfSegment &seg : image.segments) {
    fprintf(stderr, "Loading 0x%08x (%zu bytes)...", seg.addr, seg.data.size());
    if (!host.WriteData(seg.addr, seg.data.data(), seg.data.size())) {
      fprintf(stderr, " failed\n");
      return false;
    }
    fprintf(stderr, " Done.\n");
    total += seg.data.size();
    if (verify) {
      std::vector<uint8_t> back(seg.data.size());
      if (!host.ReadData(seg.addr, back.size(), back.data()))
        return false;
      for (size_t i = 0; i < back.size(); ++i) {
        if (back[i] != seg.data[i]) {
          fprintf(stderr, " Verify FAIL at 0x%08x: 0x%02x, expected 0x%02x\n",
                  seg.addr + static_cast<uint32_t>(i), back[i], seg.data[i]);
          return false;
        }
      }
      fprintf(stderr, " Verify OK\n");
    }
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  fprintf(stderr, "Transfer complete: %zu bytes in %.2fs (%.2f KB/s)\n", total, s,
          s > 0 ? total / 1024.0 / s : 0.0);
  return true;
}

// nexus_loader's --poll_halt, with halt and fault read from the CL's STATUS
// register instead of over SPI.
bool PollHalt(F2Host &host, f2::Ocl &ocl) {
  const double timeout       = absl::GetFlag(FLAGS_poll_halt);
  const uint32_t status_addr = absl::GetFlag(FLAGS_poll_status_addr);
  const uint32_t status_size = absl::GetFlag(FLAGS_poll_status_size);
  const uint32_t interval    = absl::GetFlag(FLAGS_poll_interval_us);
  fprintf(stderr, "Polling for halt (timeout: %.1fs)...\n", timeout);
  // Time is counted in poll intervals, so a simulation (where SleepUs passes
  // simulated time) and F2 use the same timeout.
  const double polls = timeout * 1e6 / (interval ? interval : 1);
  std::string last_status;
  for (double i = 0; i <= polls; ++i) {
    const uint32_t status = host.Status();
    if (status & f2::kStatusFault) {
      fprintf(stderr, "Core halted with a fault.\n");
      return false;
    }
    if (status_addr != 0xFFFFFFFF && status_size > 0) {
      std::vector<uint8_t> buf(status_size);
      if (host.ReadData(status_addr, status_size, buf.data())) {
        std::string s(reinterpret_cast<char *>(buf.data()), status_size);
        s.resize(s.find('\0') == std::string::npos ? s.size() : s.find('\0'));
        if (!s.empty() && s != last_status) {
          fprintf(stderr, "Status: %s\n", s.c_str());
          last_status        = s;
          const uint8_t zero = 0;
          host.WriteData(status_addr, &zero, 1);
        }
      }
    }
    if (status & f2::kStatusHalted) {
      fprintf(stderr, "Core halted.\n");
      return true;
    }
    ocl.SleepUs(interval);
  }
  fprintf(stderr, "Timed out waiting for core to halt.\n");
  return false;
}

// The SoC's UART keeps sending after the core halts, so read until the
// console has been quiet for console_idle_us.
void PrintConsole(F2Host &host, f2::Ocl &ocl) {
  const uint32_t idle = absl::GetFlag(FLAGS_console_idle_us);
  std::string text;
  for (;;) {
    std::string more = host.ReadConsole();
    if (more.empty()) {
      ocl.SleepUs(idle);
      more = host.ReadConsole();
      if (more.empty())
        break;
    }
    text += more;
  }
  size_t pos = 0;
  while (pos < text.size()) {
    size_t nl = text.find('\n', pos);
    if (nl == std::string::npos)
      nl = text.size();
    std::string line = text.substr(pos, nl - pos);
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    printf("CONSOLE: %s\n", line.c_str());
    pos = nl + 1;
  }
}

void PrintLine(uint32_t addr, const uint8_t *line) {
  printf("0x%08x: 0x", addr);
  for (int i = 15; i >= 0; --i)
    printf("%02x", line[i]);
  printf("\n");
}

int Run() {
  std::unique_ptr<f2::Ocl> ocl = OpenOcl();
  if (!ocl)
    return 1;
  F2Host host(ocl.get());

  uint32_t csr_base = absl::GetFlag(FLAGS_csr_base);
  if (absl::GetFlag(FLAGS_highmem))
    csr_base = f2::kCsrBaseHighmem;
  if (absl::GetFlag(FLAGS_lowmem))
    csr_base = 0x30000;

  // nexus_loader's order: reset, load ELF or data, write word, start core,
  // poll halt, then reads. The console comes last.
  if (absl::GetFlag(FLAGS_soft_reset)) {
    fprintf(stderr, "Soft resetting the SoC (CTRL[0])...\n");
    host.SoftReset();
  }
  if (!host.Init())
    return Fail(host, "init");
  if (absl::GetFlag(FLAGS_spi_half) != 0)
    host.SetSpiHalf(absl::GetFlag(FLAGS_spi_half));

  uint32_t entry = absl::GetFlag(FLAGS_set_entry_point);
  if (!absl::GetFlag(FLAGS_load_elf).empty()) {
    f2::ElfImage image;
    std::string error;
    if (!f2::ReadElfImage(absl::GetFlag(FLAGS_load_elf), &image, &error)) {
      fprintf(stderr, "f2_loader: %s\n", error.c_str());
      return 1;
    }
    if (!LoadElf(host, image, absl::GetFlag(FLAGS_verify)))
      return Fail(host, "load");
    if (entry == 0xFFFFFFFF && absl::GetFlag(FLAGS_start_core))
      entry = image.entry;
  }
  if (!absl::GetFlag(FLAGS_load_data).empty()) {
    std::vector<uint8_t> data;
    if (!ReadFile(absl::GetFlag(FLAGS_load_data), &data)) {
      fprintf(stderr, "f2_loader: cannot read %s\n", absl::GetFlag(FLAGS_load_data).c_str());
      return 1;
    }
    const uint32_t addr = absl::GetFlag(FLAGS_load_data_addr);
    fprintf(stderr, "Loading %zu bytes to 0x%08x\n", data.size(), addr);
    if (!host.WriteData(addr, data.data(), data.size()))
      return Fail(host, "load_data");
  }
  if (absl::GetFlag(FLAGS_write_word_addr) != 0xFFFFFFFF &&
      !host.WriteWord(absl::GetFlag(FLAGS_write_word_addr), absl::GetFlag(FLAGS_write_word_val))) {
    return Fail(host, "write_word");
  }
  if (entry != 0xFFFFFFFF) {
    fprintf(stderr, "Setting entry point to 0x%08x\n", entry);
    if (!host.SetEntryPoint(csr_base, entry))
      return Fail(host, "set_entry_point");
  }
  if (absl::GetFlag(FLAGS_start_core)) {
    fprintf(stderr, "Starting core...\n");
    if (!host.StartCore(csr_base))
      return Fail(host, "start_core");
  }
  if (absl::GetFlag(FLAGS_poll_halt) > 0.0 && !PollHalt(host, *ocl)) {
    if (absl::GetFlag(FLAGS_console))
      PrintConsole(host, *ocl);
    return 1;
  }

  if (absl::GetFlag(FLAGS_read_word_addr) != 0xFFFFFFFF) {
    uint32_t v;
    if (!host.ReadWord(absl::GetFlag(FLAGS_read_word_addr), &v))
      return Fail(host, "read_word");
    printf("DATA_WORD: 0x%08x\n", v);
  }
  if (absl::GetFlag(FLAGS_read_data_addr) != 0xFFFFFFFF && absl::GetFlag(FLAGS_read_data_size)) {
    std::vector<uint8_t> buf(absl::GetFlag(FLAGS_read_data_size));
    if (!host.ReadData(absl::GetFlag(FLAGS_read_data_addr), buf.size(), buf.data())) {
      return Fail(host, "read_data");
    }
    fwrite(buf.data(), 1, buf.size(), stdout);
  }
  if (absl::GetFlag(FLAGS_read_line) != 0xFFFFFFFF) {
    uint8_t line[16];
    if (!host.ReadLine(absl::GetFlag(FLAGS_read_line), line))
      return Fail(host, "read_line");
    PrintLine(absl::GetFlag(FLAGS_read_line), line);
  }
  if (absl::GetFlag(FLAGS_read_lines_addr) != 0xFFFFFFFF) {
    const uint32_t addr = absl::GetFlag(FLAGS_read_lines_addr);
    for (uint32_t c = 0; c < absl::GetFlag(FLAGS_read_lines_count); ++c) {
      uint8_t line[16];
      if (!host.ReadLine(addr + c * 16, line))
        return Fail(host, "read_lines");
      PrintLine(addr + c * 16, line);
    }
  }
  if (absl::GetFlag(FLAGS_console))
    PrintConsole(host, *ocl);
  fflush(stdout);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  absl::SetProgramUsageMessage(
      "Loads and runs programs on the CoralNPU custom logic on AWS F2.\n"
      "  f2_loader --slot=0 --load_elf=prog.elf --start_core --poll_halt=10 --console");
  absl::ParseCommandLine(argc, argv);
  return Run();
}
