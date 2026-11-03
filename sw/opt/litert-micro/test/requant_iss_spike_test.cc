// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Runs requant_iss_test.elf on Spike (idea 2, #9, step 3); requant_iss_test.py
// runs the same ELF on npusim.

#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sw/coralnpu_sim/spike_simulator.h"
#include "tests/verilator_sim/elf.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace {

using bazel::tools::cpp::runfiles::Runfiles;
using coralnpu::sim::SpikeSimulator;

constexpr char kElf[] = "coralnpu_hw/sw/opt/litert-micro/test/requant_iss_test.elf";

class RequantIssSpikeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string error;
    std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest(&error));
    ASSERT_NE(runfiles, nullptr) << error;
    path_ = runfiles->Rlocation(kElf);
    std::ifstream file(path_, std::ios::binary);
    ASSERT_TRUE(file.good()) << path_;
    elf_.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  }

  uint32_t Read(const char *symbol) {
    uint64_t addr = 0;
    EXPECT_TRUE(LookupSymbol(elf_.data(), symbol, &addr)) << symbol;
    uint32_t value = 0;
    EXPECT_TRUE(sim_.ReadMemory(addr, &value, sizeof(value))) << symbol;
    return value;
  }

  std::string path_;
  std::vector<uint8_t> elf_;
  SpikeSimulator sim_;
};

TEST_F(RequantIssSpikeTest, MatchesReferenceModel) {
  ASSERT_TRUE(sim_.LoadProgram(path_));
  for (int i = 0; i < 400 && Read("done") == 0 && !sim_.IsHalted(); ++i) {
    sim_.Step(10000);
  }
  ASSERT_EQ(Read("done"), 1u) << "program did not finish";
  EXPECT_EQ(static_cast<int32_t>(Read("insn_mismatches")), 0);
  EXPECT_EQ(static_cast<int32_t>(Read("fused_mismatches")), 0);
  EXPECT_GT(static_cast<int32_t>(Read("ties_checked")), 0);
  const uint32_t baseline = Read("baseline_instret");
  const uint32_t fused    = Read("fused_instret");
  std::cout << "Spike instret: PostprocessAcc " << baseline << ", fused " << fused << std::endl;
  EXPECT_LT(fused, baseline);
}

}  // namespace
