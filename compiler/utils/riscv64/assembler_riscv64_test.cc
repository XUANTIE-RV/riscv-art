/*
 * Copyright (C) 2015 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "assembler_riscv64.h"

#include <inttypes.h>
#include <map>
#include <random>

#include "base/bit_utils.h"
#include "base/stl_util.h"
#include "utils/assembler_test.h"

#define TEST_RV_ALL           1
#define TEST_RV32I_R          TEST_RV_ALL  // passed
#define TEST_RV32I_I          TEST_RV_ALL  // passed
#define TEST_RV32I_S          TEST_RV_ALL
#define TEST_RV32I_B          1  // passed
#define TEST_RV32I_J          1  // not passed
#define TEST_RV64I_R          TEST_RV_ALL  // passed
#define TEST_RV64I_I          TEST_RV_ALL  // passed
#define TEST_RV64I_S          TEST_RV_ALL  // passed

#define TEST_RV32M_R          TEST_RV_ALL  // passed
#define TEST_RV64M_R          TEST_RV_ALL  // passed

#define TEST_RV32A_R          TEST_RV_ALL  // passed
#define TEST_RV64A_R          TEST_RV_ALL  // passed

#define TEST_RV32F_R          TEST_RV_ALL  // passed
#define TEST_RV32F_I          TEST_RV_ALL  // passed
#define TEST_RV32F_S          TEST_RV_ALL  // passed
#define TEST_RV64F_R          TEST_RV_ALL  // passed

#define TEST_RV32D_R          TEST_RV_ALL  // passed
#define TEST_RV32D_I          TEST_RV_ALL  // passed
#define TEST_RV32D_S          TEST_RV_ALL  // passed
#define TEST_RV64D_R          TEST_RV_ALL  // passed

#define __ GetAssembler()->

namespace art {

struct RISCV64CpuRegisterCompare {
  bool operator()(const riscv64::GpuRegister& a, const riscv64::GpuRegister& b) const {
    return a < b;
  }
};

class AssemblerRISCV64Test : public AssemblerTest<riscv64::Riscv64Assembler,
                                                 riscv64::Riscv64Label,
                                                 riscv64::GpuRegister,
                                                 riscv64::FpuRegister,
                                                 uint32_t,
                                                 riscv64::VectorRegister> {
 public:
  using Base = AssemblerTest<riscv64::Riscv64Assembler,
                             riscv64::Riscv64Label,
                             riscv64::GpuRegister,
                             riscv64::FpuRegister,
                             uint32_t,
                             riscv64::VectorRegister>;
/*
  // These tests were taking too long, so we hide the DriverStr() from AssemblerTest<>
  // and reimplement it without the verification against `assembly_string`. b/73903608
  void DriverStr(const std::string& assembly_string ATTRIBUTE_UNUSED,
                 const std::string& test_name ATTRIBUTE_UNUSED) {
    GetAssembler()->FinalizeCode();
    std::vector<uint8_t> data(GetAssembler()->CodeSize());
    MemoryRegion code(data.data(), data.size());
    GetAssembler()->FinalizeInstructions(code);
  }*/

  AssemblerRISCV64Test()
      : instruction_set_features_(Riscv64InstructionSetFeatures::FromVariant("default", nullptr)) {}

 protected:
  // Get the typically used name for this architecture, e.g., aarch64, x86-64, ...
  std::string GetArchitectureString() override {
    return "riscv64";
  }

  std::string GetAssemblerCmdName() override {
    // We assemble and link for RISCV64R6. See GetAssemblerParameters() for details.
    return "gcc";
  }

  std::string GetAssemblerParameters() override {
    // We assemble and link for RISCV64R6. The reason is that object files produced for RISCV64R6
    // (and MIPS32R6) with the GNU assembler don't have correct final offsets in PC-relative
    // branches in the .text section and so they require a relocation pass (there's a relocation
    // section, .rela.text, that has the needed info to fix up the branches).
    // return " -march=mips64r6 -mmsa -Wa,--no-warn -Wl,-Ttext=0 -Wl,-e0 -nostdlib";
    return " -march=rv64imafd -mabi=lp64 -Wa,--no-warn -Wl,-Ttext=0 -Wl,-e0 -nostdlib";
  }

  void Pad(std::vector<uint8_t>& data ATTRIBUTE_UNUSED) override {
    // The GNU linker unconditionally pads the code segment with NOPs to a size that is a multiple
    // of 16 and there doesn't appear to be a way to suppress this padding. Our assembler doesn't
    // pad, so, in order for two assembler outputs to match, we need to match the padding as well.
    // NOP is encoded as four zero bytes on MIPS.
    // size_t pad_size = RoundUp(data.size(), 16u) - data.size();
    // data.insert(data.end(), pad_size, 0);
  }

  std::string GetDisassembleParameters() override {
    return " -D -bbinary -mriscv:rv64";
  }

  riscv64::Riscv64Assembler* CreateAssembler(ArenaAllocator* allocator) override {
    return new (allocator) riscv64::Riscv64Assembler(allocator, instruction_set_features_.get());
  }

  void SetUpHelpers() override {
    if (registers_.size() == 0) {
      registers_.push_back(new riscv64::GpuRegister(riscv64::ZERO));
      registers_.push_back(new riscv64::GpuRegister(riscv64::RA));
      registers_.push_back(new riscv64::GpuRegister(riscv64::SP));
      registers_.push_back(new riscv64::GpuRegister(riscv64::GP));
      registers_.push_back(new riscv64::GpuRegister(riscv64::TP));
      registers_.push_back(new riscv64::GpuRegister(riscv64::T0));
      registers_.push_back(new riscv64::GpuRegister(riscv64::T1));
      registers_.push_back(new riscv64::GpuRegister(riscv64::T2));
      registers_.push_back(new riscv64::GpuRegister(riscv64::S0));
      registers_.push_back(new riscv64::GpuRegister(riscv64::S1));
      registers_.push_back(new riscv64::GpuRegister(riscv64::A0));
      registers_.push_back(new riscv64::GpuRegister(riscv64::A1));
      registers_.push_back(new riscv64::GpuRegister(riscv64::A2));
      registers_.push_back(new riscv64::GpuRegister(riscv64::A3));
      registers_.push_back(new riscv64::GpuRegister(riscv64::A4));
      registers_.push_back(new riscv64::GpuRegister(riscv64::A5));
      registers_.push_back(new riscv64::GpuRegister(riscv64::A6));
      registers_.push_back(new riscv64::GpuRegister(riscv64::A7));
      registers_.push_back(new riscv64::GpuRegister(riscv64::S2));
      registers_.push_back(new riscv64::GpuRegister(riscv64::S3));
      registers_.push_back(new riscv64::GpuRegister(riscv64::S4));
      registers_.push_back(new riscv64::GpuRegister(riscv64::S5));
      registers_.push_back(new riscv64::GpuRegister(riscv64::S6));
      registers_.push_back(new riscv64::GpuRegister(riscv64::S7));
      registers_.push_back(new riscv64::GpuRegister(riscv64::S8));
      registers_.push_back(new riscv64::GpuRegister(riscv64::S9));
      registers_.push_back(new riscv64::GpuRegister(riscv64::S10));
      registers_.push_back(new riscv64::GpuRegister(riscv64::S11));
      registers_.push_back(new riscv64::GpuRegister(riscv64::T3));
      registers_.push_back(new riscv64::GpuRegister(riscv64::T4));
      registers_.push_back(new riscv64::GpuRegister(riscv64::T5));
      registers_.push_back(new riscv64::GpuRegister(riscv64::T6));

      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::ZERO), "zero");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::RA), "ra");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::SP), "sp");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::GP), "gp");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::TP), "tp");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::T0), "t0");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::T1), "t1");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::T2), "t2");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::S0), "s0");  // s0/fp
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::S1), "s1");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::A0), "a0");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::A1), "a1");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::A2), "a2");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::A3), "a3");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::A4), "a4");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::A5), "a5");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::A6), "a6");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::A7), "a7");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::S2), "s2");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::S3), "s3");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::S4), "s4");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::S5), "s5");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::S6), "s6");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::S7), "s7");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::S8), "s8");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::S9), "s9");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::S10), "s10");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::S11), "s11");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::T3), "t3");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::T4), "t4");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::T5), "t5");
      secondary_register_names_.emplace(riscv64::GpuRegister(riscv64::T6), "t6");

      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FT0));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FT1));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FT2));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FT3));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FT4));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FT5));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FT6));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FT7));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FS0));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FS1));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FA0));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FA1));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FA2));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FA3));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FA4));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FA5));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FA6));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FA7));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FS2));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FS3));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FS4));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FS5));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FS6));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FS7));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FS8));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FS9));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FS10));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FS11));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FT8));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FT9));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FT10));
      fp_registers_.push_back(new riscv64::FpuRegister(riscv64::FT11));

      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W0));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W1));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W2));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W3));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W4));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W5));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W6));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W7));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W8));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W9));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W10));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W11));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W12));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W13));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W14));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W15));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W16));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W17));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W18));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W19));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W20));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W21));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W22));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W23));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W24));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W25));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W26));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W27));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W28));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W29));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W30));
      vec_registers_.push_back(new riscv64::VectorRegister(riscv64::W31));
    }
  }

  void TearDown() override {
    AssemblerTest::TearDown();
    STLDeleteElements(&registers_);
    STLDeleteElements(&fp_registers_);
    STLDeleteElements(&vec_registers_);
  }

  std::vector<riscv64::Riscv64Label> GetAddresses() override {
    UNIMPLEMENTED(FATAL) << "Feature not implemented yet";
    UNREACHABLE();
  }

  std::vector<riscv64::GpuRegister*> GetRegisters() override {
    return registers_;
  }

  std::vector<riscv64::FpuRegister*> GetFPRegisters() override {
    return fp_registers_;
  }

  std::vector<riscv64::VectorRegister*> GetVectorRegisters() override {
    return vec_registers_;
  }

  uint32_t CreateImmediate(int64_t imm_value) override {
    return imm_value;
  }

  std::string GetSecondaryRegisterName(const riscv64::GpuRegister& reg) override {
    CHECK(secondary_register_names_.find(reg) != secondary_register_names_.end());
    return secondary_register_names_[reg];
  }

  std::string RepeatInsn(size_t count, const std::string& insn) {
    std::string result;
    for (; count != 0u; --count) {
      result += insn;
    }
    return result;
  }

  void BranchHelper(void (riscv64::Riscv64Assembler::*f)(riscv64::Riscv64Label*,
                                                       bool),
                    const std::string& instr_name,
                    bool is_bare = false) {
    riscv64::Riscv64Label label1, label2;
    (Base::GetAssembler()->*f)(&label1, is_bare);
    constexpr size_t kAdduCount1 = 63;
    for (size_t i = 0; i != kAdduCount1; ++i) {
      __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
    }
    __ Bind(&label1);
    (Base::GetAssembler()->*f)(&label2, is_bare);
    constexpr size_t kAdduCount2 = 64;
    for (size_t i = 0; i != kAdduCount2; ++i) {
      __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
    }
    __ Bind(&label2);
    (Base::GetAssembler()->*f)(&label1, is_bare);
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);

    std::string expected =
        ".set noreorder\n" +
        instr_name + " 1f\n" +
        RepeatInsn(kAdduCount1, "addu $zero, $zero, $zero\n") +
        "1:\n" +
        instr_name + " 2f\n" +
        RepeatInsn(kAdduCount2, "addu $zero, $zero, $zero\n") +
        "2:\n" +
        instr_name + " 1b\n" +
        "addu $zero, $zero, $zero\n";
    DriverStr(expected, instr_name);
  }

  void BranchHelper1(void (riscv64::Riscv64Assembler::*f)(riscv64::Riscv64Label*,
                                                       bool),
                    const std::string& instr_name,
                    bool is_bare = false) {
    riscv64::Riscv64Label label1, label2;
    (Base::GetAssembler()->*f)(&label1, is_bare);
    constexpr size_t kAdduCount1 = 63;
    for (size_t i = 0; i != kAdduCount1; ++i) {
      __ Add(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
    }
    __ Bind(&label1);
    (Base::GetAssembler()->*f)(&label2, is_bare);
    constexpr size_t kAdduCount2 = 64;
    for (size_t i = 0; i != kAdduCount2; ++i) {
      __ Add(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
    }
    __ Bind(&label2);
    (Base::GetAssembler()->*f)(&label1, is_bare);
    __ Add(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);

    std::string expected =
        instr_name + " 1f\n" +
        RepeatInsn(kAdduCount1, "add zero, zero, zero\n") +
        "1:\n" +
        instr_name + " 2f\n" +
        RepeatInsn(kAdduCount2, "add zero, zero, zero\n") +
        "2:\n" +
        instr_name + " 1b\n" +
        "add zero, zero, zero\n";
    DriverStr(expected, instr_name);
  }

  void BranchCondOneRegHelper(void (riscv64::Riscv64Assembler::*f)(riscv64::GpuRegister,
                                                                 riscv64::Riscv64Label*,
                                                                 bool),
                              const std::string& instr_name,
                              bool is_bare = false) {
    riscv64::Riscv64Label label;
    (Base::GetAssembler()->*f)(riscv64::A0, &label, is_bare);
    constexpr size_t kAdduCount1 = 63;
    for (size_t i = 0; i != kAdduCount1; ++i) {
      __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
    }
    __ Bind(&label);
    constexpr size_t kAdduCount2 = 64;
    for (size_t i = 0; i != kAdduCount2; ++i) {
      __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
    }
    (Base::GetAssembler()->*f)(riscv64::A1, &label, is_bare);
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);

    std::string expected =
        ".set noreorder\n" +
        instr_name + " $a0, 1f\n" +
        (is_bare ? "" : "nop\n") +
        RepeatInsn(kAdduCount1, "addu $zero, $zero, $zero\n") +
        "1:\n" +
        RepeatInsn(kAdduCount2, "addu $zero, $zero, $zero\n") +
        instr_name + " $a1, 1b\n" +
        (is_bare ? "" : "nop\n") +
        "addu $zero, $zero, $zero\n";
    DriverStr(expected, instr_name);
  }

  void BranchCondTwoRegsHelper(void (riscv64::Riscv64Assembler::*f)(riscv64::GpuRegister,
                                                                  riscv64::GpuRegister,
                                                                  riscv64::Riscv64Label*,
                                                                  bool),
                               const std::string& instr_name,
                               bool is_bare = false) {
    riscv64::Riscv64Label label;
    (Base::GetAssembler()->*f)(riscv64::A0, riscv64::A1, &label, is_bare);
    constexpr size_t kAdduCount1 = 63;
    for (size_t i = 0; i != kAdduCount1; ++i) {
      __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
    }
    __ Bind(&label);
    constexpr size_t kAdduCount2 = 64;
    for (size_t i = 0; i != kAdduCount2; ++i) {
      __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
    }
    (Base::GetAssembler()->*f)(riscv64::A2, riscv64::A3, &label, is_bare);
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);

    std::string expected =
        ".set noreorder\n" +
        instr_name + " $a0, $a1, 1f\n" +
        (is_bare ? "" : "nop\n") +
        RepeatInsn(kAdduCount1, "addu $zero, $zero, $zero\n") +
        "1:\n" +
        RepeatInsn(kAdduCount2, "addu $zero, $zero, $zero\n") +
        instr_name + " $a2, $a3, 1b\n" +
        (is_bare ? "" : "nop\n") +
        "addu $zero, $zero, $zero\n";
    DriverStr(expected, instr_name);
  }

  void BranchCondTwoRegsHelper1(void (riscv64::Riscv64Assembler::*f)(riscv64::GpuRegister,
                                                                  riscv64::GpuRegister,
                                                                  riscv64::Riscv64Label*,
                                                                  bool),
                               const std::string& instr_name,
                               bool is_bare = false) {
    riscv64::Riscv64Label label;
    (Base::GetAssembler()->*f)(riscv64::A0, riscv64::A1, &label, is_bare);
    constexpr size_t kAdduCount1 = 63;
    for (size_t i = 0; i != kAdduCount1; ++i) {
      __ Add(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
    }
    __ Bind(&label);
    constexpr size_t kAdduCount2 = 64;
    for (size_t i = 0; i != kAdduCount2; ++i) {
      __ Add(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
    }
    (Base::GetAssembler()->*f)(riscv64::A2, riscv64::A3, &label, is_bare);
    __ Add(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);

    std::string expected =
        instr_name + " a0, a1, 1f\n" +
        (is_bare ? "" : "nop\n") +
        RepeatInsn(kAdduCount1, "add zero, zero, zero\n") +
        "1:\n" +
        RepeatInsn(kAdduCount2, "add zero, zero, zero\n") +
        instr_name + " a2, a3, 1b\n" +
        (is_bare ? "" : "nop\n") +
        "add zero, zero, zero\n";
    DriverStr(expected, instr_name);
  }

  void BranchFpuCondHelper(void (riscv64::Riscv64Assembler::*f)(riscv64::FpuRegister,
                                                              riscv64::Riscv64Label*,
                                                              bool),
                           const std::string& instr_name,
                           bool is_bare = false) {
    riscv64::Riscv64Label label;
    (Base::GetAssembler()->*f)(riscv64::F0, &label, is_bare);
    constexpr size_t kAdduCount1 = 63;
    for (size_t i = 0; i != kAdduCount1; ++i) {
      __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
    }
    __ Bind(&label);
    constexpr size_t kAdduCount2 = 64;
    for (size_t i = 0; i != kAdduCount2; ++i) {
      __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
    }
    (Base::GetAssembler()->*f)(riscv64::FT11, &label, is_bare);
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);

    std::string expected =
        ".set noreorder\n" +
        instr_name + " $f0, 1f\n" +
        (is_bare ? "" : "nop\n") +
        RepeatInsn(kAdduCount1, "addu $zero, $zero, $zero\n") +
        "1:\n" +
        RepeatInsn(kAdduCount2, "addu $zero, $zero, $zero\n") +
        instr_name + " $f31, 1b\n" +
        (is_bare ? "" : "nop\n") +
        "addu $zero, $zero, $zero\n";
    DriverStr(expected, instr_name);
  }

 private:
  std::vector<riscv64::GpuRegister*> registers_;
  std::map<riscv64::GpuRegister, std::string, RISCV64CpuRegisterCompare> secondary_register_names_;

  std::vector<riscv64::FpuRegister*> fp_registers_;
  std::vector<riscv64::VectorRegister*> vec_registers_;

  std::unique_ptr<const Riscv64InstructionSetFeatures> instruction_set_features_;
};

TEST_F(AssemblerRISCV64Test, Toolchain) {
  EXPECT_TRUE(CheckTools());
}
#if 0
///////////////////
// FP Operations //
///////////////////

TEST_F(AssemblerRISCV64Test, AddS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::AddS, "add.s ${reg1}, ${reg2}, ${reg3}"), "add.s");
}

TEST_F(AssemblerRISCV64Test, AddD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::AddD, "add.d ${reg1}, ${reg2}, ${reg3}"), "add.d");
}

TEST_F(AssemblerRISCV64Test, SubS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::SubS, "sub.s ${reg1}, ${reg2}, ${reg3}"), "sub.s");
}

TEST_F(AssemblerRISCV64Test, SubD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::SubD, "sub.d ${reg1}, ${reg2}, ${reg3}"), "sub.d");
}

TEST_F(AssemblerRISCV64Test, MulS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::MulS, "mul.s ${reg1}, ${reg2}, ${reg3}"), "mul.s");
}

TEST_F(AssemblerRISCV64Test, MulD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::MulD, "mul.d ${reg1}, ${reg2}, ${reg3}"), "mul.d");
}

TEST_F(AssemblerRISCV64Test, DivS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::DivS, "div.s ${reg1}, ${reg2}, ${reg3}"), "div.s");
}

TEST_F(AssemblerRISCV64Test, DivD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::DivD, "div.d ${reg1}, ${reg2}, ${reg3}"), "div.d");
}

TEST_F(AssemblerRISCV64Test, SqrtS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::SqrtS, "sqrt.s ${reg1}, ${reg2}"), "sqrt.s");
}

TEST_F(AssemblerRISCV64Test, SqrtD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::SqrtD, "sqrt.d ${reg1}, ${reg2}"), "sqrt.d");
}

TEST_F(AssemblerRISCV64Test, AbsS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::AbsS, "abs.s ${reg1}, ${reg2}"), "abs.s");
}

TEST_F(AssemblerRISCV64Test, AbsD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::AbsD, "abs.d ${reg1}, ${reg2}"), "abs.d");
}

TEST_F(AssemblerRISCV64Test, MovS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::MovS, "mov.s ${reg1}, ${reg2}"), "mov.s");
}

TEST_F(AssemblerRISCV64Test, MovD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::MovD, "mov.d ${reg1}, ${reg2}"), "mov.d");
}

TEST_F(AssemblerRISCV64Test, NegS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::NegS, "neg.s ${reg1}, ${reg2}"), "neg.s");
}

TEST_F(AssemblerRISCV64Test, NegD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::NegD, "neg.d ${reg1}, ${reg2}"), "neg.d");
}

TEST_F(AssemblerRISCV64Test, RoundLS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::RoundLS, "round.l.s ${reg1}, ${reg2}"), "round.l.s");
}

TEST_F(AssemblerRISCV64Test, RoundLD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::RoundLD, "round.l.d ${reg1}, ${reg2}"), "round.l.d");
}

TEST_F(AssemblerRISCV64Test, RoundWS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::RoundWS, "round.w.s ${reg1}, ${reg2}"), "round.w.s");
}

TEST_F(AssemblerRISCV64Test, RoundWD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::RoundWD, "round.w.d ${reg1}, ${reg2}"), "round.w.d");
}

TEST_F(AssemblerRISCV64Test, CeilLS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::CeilLS, "ceil.l.s ${reg1}, ${reg2}"), "ceil.l.s");
}

TEST_F(AssemblerRISCV64Test, CeilLD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::CeilLD, "ceil.l.d ${reg1}, ${reg2}"), "ceil.l.d");
}

TEST_F(AssemblerRISCV64Test, CeilWS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::CeilWS, "ceil.w.s ${reg1}, ${reg2}"), "ceil.w.s");
}

TEST_F(AssemblerRISCV64Test, CeilWD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::CeilWD, "ceil.w.d ${reg1}, ${reg2}"), "ceil.w.d");
}

TEST_F(AssemblerRISCV64Test, FloorLS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::FloorLS, "floor.l.s ${reg1}, ${reg2}"), "floor.l.s");
}

TEST_F(AssemblerRISCV64Test, FloorLD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::FloorLD, "floor.l.d ${reg1}, ${reg2}"), "floor.l.d");
}

TEST_F(AssemblerRISCV64Test, FloorWS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::FloorWS, "floor.w.s ${reg1}, ${reg2}"), "floor.w.s");
}

TEST_F(AssemblerRISCV64Test, FloorWD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::FloorWD, "floor.w.d ${reg1}, ${reg2}"), "floor.w.d");
}

TEST_F(AssemblerRISCV64Test, SelS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::SelS, "sel.s ${reg1}, ${reg2}, ${reg3}"), "sel.s");
}

TEST_F(AssemblerRISCV64Test, SelD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::SelD, "sel.d ${reg1}, ${reg2}, ${reg3}"), "sel.d");
}

TEST_F(AssemblerRISCV64Test, SeleqzS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::SeleqzS, "seleqz.s ${reg1}, ${reg2}, ${reg3}"),
            "seleqz.s");
}

TEST_F(AssemblerRISCV64Test, SeleqzD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::SeleqzD, "seleqz.d ${reg1}, ${reg2}, ${reg3}"),
            "seleqz.d");
}

TEST_F(AssemblerRISCV64Test, SelnezS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::SelnezS, "selnez.s ${reg1}, ${reg2}, ${reg3}"),
            "selnez.s");
}

TEST_F(AssemblerRISCV64Test, SelnezD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::SelnezD, "selnez.d ${reg1}, ${reg2}, ${reg3}"),
            "selnez.d");
}

TEST_F(AssemblerRISCV64Test, RintS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::RintS, "rint.s ${reg1}, ${reg2}"), "rint.s");
}

TEST_F(AssemblerRISCV64Test, RintD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::RintD, "rint.d ${reg1}, ${reg2}"), "rint.d");
}

TEST_F(AssemblerRISCV64Test, ClassS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::ClassS, "class.s ${reg1}, ${reg2}"), "class.s");
}

TEST_F(AssemblerRISCV64Test, ClassD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::ClassD, "class.d ${reg1}, ${reg2}"), "class.d");
}

TEST_F(AssemblerRISCV64Test, MinS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::MinS, "min.s ${reg1}, ${reg2}, ${reg3}"), "min.s");
}

TEST_F(AssemblerRISCV64Test, MinD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::MinD, "min.d ${reg1}, ${reg2}, ${reg3}"), "min.d");
}

TEST_F(AssemblerRISCV64Test, MaxS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::MaxS, "max.s ${reg1}, ${reg2}, ${reg3}"), "max.s");
}

TEST_F(AssemblerRISCV64Test, MaxD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::MaxD, "max.d ${reg1}, ${reg2}, ${reg3}"), "max.d");
}

TEST_F(AssemblerRISCV64Test, CmpUnS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpUnS, "cmp.un.s ${reg1}, ${reg2}, ${reg3}"),
            "cmp.un.s");
}

TEST_F(AssemblerRISCV64Test, CmpEqS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpEqS, "cmp.eq.s ${reg1}, ${reg2}, ${reg3}"),
            "cmp.eq.s");
}

TEST_F(AssemblerRISCV64Test, CmpUeqS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpUeqS, "cmp.ueq.s ${reg1}, ${reg2}, ${reg3}"),
            "cmp.ueq.s");
}

TEST_F(AssemblerRISCV64Test, CmpLtS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpLtS, "cmp.lt.s ${reg1}, ${reg2}, ${reg3}"),
            "cmp.lt.s");
}

TEST_F(AssemblerRISCV64Test, CmpUltS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpUltS, "cmp.ult.s ${reg1}, ${reg2}, ${reg3}"),
            "cmp.ult.s");
}

TEST_F(AssemblerRISCV64Test, CmpLeS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpLeS, "cmp.le.s ${reg1}, ${reg2}, ${reg3}"),
            "cmp.le.s");
}

TEST_F(AssemblerRISCV64Test, CmpUleS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpUleS, "cmp.ule.s ${reg1}, ${reg2}, ${reg3}"),
            "cmp.ule.s");
}

TEST_F(AssemblerRISCV64Test, CmpOrS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpOrS, "cmp.or.s ${reg1}, ${reg2}, ${reg3}"),
            "cmp.or.s");
}

TEST_F(AssemblerRISCV64Test, CmpUneS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpUneS, "cmp.une.s ${reg1}, ${reg2}, ${reg3}"),
            "cmp.une.s");
}

TEST_F(AssemblerRISCV64Test, CmpNeS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpNeS, "cmp.ne.s ${reg1}, ${reg2}, ${reg3}"),
            "cmp.ne.s");
}

TEST_F(AssemblerRISCV64Test, CmpUnD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpUnD, "cmp.un.d ${reg1}, ${reg2}, ${reg3}"),
            "cmp.un.d");
}

TEST_F(AssemblerRISCV64Test, CmpEqD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpEqD, "cmp.eq.d ${reg1}, ${reg2}, ${reg3}"),
            "cmp.eq.d");
}

TEST_F(AssemblerRISCV64Test, CmpUeqD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpUeqD, "cmp.ueq.d ${reg1}, ${reg2}, ${reg3}"),
            "cmp.ueq.d");
}

TEST_F(AssemblerRISCV64Test, CmpLtD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpLtD, "cmp.lt.d ${reg1}, ${reg2}, ${reg3}"),
            "cmp.lt.d");
}

TEST_F(AssemblerRISCV64Test, CmpUltD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpUltD, "cmp.ult.d ${reg1}, ${reg2}, ${reg3}"),
            "cmp.ult.d");
}

TEST_F(AssemblerRISCV64Test, CmpLeD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpLeD, "cmp.le.d ${reg1}, ${reg2}, ${reg3}"),
            "cmp.le.d");
}

TEST_F(AssemblerRISCV64Test, CmpUleD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpUleD, "cmp.ule.d ${reg1}, ${reg2}, ${reg3}"),
            "cmp.ule.d");
}

TEST_F(AssemblerRISCV64Test, CmpOrD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpOrD, "cmp.or.d ${reg1}, ${reg2}, ${reg3}"),
            "cmp.or.d");
}

TEST_F(AssemblerRISCV64Test, CmpUneD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpUneD, "cmp.une.d ${reg1}, ${reg2}, ${reg3}"),
            "cmp.une.d");
}

TEST_F(AssemblerRISCV64Test, CmpNeD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::CmpNeD, "cmp.ne.d ${reg1}, ${reg2}, ${reg3}"),
            "cmp.ne.d");
}

TEST_F(AssemblerRISCV64Test, CvtDL) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::Cvtdl, "cvt.d.l ${reg1}, ${reg2}"), "cvt.d.l");
}

TEST_F(AssemblerRISCV64Test, CvtDS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::Cvtds, "cvt.d.s ${reg1}, ${reg2}"), "cvt.d.s");
}

TEST_F(AssemblerRISCV64Test, CvtDW) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::Cvtdw, "cvt.d.w ${reg1}, ${reg2}"), "cvt.d.w");
}

TEST_F(AssemblerRISCV64Test, CvtSL) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::Cvtsl, "cvt.s.l ${reg1}, ${reg2}"), "cvt.s.l");
}

TEST_F(AssemblerRISCV64Test, CvtSD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::Cvtsd, "cvt.s.d ${reg1}, ${reg2}"), "cvt.s.d");
}

TEST_F(AssemblerRISCV64Test, CvtSW) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::Cvtsw, "cvt.s.w ${reg1}, ${reg2}"), "cvt.s.w");
}

TEST_F(AssemblerRISCV64Test, TruncWS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::TruncWS, "trunc.w.s ${reg1}, ${reg2}"), "trunc.w.s");
}

TEST_F(AssemblerRISCV64Test, TruncWD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::TruncWD, "trunc.w.d ${reg1}, ${reg2}"), "trunc.w.d");
}

TEST_F(AssemblerRISCV64Test, TruncLS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::TruncLS, "trunc.l.s ${reg1}, ${reg2}"), "trunc.l.s");
}

TEST_F(AssemblerRISCV64Test, TruncLD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::TruncLD, "trunc.l.d ${reg1}, ${reg2}"), "trunc.l.d");
}

TEST_F(AssemblerRISCV64Test, Mfc1) {
  DriverStr(RepeatRF(&riscv64::Riscv64Assembler::Mfc1, "mfc1 ${reg1}, ${reg2}"), "Mfc1");
}

TEST_F(AssemblerRISCV64Test, Mfhc1) {
  DriverStr(RepeatRF(&riscv64::Riscv64Assembler::Mfhc1, "mfhc1 ${reg1}, ${reg2}"), "Mfhc1");
}

TEST_F(AssemblerRISCV64Test, Mtc1) {
  DriverStr(RepeatRF(&riscv64::Riscv64Assembler::Mtc1, "mtc1 ${reg1}, ${reg2}"), "Mtc1");
}

TEST_F(AssemblerRISCV64Test, Mthc1) {
  DriverStr(RepeatRF(&riscv64::Riscv64Assembler::Mthc1, "mthc1 ${reg1}, ${reg2}"), "Mthc1");
}

TEST_F(AssemblerRISCV64Test, Dmfc1) {
  DriverStr(RepeatRF(&riscv64::Riscv64Assembler::Dmfc1, "dmfc1 ${reg1}, ${reg2}"), "Dmfc1");
}

TEST_F(AssemblerRISCV64Test, Dmtc1) {
  DriverStr(RepeatRF(&riscv64::Riscv64Assembler::Dmtc1, "dmtc1 ${reg1}, ${reg2}"), "Dmtc1");
}

TEST_F(AssemblerRISCV64Test, Lwc1) {
  DriverStr(RepeatFRIb(&riscv64::Riscv64Assembler::Lwc1, -16, "lwc1 ${reg1}, {imm}(${reg2})"),
            "lwc1");
}

TEST_F(AssemblerRISCV64Test, Ldc1) {
  DriverStr(RepeatFRIb(&riscv64::Riscv64Assembler::Ldc1, -16, "ldc1 ${reg1}, {imm}(${reg2})"),
            "ldc1");
}

TEST_F(AssemblerRISCV64Test, Swc1) {
  DriverStr(RepeatFRIb(&riscv64::Riscv64Assembler::Swc1, -16, "swc1 ${reg1}, {imm}(${reg2})"),
            "swc1");
}

TEST_F(AssemblerRISCV64Test, Sdc1) {
  DriverStr(RepeatFRIb(&riscv64::Riscv64Assembler::Sdc1, -16, "sdc1 ${reg1}, {imm}(${reg2})"),
            "sdc1");
}

//////////////
// BRANCHES //
//////////////

TEST_F(AssemblerRISCV64Test, Jalr) {
  DriverStr(".set noreorder\n" +
            RepeatRRNoDupes(&riscv64::Riscv64Assembler::Jalr, "jalr ${reg1}, ${reg2}"), "jalr");
}

TEST_F(AssemblerRISCV64Test, Bc) {
  BranchHelper(&riscv64::Riscv64Assembler::Bc, "Bc");
}

TEST_F(AssemblerRISCV64Test, Balc) {
  BranchHelper(&riscv64::Riscv64Assembler::Balc, "Balc");
}

TEST_F(AssemblerRISCV64Test, Beqzc) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Beqzc, "Beqzc");
}

TEST_F(AssemblerRISCV64Test, Bnezc) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Bnezc, "Bnezc");
}

TEST_F(AssemblerRISCV64Test, Bltzc) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Bltzc, "Bltzc");
}

TEST_F(AssemblerRISCV64Test, Bgezc) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Bgezc, "Bgezc");
}

TEST_F(AssemblerRISCV64Test, Blezc) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Blezc, "Blezc");
}

TEST_F(AssemblerRISCV64Test, Bgtzc) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Bgtzc, "Bgtzc");
}

TEST_F(AssemblerRISCV64Test, Beqc) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Beqc, "Beqc");
}

TEST_F(AssemblerRISCV64Test, Bnec) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Bnec, "Bnec");
}

TEST_F(AssemblerRISCV64Test, Bltc) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Bltc, "Bltc");
}

TEST_F(AssemblerRISCV64Test, Bgec) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Bgec, "Bgec");
}

TEST_F(AssemblerRISCV64Test, Bltuc) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Bltuc, "Bltuc");
}

TEST_F(AssemblerRISCV64Test, Bgeuc) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Bgeuc, "Bgeuc");
}

TEST_F(AssemblerRISCV64Test, Bc1eqz) {
  BranchFpuCondHelper(&riscv64::Riscv64Assembler::Bc1eqz, "Bc1eqz");
}

TEST_F(AssemblerRISCV64Test, Bc1nez) {
  BranchFpuCondHelper(&riscv64::Riscv64Assembler::Bc1nez, "Bc1nez");
}

TEST_F(AssemblerRISCV64Test, BareBc) {
  BranchHelper(&riscv64::Riscv64Assembler::Bc, "Bc", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBalc) {
  BranchHelper(&riscv64::Riscv64Assembler::Balc, "Balc", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBeqzc) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Beqzc, "Beqzc", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBnezc) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Bnezc, "Bnezc", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBltzc) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Bltzc, "Bltzc", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBgezc) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Bgezc, "Bgezc", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBlezc) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Blezc, "Blezc", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBgtzc) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Bgtzc, "Bgtzc", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBeqc) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Beqc, "Beqc", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBnec) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Bnec, "Bnec", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBltc) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Bltc, "Bltc", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBgec) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Bgec, "Bgec", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBltuc) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Bltuc, "Bltuc", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBgeuc) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Bgeuc, "Bgeuc", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBc1eqz) {
  BranchFpuCondHelper(&riscv64::Riscv64Assembler::Bc1eqz, "Bc1eqz", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBc1nez) {
  BranchFpuCondHelper(&riscv64::Riscv64Assembler::Bc1nez, "Bc1nez", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBeqz) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Beqz, "Beqz", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBnez) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Bnez, "Bnez", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBltz) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Bltz, "Bltz", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBgez) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Bgez, "Bgez", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBlez) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Blez, "Blez", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBgtz) {
  BranchCondOneRegHelper(&riscv64::Riscv64Assembler::Bgtz, "Bgtz", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBeq) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Beq, "Beq", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, BareBne) {
  BranchCondTwoRegsHelper(&riscv64::Riscv64Assembler::Bne, "Bne", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, LongBeqc) {
  riscv64::Riscv64Label label;
  __ Beqc(riscv64::A0, riscv64::A1, &label);
  constexpr uint32_t kAdduCount1 = (1u << 15) + 1;
  for (uint32_t i = 0; i != kAdduCount1; ++i) {
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
  }
  __ Bind(&label);
  constexpr uint32_t kAdduCount2 = (1u << 15) + 1;
  for (uint32_t i = 0; i != kAdduCount2; ++i) {
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
  }
  __ Beqc(riscv64::A2, riscv64::A3, &label);

  uint32_t offset_forward = 2 + kAdduCount1;  // 2: account for auipc and jic.
  offset_forward <<= 2;
  offset_forward += (offset_forward & 0x8000) << 1;  // Account for sign extension in jic.

  uint32_t offset_back = -(kAdduCount2 + 1);  // 1: account for bnec.
  offset_back <<= 2;
  offset_back += (offset_back & 0x8000) << 1;  // Account for sign extension in jic.

  std::ostringstream oss;
  oss <<
      ".set noreorder\n"
      "bnec $a0, $a1, 1f\n"
      "auipc $at, 0x" << std::hex << High16Bits(offset_forward) << "\n"
      "jic $at, 0x" << std::hex << Low16Bits(offset_forward) << "\n"
      "1:\n" <<
      RepeatInsn(kAdduCount1, "addu $zero, $zero, $zero\n") <<
      "2:\n" <<
      RepeatInsn(kAdduCount2, "addu $zero, $zero, $zero\n") <<
      "bnec $a2, $a3, 3f\n"
      "auipc $at, 0x" << std::hex << High16Bits(offset_back) << "\n"
      "jic $at, 0x" << std::hex << Low16Bits(offset_back) << "\n"
      "3:\n";
  std::string expected = oss.str();
  DriverStr(expected, "LongBeqc");
}

TEST_F(AssemblerRISCV64Test, LongBeqzc) {
  constexpr uint32_t kNopCount1 = (1u << 20) + 1;
  constexpr uint32_t kNopCount2 = (1u << 20) + 1;
  constexpr uint32_t kRequiredCapacity = (kNopCount1 + kNopCount2 + 6u) * 4u;
  ASSERT_LT(__ GetBuffer()->Capacity(), kRequiredCapacity);
  __ GetBuffer()->ExtendCapacity(kRequiredCapacity);
  riscv64::Riscv64Label label;
  __ Beqzc(riscv64::A0, &label);
  for (uint32_t i = 0; i != kNopCount1; ++i) {
    __ Nop();
  }
  __ Bind(&label);
  for (uint32_t i = 0; i != kNopCount2; ++i) {
    __ Nop();
  }
  __ Beqzc(riscv64::A2, &label);

  uint32_t offset_forward = 2 + kNopCount1;  // 2: account for auipc and jic.
  offset_forward <<= 2;
  offset_forward += (offset_forward & 0x8000) << 1;  // Account for sign extension in jic.

  uint32_t offset_back = -(kNopCount2 + 1);  // 1: account for bnezc.
  offset_back <<= 2;
  offset_back += (offset_back & 0x8000) << 1;  // Account for sign extension in jic.

  // Note, we're using the ".fill" directive to tell the assembler to generate many NOPs
  // instead of generating them ourselves in the source code. This saves test time.
  std::ostringstream oss;
  oss <<
      ".set noreorder\n"
      "bnezc $a0, 1f\n"
      "auipc $at, 0x" << std::hex << High16Bits(offset_forward) << "\n"
      "jic $at, 0x" << std::hex << Low16Bits(offset_forward) << "\n"
      "1:\n" <<
      ".fill 0x" << std::hex << kNopCount1 << " , 4, 0\n"
      "2:\n" <<
      ".fill 0x" << std::hex << kNopCount2 << " , 4, 0\n"
      "bnezc $a2, 3f\n"
      "auipc $at, 0x" << std::hex << High16Bits(offset_back) << "\n"
      "jic $at, 0x" << std::hex << Low16Bits(offset_back) << "\n"
      "3:\n";
  std::string expected = oss.str();
  DriverStr(expected, "LongBeqzc");
}

TEST_F(AssemblerRISCV64Test, LongBalc) {
  constexpr uint32_t kNopCount1 = (1u << 25) + 1;
  constexpr uint32_t kNopCount2 = (1u << 25) + 1;
  constexpr uint32_t kRequiredCapacity = (kNopCount1 + kNopCount2 + 6u) * 4u;
  ASSERT_LT(__ GetBuffer()->Capacity(), kRequiredCapacity);
  __ GetBuffer()->ExtendCapacity(kRequiredCapacity);
  riscv64::Riscv64Label label1, label2;
  __ Balc(&label1);
  for (uint32_t i = 0; i != kNopCount1; ++i) {
    __ Nop();
  }
  __ Bind(&label1);
  __ Balc(&label2);
  for (uint32_t i = 0; i != kNopCount2; ++i) {
    __ Nop();
  }
  __ Bind(&label2);
  __ Balc(&label1);

  uint32_t offset_forward1 = 2 + kNopCount1;  // 2: account for auipc and jialc.
  offset_forward1 <<= 2;
  offset_forward1 += (offset_forward1 & 0x8000) << 1;  // Account for sign extension in jialc.

  uint32_t offset_forward2 = 2 + kNopCount2;  // 2: account for auipc and jialc.
  offset_forward2 <<= 2;
  offset_forward2 += (offset_forward2 & 0x8000) << 1;  // Account for sign extension in jialc.

  uint32_t offset_back = -(2 + kNopCount2);  // 2: account for auipc and jialc.
  offset_back <<= 2;
  offset_back += (offset_back & 0x8000) << 1;  // Account for sign extension in jialc.

  // Note, we're using the ".fill" directive to tell the assembler to generate many NOPs
  // instead of generating them ourselves in the source code. This saves a few minutes
  // of test time.
  std::ostringstream oss;
  oss <<
      ".set noreorder\n"
      "auipc $at, 0x" << std::hex << High16Bits(offset_forward1) << "\n"
      "jialc $at, 0x" << std::hex << Low16Bits(offset_forward1) << "\n"
      ".fill 0x" << std::hex << kNopCount1 << " , 4, 0\n"
      "1:\n"
      "auipc $at, 0x" << std::hex << High16Bits(offset_forward2) << "\n"
      "jialc $at, 0x" << std::hex << Low16Bits(offset_forward2) << "\n"
      ".fill 0x" << std::hex << kNopCount2 << " , 4, 0\n"
      "2:\n"
      "auipc $at, 0x" << std::hex << High16Bits(offset_back) << "\n"
      "jialc $at, 0x" << std::hex << Low16Bits(offset_back) << "\n";
  std::string expected = oss.str();
  DriverStr(expected, "LongBalc");
}

//////////
// MISC //
//////////

TEST_F(AssemblerRISCV64Test, Lwpc) {
  // Lwpc() takes an unsigned 19-bit immediate, while the GNU assembler needs a signed offset,
  // hence the sign extension from bit 18 with `imm - ((imm & 0x40000) << 1)`.
  // The GNU assembler also wants the offset to be a multiple of 4, which it will shift right
  // by 2 positions when encoding, hence `<< 2` to compensate for that shift.
  // We capture the value of the immediate with `.set imm, {imm}` because the value is needed
  // twice for the sign extension, but `{imm}` is substituted only once.
  const char* code = ".set imm, {imm}\nlw ${reg}, ((imm - ((imm & 0x40000) << 1)) << 2)($pc)";
  DriverStr(RepeatRIb(&riscv64::Riscv64Assembler::Lwpc, 19, code), "Lwpc");
}

TEST_F(AssemblerRISCV64Test, Lwupc) {
  // The comment for the Lwpc test applies here as well.
  const char* code = ".set imm, {imm}\nlwu ${reg}, ((imm - ((imm & 0x40000) << 1)) << 2)($pc)";
  DriverStr(RepeatRIb(&riscv64::Riscv64Assembler::Lwupc, 19, code), "Lwupc");
}

TEST_F(AssemblerRISCV64Test, Ldpc) {
  // The comment for the Lwpc test applies here as well.
  const char* code = ".set imm, {imm}\nld ${reg}, ((imm - ((imm & 0x20000) << 1)) << 3)($pc)";
  DriverStr(RepeatRIb(&riscv64::Riscv64Assembler::Ldpc, 18, code), "Ldpc");
}

TEST_F(AssemblerRISCV64Test, Auipc) {
  DriverStr(RepeatRIb(&riscv64::Riscv64Assembler::Auipc, 16, "auipc ${reg}, {imm}"), "Auipc");
}

TEST_F(AssemblerRISCV64Test, Addiupc) {
  // The comment from the Lwpc() test applies to this Addiupc() test as well.
  const char* code = ".set imm, {imm}\naddiupc ${reg}, (imm - ((imm & 0x40000) << 1)) << 2";
  DriverStr(RepeatRIb(&riscv64::Riscv64Assembler::Addiupc, 19, code), "Addiupc");
}

TEST_F(AssemblerRISCV64Test, Addu) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Addu, "addu ${reg1}, ${reg2}, ${reg3}"), "addu");
}

TEST_F(AssemblerRISCV64Test, Addiu) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Addiu, -16, "addiu ${reg1}, ${reg2}, {imm}"),
            "addiu");
}

TEST_F(AssemblerRISCV64Test, Daddu) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Daddu, "daddu ${reg1}, ${reg2}, ${reg3}"), "daddu");
}

TEST_F(AssemblerRISCV64Test, Daddiu) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Daddiu, -16, "daddiu ${reg1}, ${reg2}, {imm}"),
            "daddiu");
}

TEST_F(AssemblerRISCV64Test, Subu) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Subu, "subu ${reg1}, ${reg2}, ${reg3}"), "subu");
}

TEST_F(AssemblerRISCV64Test, Dsubu) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Dsubu, "dsubu ${reg1}, ${reg2}, ${reg3}"), "dsubu");
}

TEST_F(AssemblerRISCV64Test, MulR6) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::MulR6, "mul ${reg1}, ${reg2}, ${reg3}"), "mulR6");
}

TEST_F(AssemblerRISCV64Test, DivR6) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::DivR6, "div ${reg1}, ${reg2}, ${reg3}"), "divR6");
}

TEST_F(AssemblerRISCV64Test, ModR6) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::ModR6, "mod ${reg1}, ${reg2}, ${reg3}"), "modR6");
}

TEST_F(AssemblerRISCV64Test, DivuR6) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::DivuR6, "divu ${reg1}, ${reg2}, ${reg3}"),
            "divuR6");
}

TEST_F(AssemblerRISCV64Test, ModuR6) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::ModuR6, "modu ${reg1}, ${reg2}, ${reg3}"),
            "moduR6");
}

TEST_F(AssemblerRISCV64Test, Dmul) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Dmul, "dmul ${reg1}, ${reg2}, ${reg3}"), "dmul");
}

TEST_F(AssemblerRISCV64Test, Ddiv) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Ddiv, "ddiv ${reg1}, ${reg2}, ${reg3}"), "ddiv");
}

TEST_F(AssemblerRISCV64Test, Dmod) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Dmod, "dmod ${reg1}, ${reg2}, ${reg3}"), "dmod");
}

TEST_F(AssemblerRISCV64Test, Ddivu) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Ddivu, "ddivu ${reg1}, ${reg2}, ${reg3}"), "ddivu");
}

TEST_F(AssemblerRISCV64Test, Dmodu) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Dmodu, "dmodu ${reg1}, ${reg2}, ${reg3}"), "dmodu");
}

TEST_F(AssemblerRISCV64Test, And) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::And, "and ${reg1}, ${reg2}, ${reg3}"), "and");
}

TEST_F(AssemblerRISCV64Test, Andi) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Andi, 16, "andi ${reg1}, ${reg2}, {imm}"), "andi");
}

TEST_F(AssemblerRISCV64Test, Or) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Or, "or ${reg1}, ${reg2}, ${reg3}"), "or");
}

TEST_F(AssemblerRISCV64Test, Ori) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Ori, 16, "ori ${reg1}, ${reg2}, {imm}"), "ori");
}

TEST_F(AssemblerRISCV64Test, Xor) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Xor, "xor ${reg1}, ${reg2}, ${reg3}"), "xor");
}

TEST_F(AssemblerRISCV64Test, Xori) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Xori, 16, "xori ${reg1}, ${reg2}, {imm}"), "xori");
}

TEST_F(AssemblerRISCV64Test, Nor) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Nor, "nor ${reg1}, ${reg2}, ${reg3}"), "nor");
}

TEST_F(AssemblerRISCV64Test, Lb) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Lb, -16, "lb ${reg1}, {imm}(${reg2})"), "lb");
}

TEST_F(AssemblerRISCV64Test, Lh) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Lh, -16, "lh ${reg1}, {imm}(${reg2})"), "lh");
}

TEST_F(AssemblerRISCV64Test, Lw) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Lw, -16, "lw ${reg1}, {imm}(${reg2})"), "lw");
}

TEST_F(AssemblerRISCV64Test, Ld) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Ld, -16, "ld ${reg1}, {imm}(${reg2})"), "ld");
}

TEST_F(AssemblerRISCV64Test, Lbu) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Lbu, -16, "lbu ${reg1}, {imm}(${reg2})"), "lbu");
}

TEST_F(AssemblerRISCV64Test, Lhu) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Lhu, -16, "lhu ${reg1}, {imm}(${reg2})"), "lhu");
}

TEST_F(AssemblerRISCV64Test, Lwu) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Lwu, -16, "lwu ${reg1}, {imm}(${reg2})"), "lwu");
}

TEST_F(AssemblerRISCV64Test, Lui) {
  DriverStr(RepeatRIb(&riscv64::Riscv64Assembler::Lui, 16, "lui ${reg}, {imm}"), "lui");
}

TEST_F(AssemblerRISCV64Test, Daui) {
  std::vector<riscv64::GpuRegister*> reg1_registers = GetRegisters();
  std::vector<riscv64::GpuRegister*> reg2_registers = GetRegisters();
  reg2_registers.erase(reg2_registers.begin());  // reg2 can't be ZERO, remove it.
  std::vector<int64_t> imms = CreateImmediateValuesBits(/* imm_bits= */ 16, /* as_uint= */ true);
  WarnOnCombinations(reg1_registers.size() * reg2_registers.size() * imms.size());
  std::ostringstream expected;
  for (riscv64::GpuRegister* reg1 : reg1_registers) {
    for (riscv64::GpuRegister* reg2 : reg2_registers) {
      for (int64_t imm : imms) {
        __ Daui(*reg1, *reg2, imm);
        expected << "daui $" << *reg1 << ", $" << *reg2 << ", " << imm << "\n";
      }
    }
  }
  DriverStr(expected.str(), "daui");
}

TEST_F(AssemblerRISCV64Test, Dahi) {
  DriverStr(RepeatRIb(&riscv64::Riscv64Assembler::Dahi, 16, "dahi ${reg}, ${reg}, {imm}"), "dahi");
}

TEST_F(AssemblerRISCV64Test, Dati) {
  DriverStr(RepeatRIb(&riscv64::Riscv64Assembler::Dati, 16, "dati ${reg}, ${reg}, {imm}"), "dati");
}

TEST_F(AssemblerRISCV64Test, Sb) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Sb, -16, "sb ${reg1}, {imm}(${reg2})"), "sb");
}

TEST_F(AssemblerRISCV64Test, Sh) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Sh, -16, "sh ${reg1}, {imm}(${reg2})"), "sh");
}

TEST_F(AssemblerRISCV64Test, Sw) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Sw, -16, "sw ${reg1}, {imm}(${reg2})"), "sw");
}

TEST_F(AssemblerRISCV64Test, Sd) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Sd, -16, "sd ${reg1}, {imm}(${reg2})"), "sd");
}

TEST_F(AssemblerRISCV64Test, Slt) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Slt, "slt ${reg1}, ${reg2}, ${reg3}"), "slt");
}

TEST_F(AssemblerRISCV64Test, Sltu) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Sltu, "sltu ${reg1}, ${reg2}, ${reg3}"), "sltu");
}

TEST_F(AssemblerRISCV64Test, Slti) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Slti, -16, "slti ${reg1}, ${reg2}, {imm}"),
            "slti");
}

TEST_F(AssemblerRISCV64Test, Sltiu) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Sltiu, -16, "sltiu ${reg1}, ${reg2}, {imm}"),
            "sltiu");
}

TEST_F(AssemblerRISCV64Test, Move) {
  DriverStr(RepeatRR(&riscv64::Riscv64Assembler::Move, "or ${reg1}, ${reg2}, $zero"), "move");
}

TEST_F(AssemblerRISCV64Test, Clear) {
  DriverStr(RepeatR(&riscv64::Riscv64Assembler::Clear, "or ${reg}, $zero, $zero"), "clear");
}

TEST_F(AssemblerRISCV64Test, Not) {
  DriverStr(RepeatRR(&riscv64::Riscv64Assembler::Not, "nor ${reg1}, ${reg2}, $zero"), "not");
}

TEST_F(AssemblerRISCV64Test, Bitswap) {
  DriverStr(RepeatRR(&riscv64::Riscv64Assembler::Bitswap, "bitswap ${reg1}, ${reg2}"), "bitswap");
}

TEST_F(AssemblerRISCV64Test, Dbitswap) {
  DriverStr(RepeatRR(&riscv64::Riscv64Assembler::Dbitswap, "dbitswap ${reg1}, ${reg2}"), "dbitswap");
}

TEST_F(AssemblerRISCV64Test, Seb) {
  DriverStr(RepeatRR(&riscv64::Riscv64Assembler::Seb, "seb ${reg1}, ${reg2}"), "seb");
}

TEST_F(AssemblerRISCV64Test, Seh) {
  DriverStr(RepeatRR(&riscv64::Riscv64Assembler::Seh, "seh ${reg1}, ${reg2}"), "seh");
}

TEST_F(AssemblerRISCV64Test, Dsbh) {
  DriverStr(RepeatRR(&riscv64::Riscv64Assembler::Dsbh, "dsbh ${reg1}, ${reg2}"), "dsbh");
}

TEST_F(AssemblerRISCV64Test, Dshd) {
  DriverStr(RepeatRR(&riscv64::Riscv64Assembler::Dshd, "dshd ${reg1}, ${reg2}"), "dshd");
}

TEST_F(AssemblerRISCV64Test, Dext) {
  std::vector<riscv64::GpuRegister*> reg1_registers = GetRegisters();
  std::vector<riscv64::GpuRegister*> reg2_registers = GetRegisters();
  WarnOnCombinations(reg1_registers.size() * reg2_registers.size() * 33 * 16);
  std::ostringstream expected;
  for (riscv64::GpuRegister* reg1 : reg1_registers) {
    for (riscv64::GpuRegister* reg2 : reg2_registers) {
      for (int32_t pos = 0; pos < 32; pos++) {
        for (int32_t size = 1; size <= 32; size++) {
          __ Dext(*reg1, *reg2, pos, size);
          expected << "dext $" << *reg1 << ", $" << *reg2 << ", " << pos << ", " << size << "\n";
        }
      }
    }
  }

  DriverStr(expected.str(), "Dext");
}

TEST_F(AssemblerRISCV64Test, Ins) {
  std::vector<riscv64::GpuRegister*> regs = GetRegisters();
  WarnOnCombinations(regs.size() * regs.size() * 33 * 16);
  std::string expected;
  for (riscv64::GpuRegister* reg1 : regs) {
    for (riscv64::GpuRegister* reg2 : regs) {
      for (int32_t pos = 0; pos < 32; pos++) {
        for (int32_t size = 1; pos + size <= 32; size++) {
          __ Ins(*reg1, *reg2, pos, size);
          std::ostringstream instr;
          instr << "ins $" << *reg1 << ", $" << *reg2 << ", " << pos << ", " << size << "\n";
          expected += instr.str();
        }
      }
    }
  }
  DriverStr(expected, "Ins");
}

TEST_F(AssemblerRISCV64Test, DblIns) {
  std::vector<riscv64::GpuRegister*> reg1_registers = GetRegisters();
  std::vector<riscv64::GpuRegister*> reg2_registers = GetRegisters();
  WarnOnCombinations(reg1_registers.size() * reg2_registers.size() * 65 * 32);
  std::ostringstream expected;
  for (riscv64::GpuRegister* reg1 : reg1_registers) {
    for (riscv64::GpuRegister* reg2 : reg2_registers) {
      for (int32_t pos = 0; pos < 64; pos++) {
        for (int32_t size = 1; pos + size <= 64; size++) {
          __ DblIns(*reg1, *reg2, pos, size);
          expected << "dins $" << *reg1 << ", $" << *reg2 << ", " << pos << ", " << size << "\n";
        }
      }
    }
  }

  DriverStr(expected.str(), "DblIns");
}

TEST_F(AssemblerRISCV64Test, Lsa) {
  DriverStr(RepeatRRRIb(&riscv64::Riscv64Assembler::Lsa,
                        2,
                        "lsa ${reg1}, ${reg2}, ${reg3}, {imm}",
                        1),
            "lsa");
}

TEST_F(AssemblerRISCV64Test, Dlsa) {
  DriverStr(RepeatRRRIb(&riscv64::Riscv64Assembler::Dlsa,
                        2,
                        "dlsa ${reg1}, ${reg2}, ${reg3}, {imm}",
                        1),
            "dlsa");
}

TEST_F(AssemblerRISCV64Test, Wsbh) {
  DriverStr(RepeatRR(&riscv64::Riscv64Assembler::Wsbh, "wsbh ${reg1}, ${reg2}"), "wsbh");
}

TEST_F(AssemblerRISCV64Test, Sll) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Sll, 5, "sll ${reg1}, ${reg2}, {imm}"), "sll");
}

TEST_F(AssemblerRISCV64Test, Srl) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Srl, 5, "srl ${reg1}, ${reg2}, {imm}"), "srl");
}

TEST_F(AssemblerRISCV64Test, Rotr) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Rotr, 5, "rotr ${reg1}, ${reg2}, {imm}"), "rotr");
}

TEST_F(AssemblerRISCV64Test, Sra) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Sra, 5, "sra ${reg1}, ${reg2}, {imm}"), "sra");
}

TEST_F(AssemblerRISCV64Test, Sllv) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Sllv, "sllv ${reg1}, ${reg2}, ${reg3}"), "sllv");
}

TEST_F(AssemblerRISCV64Test, Srlv) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Srlv, "srlv ${reg1}, ${reg2}, ${reg3}"), "srlv");
}

TEST_F(AssemblerRISCV64Test, Rotrv) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Rotrv, "rotrv ${reg1}, ${reg2}, ${reg3}"), "rotrv");
}

TEST_F(AssemblerRISCV64Test, Srav) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Srav, "srav ${reg1}, ${reg2}, ${reg3}"), "srav");
}

TEST_F(AssemblerRISCV64Test, Dsll) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Dsll, 5, "dsll ${reg1}, ${reg2}, {imm}"), "dsll");
}

TEST_F(AssemblerRISCV64Test, Dsrl) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Dsrl, 5, "dsrl ${reg1}, ${reg2}, {imm}"), "dsrl");
}

TEST_F(AssemblerRISCV64Test, Drotr) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Drotr, 5, "drotr ${reg1}, ${reg2}, {imm}"),
            "drotr");
}

TEST_F(AssemblerRISCV64Test, Dsra) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Dsra, 5, "dsra ${reg1}, ${reg2}, {imm}"), "dsra");
}

TEST_F(AssemblerRISCV64Test, Dsll32) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Dsll32, 5, "dsll32 ${reg1}, ${reg2}, {imm}"),
            "dsll32");
}

TEST_F(AssemblerRISCV64Test, Dsrl32) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Dsrl32, 5, "dsrl32 ${reg1}, ${reg2}, {imm}"),
            "dsrl32");
}

TEST_F(AssemblerRISCV64Test, Drotr32) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Drotr32, 5, "drotr32 ${reg1}, ${reg2}, {imm}"),
            "drotr32");
}

TEST_F(AssemblerRISCV64Test, Dsra32) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Dsra32, 5, "dsra32 ${reg1}, ${reg2}, {imm}"),
            "dsra32");
}

TEST_F(AssemblerRISCV64Test, Dsllv) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Dsllv, "dsllv ${reg1}, ${reg2}, ${reg3}"), "dsllv");
}

TEST_F(AssemblerRISCV64Test, Dsrlv) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Dsrlv, "dsrlv ${reg1}, ${reg2}, ${reg3}"), "dsrlv");
}

TEST_F(AssemblerRISCV64Test, Dsrav) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Dsrav, "dsrav ${reg1}, ${reg2}, ${reg3}"), "dsrav");
}

TEST_F(AssemblerRISCV64Test, Sc) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Sc, -9, "sc ${reg1}, {imm}(${reg2})"), "sc");
}

TEST_F(AssemblerRISCV64Test, Scd) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Scd, -9, "scd ${reg1}, {imm}(${reg2})"), "scd");
}

TEST_F(AssemblerRISCV64Test, Ll) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Ll, -9, "ll ${reg1}, {imm}(${reg2})"), "ll");
}

TEST_F(AssemblerRISCV64Test, Lld) {
  DriverStr(RepeatRRIb(&riscv64::Riscv64Assembler::Lld, -9, "lld ${reg1}, {imm}(${reg2})"), "lld");
}

TEST_F(AssemblerRISCV64Test, Seleqz) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Seleqz, "seleqz ${reg1}, ${reg2}, ${reg3}"),
            "seleqz");
}

TEST_F(AssemblerRISCV64Test, Selnez) {
  DriverStr(RepeatRRR(&riscv64::Riscv64Assembler::Selnez, "selnez ${reg1}, ${reg2}, ${reg3}"),
            "selnez");
}

TEST_F(AssemblerRISCV64Test, Clz) {
  DriverStr(RepeatRR(&riscv64::Riscv64Assembler::Clz, "clz ${reg1}, ${reg2}"), "clz");
}

TEST_F(AssemblerRISCV64Test, Clo) {
  DriverStr(RepeatRR(&riscv64::Riscv64Assembler::Clo, "clo ${reg1}, ${reg2}"), "clo");
}

TEST_F(AssemblerRISCV64Test, Dclz) {
  DriverStr(RepeatRR(&riscv64::Riscv64Assembler::Dclz, "dclz ${reg1}, ${reg2}"), "dclz");
}

TEST_F(AssemblerRISCV64Test, Dclo) {
  DriverStr(RepeatRR(&riscv64::Riscv64Assembler::Dclo, "dclo ${reg1}, ${reg2}"), "dclo");
}

TEST_F(AssemblerRISCV64Test, LoadFromOffset) {
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A0, 0);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 0);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 1);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 256);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 1000);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 0x7FFF);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 0x8000);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 0x8001);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 0x10000);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 0x12345678);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, -256);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, -32768);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 0xABCDEF00);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 0x7FFFFFFE);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 0x7FFFFFFF);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 0x80000000);
  __ LoadFromOffset(riscv64::kLoadSignedByte, riscv64::A0, riscv64::A1, 0x80000001);

  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A0, 0);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 0);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 1);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 256);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 1000);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 0x7FFF);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 0x8000);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 0x8001);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 0x10000);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 0x12345678);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, -256);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, -32768);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 0xABCDEF00);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 0x7FFFFFFE);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 0x7FFFFFFF);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 0x80000000);
  __ LoadFromOffset(riscv64::kLoadUnsignedByte, riscv64::A0, riscv64::A1, 0x80000001);

  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A0, 0);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 0);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 2);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 256);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 1000);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 0x7FFE);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 0x8000);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 0x8002);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 0x10000);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 0x12345678);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, -256);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, -32768);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 0xABCDEF00);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 0x7FFFFFFC);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 0x7FFFFFFE);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 0x80000000);
  __ LoadFromOffset(riscv64::kLoadSignedHalfword, riscv64::A0, riscv64::A1, 0x80000002);

  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A0, 0);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 0);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 2);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 256);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 1000);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 0x7FFE);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 0x8000);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 0x8002);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 0x10000);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 0x12345678);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, -256);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, -32768);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 0xABCDEF00);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 0x7FFFFFFC);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 0x7FFFFFFE);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 0x80000000);
  __ LoadFromOffset(riscv64::kLoadUnsignedHalfword, riscv64::A0, riscv64::A1, 0x80000002);

  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A0, 0);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 0);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 4);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 256);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 1000);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 0x7FFC);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 0x8000);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 0x8004);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 0x10000);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 0x12345678);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, -256);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, -32768);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 0xABCDEF00);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 0x7FFFFFF8);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 0x7FFFFFFC);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 0x80000000);
  __ LoadFromOffset(riscv64::kLoadWord, riscv64::A0, riscv64::A1, 0x80000004);

  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A0, 0);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 0);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 4);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 256);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 1000);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 0x7FFC);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 0x8000);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 0x8004);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 0x10000);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 0x12345678);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, -256);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, -32768);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 0xABCDEF00);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 0x7FFFFFF8);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 0x7FFFFFFC);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 0x80000000);
  __ LoadFromOffset(riscv64::kLoadUnsignedWord, riscv64::A0, riscv64::A1, 0x80000004);

  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A0, 0);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 0);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 4);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 256);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 1000);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 0x7FFC);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 0x8000);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 0x8004);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 0x10000);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 0x27FFC);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 0x12345678);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, -256);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, -32768);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 0xABCDEF00);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 0x7FFFFFF8);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 0x7FFFFFFC);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 0x80000000);
  __ LoadFromOffset(riscv64::kLoadDoubleword, riscv64::A0, riscv64::A1, 0x80000004);

  const char* expected =
      "lb $a0, 0($a0)\n"
      "lb $a0, 0($a1)\n"
      "lb $a0, 1($a1)\n"
      "lb $a0, 256($a1)\n"
      "lb $a0, 1000($a1)\n"
      "lb $a0, 0x7FFF($a1)\n"
      "daddiu $at, $a1, 0x7FF8\n"
      "lb $a0, 8($at)\n"
      "daddiu $at, $a1, 32760\n"
      "lb $a0, 9($at)\n"
      "daui $at, $a1, 1\n"
      "lb $a0, 0($at)\n"
      "daui $at, $a1, 0x1234\n"
      "lb $a0, 0x5678($at)\n"
      "lb $a0, -256($a1)\n"
      "lb $a0, -32768($a1)\n"
      "daui $at, $a1, 0xABCE\n"
      "lb $a0, -4352($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "lb $a0, -2($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "lb $a0, -1($at)\n"
      "daui $at, $a1, 32768\n"
      "lb $a0, 0($at)\n"
      "daui $at, $a1, 32768\n"
      "lb $a0, 1($at)\n"

      "lbu $a0, 0($a0)\n"
      "lbu $a0, 0($a1)\n"
      "lbu $a0, 1($a1)\n"
      "lbu $a0, 256($a1)\n"
      "lbu $a0, 1000($a1)\n"
      "lbu $a0, 0x7FFF($a1)\n"
      "daddiu $at, $a1, 0x7FF8\n"
      "lbu $a0, 8($at)\n"
      "daddiu $at, $a1, 32760\n"
      "lbu $a0, 9($at)\n"
      "daui $at, $a1, 1\n"
      "lbu $a0, 0($at)\n"
      "daui $at, $a1, 0x1234\n"
      "lbu $a0, 0x5678($at)\n"
      "lbu $a0, -256($a1)\n"
      "lbu $a0, -32768($a1)\n"
      "daui $at, $a1, 0xABCE\n"
      "lbu $a0, -4352($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "lbu $a0, -2($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "lbu $a0, -1($at)\n"
      "daui $at, $a1, 32768\n"
      "lbu $a0, 0($at)\n"
      "daui $at, $a1, 32768\n"
      "lbu $a0, 1($at)\n"

      "lh $a0, 0($a0)\n"
      "lh $a0, 0($a1)\n"
      "lh $a0, 2($a1)\n"
      "lh $a0, 256($a1)\n"
      "lh $a0, 1000($a1)\n"
      "lh $a0, 0x7FFE($a1)\n"
      "daddiu $at, $a1, 0x7FF8\n"
      "lh $a0, 8($at)\n"
      "daddiu $at, $a1, 32760\n"
      "lh $a0, 10($at)\n"
      "daui $at, $a1, 1\n"
      "lh $a0, 0($at)\n"
      "daui $at, $a1, 0x1234\n"
      "lh $a0, 0x5678($at)\n"
      "lh $a0, -256($a1)\n"
      "lh $a0, -32768($a1)\n"
      "daui $at, $a1, 0xABCE\n"
      "lh $a0, -4352($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "lh $a0, -4($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "lh $a0, -2($at)\n"
      "daui $at, $a1, 32768\n"
      "lh $a0, 0($at)\n"
      "daui $at, $a1, 32768\n"
      "lh $a0, 2($at)\n"

      "lhu $a0, 0($a0)\n"
      "lhu $a0, 0($a1)\n"
      "lhu $a0, 2($a1)\n"
      "lhu $a0, 256($a1)\n"
      "lhu $a0, 1000($a1)\n"
      "lhu $a0, 0x7FFE($a1)\n"
      "daddiu $at, $a1, 0x7FF8\n"
      "lhu $a0, 8($at)\n"
      "daddiu $at, $a1, 32760\n"
      "lhu $a0, 10($at)\n"
      "daui $at, $a1, 1\n"
      "lhu $a0, 0($at)\n"
      "daui $at, $a1, 0x1234\n"
      "lhu $a0, 0x5678($at)\n"
      "lhu $a0, -256($a1)\n"
      "lhu $a0, -32768($a1)\n"
      "daui $at, $a1, 0xABCE\n"
      "lhu $a0, -4352($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "lhu $a0, -4($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "lhu $a0, -2($at)\n"
      "daui $at, $a1, 32768\n"
      "lhu $a0, 0($at)\n"
      "daui $at, $a1, 32768\n"
      "lhu $a0, 2($at)\n"

      "lw $a0, 0($a0)\n"
      "lw $a0, 0($a1)\n"
      "lw $a0, 4($a1)\n"
      "lw $a0, 256($a1)\n"
      "lw $a0, 1000($a1)\n"
      "lw $a0, 0x7FFC($a1)\n"
      "daddiu $at, $a1, 0x7FF8\n"
      "lw $a0, 8($at)\n"
      "daddiu $at, $a1, 32760\n"
      "lw $a0, 12($at)\n"
      "daui $at, $a1, 1\n"
      "lw $a0, 0($at)\n"
      "daui $at, $a1, 0x1234\n"
      "lw $a0, 0x5678($at)\n"
      "lw $a0, -256($a1)\n"
      "lw $a0, -32768($a1)\n"
      "daui $at, $a1, 0xABCE\n"
      "lw $a0, -4352($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "lw $a0, -8($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "lw $a0, -4($at)\n"
      "daui $at, $a1, 32768\n"
      "lw $a0, 0($at)\n"
      "daui $at, $a1, 32768\n"
      "lw $a0, 4($at)\n"

      "lwu $a0, 0($a0)\n"
      "lwu $a0, 0($a1)\n"
      "lwu $a0, 4($a1)\n"
      "lwu $a0, 256($a1)\n"
      "lwu $a0, 1000($a1)\n"
      "lwu $a0, 0x7FFC($a1)\n"
      "daddiu $at, $a1, 0x7FF8\n"
      "lwu $a0, 8($at)\n"
      "daddiu $at, $a1, 32760\n"
      "lwu $a0, 12($at)\n"
      "daui $at, $a1, 1\n"
      "lwu $a0, 0($at)\n"
      "daui $at, $a1, 0x1234\n"
      "lwu $a0, 0x5678($at)\n"
      "lwu $a0, -256($a1)\n"
      "lwu $a0, -32768($a1)\n"
      "daui $at, $a1, 0xABCE\n"
      "lwu $a0, -4352($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "lwu $a0, -8($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "lwu $a0, -4($at)\n"
      "daui $at, $a1, 32768\n"
      "lwu $a0, 0($at)\n"
      "daui $at, $a1, 32768\n"
      "lwu $a0, 4($at)\n"

      "ld $a0, 0($a0)\n"
      "ld $a0, 0($a1)\n"
      "lwu $a0, 4($a1)\n"
      "lwu $t3, 8($a1)\n"
      "dinsu $a0, $t3, 32, 32\n"
      "ld $a0, 256($a1)\n"
      "ld $a0, 1000($a1)\n"
      "daddiu $at, $a1, 32760\n"
      "lwu $a0, 4($at)\n"
      "lwu $t3, 8($at)\n"
      "dinsu $a0, $t3, 32, 32\n"
      "daddiu $at, $a1, 32760\n"
      "ld $a0, 8($at)\n"
      "daddiu $at, $a1, 32760\n"
      "lwu $a0, 12($at)\n"
      "lwu $t3, 16($at)\n"
      "dinsu $a0, $t3, 32, 32\n"
      "daui $at, $a1, 1\n"
      "ld $a0, 0($at)\n"
      "daui $at, $a1, 2\n"
      "daddiu $at, $at, 8\n"
      "lwu $a0, 0x7ff4($at)\n"
      "lwu $t3, 0x7ff8($at)\n"
      "dinsu $a0, $t3, 32, 32\n"
      "daui $at, $a1, 0x1234\n"
      "ld $a0, 0x5678($at)\n"
      "ld $a0, -256($a1)\n"
      "ld $a0, -32768($a1)\n"
      "daui $at, $a1, 0xABCE\n"
      "ld $a0, -4352($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "ld $a0, -8($at)\n"
      "daui $at, $a1, 32768\n"
      "dahi $at, $at, 1\n"
      "lwu $a0, -4($at)\n"
      "lwu $t3, 0($at)\n"
      "dinsu $a0, $t3, 32, 32\n"
      "daui $at, $a1, 32768\n"
      "ld $a0, 0($at)\n"
      "daui $at, $a1, 32768\n"
      "lwu $a0, 4($at)\n"
      "lwu $t3, 8($at)\n"
      "dinsu $a0, $t3, 32, 32\n";
  DriverStr(expected, "LoadFromOffset");
}

TEST_F(AssemblerRISCV64Test, LoadFpuFromOffset) {
  __ LoadFpuFromOffset(riscv64::kLoadWord, riscv64::F0, riscv64::A0, 0);
  __ LoadFpuFromOffset(riscv64::kLoadWord, riscv64::F0, riscv64::A0, 4);
  __ LoadFpuFromOffset(riscv64::kLoadWord, riscv64::F0, riscv64::A0, 256);
  __ LoadFpuFromOffset(riscv64::kLoadWord, riscv64::F0, riscv64::A0, 0x7FFC);
  __ LoadFpuFromOffset(riscv64::kLoadWord, riscv64::F0, riscv64::A0, 0x8000);
  __ LoadFpuFromOffset(riscv64::kLoadWord, riscv64::F0, riscv64::A0, 0x8004);
  __ LoadFpuFromOffset(riscv64::kLoadWord, riscv64::F0, riscv64::A0, 0x10000);
  __ LoadFpuFromOffset(riscv64::kLoadWord, riscv64::F0, riscv64::A0, 0x12345678);
  __ LoadFpuFromOffset(riscv64::kLoadWord, riscv64::F0, riscv64::A0, -256);
  __ LoadFpuFromOffset(riscv64::kLoadWord, riscv64::F0, riscv64::A0, -32768);
  __ LoadFpuFromOffset(riscv64::kLoadWord, riscv64::F0, riscv64::A0, 0xABCDEF00);

  __ LoadFpuFromOffset(riscv64::kLoadDoubleword, riscv64::F0, riscv64::A0, 0);
  __ LoadFpuFromOffset(riscv64::kLoadDoubleword, riscv64::F0, riscv64::A0, 4);
  __ LoadFpuFromOffset(riscv64::kLoadDoubleword, riscv64::F0, riscv64::A0, 256);
  __ LoadFpuFromOffset(riscv64::kLoadDoubleword, riscv64::F0, riscv64::A0, 0x7FFC);
  __ LoadFpuFromOffset(riscv64::kLoadDoubleword, riscv64::F0, riscv64::A0, 0x8000);
  __ LoadFpuFromOffset(riscv64::kLoadDoubleword, riscv64::F0, riscv64::A0, 0x8004);
  __ LoadFpuFromOffset(riscv64::kLoadDoubleword, riscv64::F0, riscv64::A0, 0x10000);
  __ LoadFpuFromOffset(riscv64::kLoadDoubleword, riscv64::F0, riscv64::A0, 0x12345678);
  __ LoadFpuFromOffset(riscv64::kLoadDoubleword, riscv64::F0, riscv64::A0, -256);
  __ LoadFpuFromOffset(riscv64::kLoadDoubleword, riscv64::F0, riscv64::A0, -32768);
  __ LoadFpuFromOffset(riscv64::kLoadDoubleword, riscv64::F0, riscv64::A0, 0xABCDEF00);

  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 0);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 1);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 2);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 4);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 8);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 511);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 512);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 513);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 514);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 516);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 1022);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 1024);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 1025);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 1026);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 1028);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 2044);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 2048);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 2049);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 2050);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 2052);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 4088);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 4096);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 4097);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 4098);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 4100);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 4104);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 0x7FFC);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 0x8000);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 0x10000);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 0x12345678);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 0x12350078);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, -256);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, -511);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, -513);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, -1022);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, -1026);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, -2044);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, -2052);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, -4096);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, -4104);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, -32768);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 0xABCDEF00);
  __ LoadFpuFromOffset(riscv64::kLoadQuadword, riscv64::F0, riscv64::A0, 0x7FFFABCD);

  const char* expected =
      "lwc1 $f0, 0($a0)\n"
      "lwc1 $f0, 4($a0)\n"
      "lwc1 $f0, 256($a0)\n"
      "lwc1 $f0, 0x7FFC($a0)\n"
      "daddiu $at, $a0, 32760 # 0x7FF8\n"
      "lwc1 $f0, 8($at)\n"
      "daddiu $at, $a0, 32760 # 0x7FF8\n"
      "lwc1 $f0, 12($at)\n"
      "daui $at, $a0, 1\n"
      "lwc1 $f0, 0($at)\n"
      "daui $at, $a0, 4660 # 0x1234\n"
      "lwc1 $f0, 22136($at) # 0x5678\n"
      "lwc1 $f0, -256($a0)\n"
      "lwc1 $f0, -32768($a0)\n"
      "daui $at, $a0, 0xABCE\n"
      "lwc1 $f0, -0x1100($at) # 0xEF00\n"

      "ldc1 $f0, 0($a0)\n"
      "lwc1 $f0, 4($a0)\n"
      "lw $t3, 8($a0)\n"
      "mthc1 $t3, $f0\n"
      "ldc1 $f0, 256($a0)\n"
      "daddiu $at, $a0, 32760 # 0x7FF8\n"
      "lwc1 $f0, 4($at)\n"
      "lw $t3, 8($at)\n"
      "mthc1 $t3, $f0\n"
      "daddiu $at, $a0, 32760 # 0x7FF8\n"
      "ldc1 $f0, 8($at)\n"
      "daddiu $at, $a0, 32760 # 0x7FF8\n"
      "lwc1 $f0, 12($at)\n"
      "lw $t3, 16($at)\n"
      "mthc1 $t3, $f0\n"
      "daui $at, $a0, 1\n"
      "ldc1 $f0, 0($at)\n"
      "daui $at, $a0, 4660 # 0x1234\n"
      "ldc1 $f0, 22136($at) # 0x5678\n"
      "ldc1 $f0, -256($a0)\n"
      "ldc1 $f0, -32768($a0)\n"
      "daui $at, $a0, 0xABCE\n"
      "ldc1 $f0, -0x1100($at) # 0xEF00\n"

      "ld.d $w0, 0($a0)\n"
      "ld.b $w0, 1($a0)\n"
      "ld.h $w0, 2($a0)\n"
      "ld.w $w0, 4($a0)\n"
      "ld.d $w0, 8($a0)\n"
      "ld.b $w0, 511($a0)\n"
      "ld.d $w0, 512($a0)\n"
      "daddiu $at, $a0, 513\n"
      "ld.b $w0, 0($at)\n"
      "ld.h $w0, 514($a0)\n"
      "ld.w $w0, 516($a0)\n"
      "ld.h $w0, 1022($a0)\n"
      "ld.d $w0, 1024($a0)\n"
      "daddiu $at, $a0, 1025\n"
      "ld.b $w0, 0($at)\n"
      "daddiu $at, $a0, 1026\n"
      "ld.h $w0, 0($at)\n"
      "ld.w $w0, 1028($a0)\n"
      "ld.w $w0, 2044($a0)\n"
      "ld.d $w0, 2048($a0)\n"
      "daddiu $at, $a0, 2049\n"
      "ld.b $w0, 0($at)\n"
      "daddiu $at, $a0, 2050\n"
      "ld.h $w0, 0($at)\n"
      "daddiu $at, $a0, 2052\n"
      "ld.w $w0, 0($at)\n"
      "ld.d $w0, 4088($a0)\n"
      "daddiu $at, $a0, 4096\n"
      "ld.d $w0, 0($at)\n"
      "daddiu $at, $a0, 4097\n"
      "ld.b $w0, 0($at)\n"
      "daddiu $at, $a0, 4098\n"
      "ld.h $w0, 0($at)\n"
      "daddiu $at, $a0, 4100\n"
      "ld.w $w0, 0($at)\n"
      "daddiu $at, $a0, 4104\n"
      "ld.d $w0, 0($at)\n"
      "daddiu $at, $a0, 0x7FFC\n"
      "ld.w $w0, 0($at)\n"
      "daddiu $at, $a0, 0x7FF8\n"
      "ld.d $w0, 8($at)\n"
      "daui $at, $a0, 0x1\n"
      "ld.d $w0, 0($at)\n"
      "daui $at, $a0, 0x1234\n"
      "daddiu $at, $at, 0x6000\n"
      "ld.d $w0, -2440($at) # 0xF678\n"
      "daui $at, $a0, 0x1235\n"
      "ld.d $w0, 0x78($at)\n"
      "ld.d $w0, -256($a0)\n"
      "ld.b $w0, -511($a0)\n"
      "daddiu $at, $a0, -513\n"
      "ld.b $w0, 0($at)\n"
      "ld.h $w0, -1022($a0)\n"
      "daddiu $at, $a0, -1026\n"
      "ld.h $w0, 0($at)\n"
      "ld.w $w0, -2044($a0)\n"
      "daddiu $at, $a0, -2052\n"
      "ld.w $w0, 0($at)\n"
      "ld.d $w0, -4096($a0)\n"
      "daddiu $at, $a0, -4104\n"
      "ld.d $w0, 0($at)\n"
      "daddiu $at, $a0, -32768\n"
      "ld.d $w0, 0($at)\n"
      "daui $at, $a0, 0xABCE\n"
      "daddiu $at, $at, -8192 # 0xE000\n"
      "ld.d $w0, 0xF00($at)\n"
      "daui $at, $a0, 0x8000\n"
      "dahi $at, $at, 1\n"
      "daddiu $at, $at, -21504 # 0xAC00\n"
      "ld.b $w0, -51($at) # 0xFFCD\n";
  DriverStr(expected, "LoadFpuFromOffset");
}

TEST_F(AssemblerRISCV64Test, StoreToOffset) {
  __ StoreToOffset(riscv64::kStoreByte, riscv64::A0, riscv64::A0, 0);
  __ StoreToOffset(riscv64::kStoreByte, riscv64::A0, riscv64::A1, 0);
  __ StoreToOffset(riscv64::kStoreByte, riscv64::A0, riscv64::A1, 1);
  __ StoreToOffset(riscv64::kStoreByte, riscv64::A0, riscv64::A1, 256);
  __ StoreToOffset(riscv64::kStoreByte, riscv64::A0, riscv64::A1, 1000);
  __ StoreToOffset(riscv64::kStoreByte, riscv64::A0, riscv64::A1, 0x7FFF);
  __ StoreToOffset(riscv64::kStoreByte, riscv64::A0, riscv64::A1, 0x8000);
  __ StoreToOffset(riscv64::kStoreByte, riscv64::A0, riscv64::A1, 0x8001);
  __ StoreToOffset(riscv64::kStoreByte, riscv64::A0, riscv64::A1, 0x10000);
  __ StoreToOffset(riscv64::kStoreByte, riscv64::A0, riscv64::A1, 0x12345678);
  __ StoreToOffset(riscv64::kStoreByte, riscv64::A0, riscv64::A1, -256);
  __ StoreToOffset(riscv64::kStoreByte, riscv64::A0, riscv64::A1, -32768);
  __ StoreToOffset(riscv64::kStoreByte, riscv64::A0, riscv64::A1, 0xABCDEF00);

  __ StoreToOffset(riscv64::kStoreHalfword, riscv64::A0, riscv64::A0, 0);
  __ StoreToOffset(riscv64::kStoreHalfword, riscv64::A0, riscv64::A1, 0);
  __ StoreToOffset(riscv64::kStoreHalfword, riscv64::A0, riscv64::A1, 2);
  __ StoreToOffset(riscv64::kStoreHalfword, riscv64::A0, riscv64::A1, 256);
  __ StoreToOffset(riscv64::kStoreHalfword, riscv64::A0, riscv64::A1, 1000);
  __ StoreToOffset(riscv64::kStoreHalfword, riscv64::A0, riscv64::A1, 0x7FFE);
  __ StoreToOffset(riscv64::kStoreHalfword, riscv64::A0, riscv64::A1, 0x8000);
  __ StoreToOffset(riscv64::kStoreHalfword, riscv64::A0, riscv64::A1, 0x8002);
  __ StoreToOffset(riscv64::kStoreHalfword, riscv64::A0, riscv64::A1, 0x10000);
  __ StoreToOffset(riscv64::kStoreHalfword, riscv64::A0, riscv64::A1, 0x12345678);
  __ StoreToOffset(riscv64::kStoreHalfword, riscv64::A0, riscv64::A1, -256);
  __ StoreToOffset(riscv64::kStoreHalfword, riscv64::A0, riscv64::A1, -32768);
  __ StoreToOffset(riscv64::kStoreHalfword, riscv64::A0, riscv64::A1, 0xABCDEF00);

  __ StoreToOffset(riscv64::kStoreWord, riscv64::A0, riscv64::A0, 0);
  __ StoreToOffset(riscv64::kStoreWord, riscv64::A0, riscv64::A1, 0);
  __ StoreToOffset(riscv64::kStoreWord, riscv64::A0, riscv64::A1, 4);
  __ StoreToOffset(riscv64::kStoreWord, riscv64::A0, riscv64::A1, 256);
  __ StoreToOffset(riscv64::kStoreWord, riscv64::A0, riscv64::A1, 1000);
  __ StoreToOffset(riscv64::kStoreWord, riscv64::A0, riscv64::A1, 0x7FFC);
  __ StoreToOffset(riscv64::kStoreWord, riscv64::A0, riscv64::A1, 0x8000);
  __ StoreToOffset(riscv64::kStoreWord, riscv64::A0, riscv64::A1, 0x8004);
  __ StoreToOffset(riscv64::kStoreWord, riscv64::A0, riscv64::A1, 0x10000);
  __ StoreToOffset(riscv64::kStoreWord, riscv64::A0, riscv64::A1, 0x12345678);
  __ StoreToOffset(riscv64::kStoreWord, riscv64::A0, riscv64::A1, -256);
  __ StoreToOffset(riscv64::kStoreWord, riscv64::A0, riscv64::A1, -32768);
  __ StoreToOffset(riscv64::kStoreWord, riscv64::A0, riscv64::A1, 0xABCDEF00);

  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A0, 0);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 0);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 4);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 256);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 1000);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 0x7FFC);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 0x8000);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 0x8004);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 0x10000);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 0x12345678);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, -256);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, -32768);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 0xABCDEF00);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 0x7FFFFFF8);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 0x7FFFFFFC);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 0x80000000);
  __ StoreToOffset(riscv64::kStoreDoubleword, riscv64::A0, riscv64::A1, 0x80000004);

  const char* expected =
      "sb $a0, 0($a0)\n"
      "sb $a0, 0($a1)\n"
      "sb $a0, 1($a1)\n"
      "sb $a0, 256($a1)\n"
      "sb $a0, 1000($a1)\n"
      "sb $a0, 0x7FFF($a1)\n"
      "daddiu $at, $a1, 0x7FF8\n"
      "sb $a0, 8($at)\n"
      "daddiu $at, $a1, 0x7FF8\n"
      "sb $a0, 9($at)\n"
      "daui $at, $a1, 1\n"
      "sb $a0, 0($at)\n"
      "daui $at, $a1, 4660 # 0x1234\n"
      "sb $a0, 22136($at) # 0x5678\n"
      "sb $a0, -256($a1)\n"
      "sb $a0, -32768($a1)\n"
      "daui $at, $a1, 43982 # 0xABCE\n"
      "sb $a0, -4352($at) # 0xEF00\n"

      "sh $a0, 0($a0)\n"
      "sh $a0, 0($a1)\n"
      "sh $a0, 2($a1)\n"
      "sh $a0, 256($a1)\n"
      "sh $a0, 1000($a1)\n"
      "sh $a0, 0x7FFE($a1)\n"
      "daddiu $at, $a1, 0x7FF8\n"
      "sh $a0, 8($at)\n"
      "daddiu $at, $a1, 0x7FF8\n"
      "sh $a0, 10($at)\n"
      "daui $at, $a1, 1\n"
      "sh $a0, 0($at)\n"
      "daui $at, $a1, 4660 # 0x1234\n"
      "sh $a0, 22136($at) # 0x5678\n"
      "sh $a0, -256($a1)\n"
      "sh $a0, -32768($a1)\n"
      "daui $at, $a1, 43982 # 0xABCE\n"
      "sh $a0, -4352($at) # 0xEF00\n"

      "sw $a0, 0($a0)\n"
      "sw $a0, 0($a1)\n"
      "sw $a0, 4($a1)\n"
      "sw $a0, 256($a1)\n"
      "sw $a0, 1000($a1)\n"
      "sw $a0, 0x7FFC($a1)\n"
      "daddiu $at, $a1, 0x7FF8\n"
      "sw $a0, 8($at)\n"
      "daddiu $at, $a1, 0x7FF8\n"
      "sw $a0, 12($at)\n"
      "daui $at, $a1, 1\n"
      "sw $a0, 0($at)\n"
      "daui $at, $a1, 4660 # 0x1234\n"
      "sw $a0, 22136($at) # 0x5678\n"
      "sw $a0, -256($a1)\n"
      "sw $a0, -32768($a1)\n"
      "daui $at, $a1, 43982 # 0xABCE\n"
      "sw $a0, -4352($at) # 0xEF00\n"

      "sd $a0, 0($a0)\n"
      "sd $a0, 0($a1)\n"
      "sw $a0, 4($a1)\n"
      "dsrl32 $t3, $a0, 0\n"
      "sw $t3, 8($a1)\n"
      "sd $a0, 256($a1)\n"
      "sd $a0, 1000($a1)\n"
      "daddiu $at, $a1, 0x7FF8\n"
      "sw $a0, 4($at)\n"
      "dsrl32 $t3, $a0, 0\n"
      "sw $t3, 8($at)\n"
      "daddiu $at, $a1, 32760 # 0x7FF8\n"
      "sd $a0, 8($at)\n"
      "daddiu $at, $a1, 32760 # 0x7FF8\n"
      "sw $a0, 12($at)\n"
      "dsrl32 $t3, $a0, 0\n"
      "sw $t3, 16($at)\n"
      "daui $at, $a1, 1\n"
      "sd $a0, 0($at)\n"
      "daui $at, $a1, 4660 # 0x1234\n"
      "sd $a0, 22136($at) # 0x5678\n"
      "sd $a0, -256($a1)\n"
      "sd $a0, -32768($a1)\n"
      "daui $at, $a1, 0xABCE\n"
      "sd $a0, -0x1100($at)\n"
      "daui $at, $a1, 0x8000\n"
      "dahi $at, $at, 1\n"
      "sd $a0, -8($at)\n"
      "daui $at, $a1, 0x8000\n"
      "dahi $at, $at, 1\n"
      "sw $a0, -4($at) # 0xFFFC\n"
      "dsrl32 $t3, $a0, 0\n"
      "sw $t3, 0($at) # 0x0\n"
      "daui $at, $a1, 0x8000\n"
      "sd $a0, 0($at) # 0x0\n"
      "daui $at, $a1, 0x8000\n"
      "sw $a0, 4($at) # 0x4\n"
      "dsrl32 $t3, $a0, 0\n"
      "sw $t3, 8($at) # 0x8\n";
  DriverStr(expected, "StoreToOffset");
}

TEST_F(AssemblerRISCV64Test, StoreFpuToOffset) {
  __ StoreFpuToOffset(riscv64::kStoreWord, riscv64::F0, riscv64::A0, 0);
  __ StoreFpuToOffset(riscv64::kStoreWord, riscv64::F0, riscv64::A0, 4);
  __ StoreFpuToOffset(riscv64::kStoreWord, riscv64::F0, riscv64::A0, 256);
  __ StoreFpuToOffset(riscv64::kStoreWord, riscv64::F0, riscv64::A0, 0x7FFC);
  __ StoreFpuToOffset(riscv64::kStoreWord, riscv64::F0, riscv64::A0, 0x8000);
  __ StoreFpuToOffset(riscv64::kStoreWord, riscv64::F0, riscv64::A0, 0x8004);
  __ StoreFpuToOffset(riscv64::kStoreWord, riscv64::F0, riscv64::A0, 0x10000);
  __ StoreFpuToOffset(riscv64::kStoreWord, riscv64::F0, riscv64::A0, 0x12345678);
  __ StoreFpuToOffset(riscv64::kStoreWord, riscv64::F0, riscv64::A0, -256);
  __ StoreFpuToOffset(riscv64::kStoreWord, riscv64::F0, riscv64::A0, -32768);
  __ StoreFpuToOffset(riscv64::kStoreWord, riscv64::F0, riscv64::A0, 0xABCDEF00);

  __ StoreFpuToOffset(riscv64::kStoreDoubleword, riscv64::F0, riscv64::A0, 0);
  __ StoreFpuToOffset(riscv64::kStoreDoubleword, riscv64::F0, riscv64::A0, 4);
  __ StoreFpuToOffset(riscv64::kStoreDoubleword, riscv64::F0, riscv64::A0, 256);
  __ StoreFpuToOffset(riscv64::kStoreDoubleword, riscv64::F0, riscv64::A0, 0x7FFC);
  __ StoreFpuToOffset(riscv64::kStoreDoubleword, riscv64::F0, riscv64::A0, 0x8000);
  __ StoreFpuToOffset(riscv64::kStoreDoubleword, riscv64::F0, riscv64::A0, 0x8004);
  __ StoreFpuToOffset(riscv64::kStoreDoubleword, riscv64::F0, riscv64::A0, 0x10000);
  __ StoreFpuToOffset(riscv64::kStoreDoubleword, riscv64::F0, riscv64::A0, 0x12345678);
  __ StoreFpuToOffset(riscv64::kStoreDoubleword, riscv64::F0, riscv64::A0, -256);
  __ StoreFpuToOffset(riscv64::kStoreDoubleword, riscv64::F0, riscv64::A0, -32768);
  __ StoreFpuToOffset(riscv64::kStoreDoubleword, riscv64::F0, riscv64::A0, 0xABCDEF00);

  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 0);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 1);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 2);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 4);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 8);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 511);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 512);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 513);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 514);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 516);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 1022);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 1024);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 1025);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 1026);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 1028);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 2044);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 2048);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 2049);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 2050);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 2052);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 4088);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 4096);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 4097);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 4098);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 4100);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 4104);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 0x7FFC);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 0x8000);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 0x10000);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 0x12345678);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 0x12350078);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, -256);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, -511);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, -513);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, -1022);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, -1026);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, -2044);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, -2052);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, -4096);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, -4104);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, -32768);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 0xABCDEF00);
  __ StoreFpuToOffset(riscv64::kStoreQuadword, riscv64::F0, riscv64::A0, 0x7FFFABCD);

  const char* expected =
      "swc1 $f0, 0($a0)\n"
      "swc1 $f0, 4($a0)\n"
      "swc1 $f0, 256($a0)\n"
      "swc1 $f0, 0x7FFC($a0)\n"
      "daddiu $at, $a0, 32760 # 0x7FF8\n"
      "swc1 $f0, 8($at)\n"
      "daddiu $at, $a0, 32760 # 0x7FF8\n"
      "swc1 $f0, 12($at)\n"
      "daui $at, $a0, 1\n"
      "swc1 $f0, 0($at)\n"
      "daui $at, $a0, 4660 # 0x1234\n"
      "swc1 $f0, 22136($at) # 0x5678\n"
      "swc1 $f0, -256($a0)\n"
      "swc1 $f0, -32768($a0)\n"
      "daui $at, $a0, 0xABCE\n"
      "swc1 $f0, -0x1100($at)\n"

      "sdc1 $f0, 0($a0)\n"
      "mfhc1 $t3, $f0\n"
      "swc1 $f0, 4($a0)\n"
      "sw $t3, 8($a0)\n"
      "sdc1 $f0, 256($a0)\n"
      "daddiu $at, $a0, 32760 # 0x7FF8\n"
      "mfhc1 $t3, $f0\n"
      "swc1 $f0, 4($at)\n"
      "sw $t3, 8($at)\n"
      "daddiu $at, $a0, 32760 # 0x7FF8\n"
      "sdc1 $f0, 8($at)\n"
      "daddiu $at, $a0, 32760 # 0x7FF8\n"
      "mfhc1 $t3, $f0\n"
      "swc1 $f0, 12($at)\n"
      "sw $t3, 16($at)\n"
      "daui $at, $a0, 1\n"
      "sdc1 $f0, 0($at)\n"
      "daui $at, $a0, 4660 # 0x1234\n"
      "sdc1 $f0, 22136($at) # 0x5678\n"
      "sdc1 $f0, -256($a0)\n"
      "sdc1 $f0, -32768($a0)\n"
      "daui $at, $a0, 0xABCE\n"
      "sdc1 $f0, -0x1100($at)\n"

      "st.d $w0, 0($a0)\n"
      "st.b $w0, 1($a0)\n"
      "st.h $w0, 2($a0)\n"
      "st.w $w0, 4($a0)\n"
      "st.d $w0, 8($a0)\n"
      "st.b $w0, 511($a0)\n"
      "st.d $w0, 512($a0)\n"
      "daddiu $at, $a0, 513\n"
      "st.b $w0, 0($at)\n"
      "st.h $w0, 514($a0)\n"
      "st.w $w0, 516($a0)\n"
      "st.h $w0, 1022($a0)\n"
      "st.d $w0, 1024($a0)\n"
      "daddiu $at, $a0, 1025\n"
      "st.b $w0, 0($at)\n"
      "daddiu $at, $a0, 1026\n"
      "st.h $w0, 0($at)\n"
      "st.w $w0, 1028($a0)\n"
      "st.w $w0, 2044($a0)\n"
      "st.d $w0, 2048($a0)\n"
      "daddiu $at, $a0, 2049\n"
      "st.b $w0, 0($at)\n"
      "daddiu $at, $a0, 2050\n"
      "st.h $w0, 0($at)\n"
      "daddiu $at, $a0, 2052\n"
      "st.w $w0, 0($at)\n"
      "st.d $w0, 4088($a0)\n"
      "daddiu $at, $a0, 4096\n"
      "st.d $w0, 0($at)\n"
      "daddiu $at, $a0, 4097\n"
      "st.b $w0, 0($at)\n"
      "daddiu $at, $a0, 4098\n"
      "st.h $w0, 0($at)\n"
      "daddiu $at, $a0, 4100\n"
      "st.w $w0, 0($at)\n"
      "daddiu $at, $a0, 4104\n"
      "st.d $w0, 0($at)\n"
      "daddiu $at, $a0, 0x7FFC\n"
      "st.w $w0, 0($at)\n"
      "daddiu $at, $a0, 0x7FF8\n"
      "st.d $w0, 8($at)\n"
      "daui $at, $a0, 0x1\n"
      "st.d $w0, 0($at)\n"
      "daui $at, $a0, 0x1234\n"
      "daddiu $at, $at, 0x6000\n"
      "st.d $w0, -2440($at) # 0xF678\n"
      "daui $at, $a0, 0x1235\n"
      "st.d $w0, 0x78($at)\n"
      "st.d $w0, -256($a0)\n"
      "st.b $w0, -511($a0)\n"
      "daddiu $at, $a0, -513\n"
      "st.b $w0, 0($at)\n"
      "st.h $w0, -1022($a0)\n"
      "daddiu $at, $a0, -1026\n"
      "st.h $w0, 0($at)\n"
      "st.w $w0, -2044($a0)\n"
      "daddiu $at, $a0, -2052\n"
      "st.w $w0, 0($at)\n"
      "st.d $w0, -4096($a0)\n"
      "daddiu $at, $a0, -4104\n"
      "st.d $w0, 0($at)\n"
      "daddiu $at, $a0, -32768\n"
      "st.d $w0, 0($at)\n"
      "daui $at, $a0, 0xABCE\n"
      "daddiu $at, $at, -8192 # 0xE000\n"
      "st.d $w0, 0xF00($at)\n"
      "daui $at, $a0, 0x8000\n"
      "dahi $at, $at, 1\n"
      "daddiu $at, $at, -21504 # 0xAC00\n"
      "st.b $w0, -51($at) # 0xFFCD\n";
  DriverStr(expected, "StoreFpuToOffset");
}

TEST_F(AssemblerRISCV64Test, StoreConstToOffset) {
  __ StoreConstToOffset(riscv64::kStoreByte, 0xFF, riscv64::A1, +0, riscv64::T8);
  __ StoreConstToOffset(riscv64::kStoreHalfword, 0xFFFF, riscv64::A1, +0, riscv64::T8);
  __ StoreConstToOffset(riscv64::kStoreWord, 0x12345678, riscv64::A1, +0, riscv64::T8);
  __ StoreConstToOffset(riscv64::kStoreDoubleword, 0x123456789ABCDEF0, riscv64::A1, +0, riscv64::T8);

  __ StoreConstToOffset(riscv64::kStoreByte, 0, riscv64::A1, +0, riscv64::T8);
  __ StoreConstToOffset(riscv64::kStoreHalfword, 0, riscv64::A1, +0, riscv64::T8);
  __ StoreConstToOffset(riscv64::kStoreWord, 0, riscv64::A1, +0, riscv64::T8);
  __ StoreConstToOffset(riscv64::kStoreDoubleword, 0, riscv64::A1, +0, riscv64::T8);

  __ StoreConstToOffset(riscv64::kStoreDoubleword, 0x1234567812345678, riscv64::A1, +0, riscv64::T8);
  __ StoreConstToOffset(riscv64::kStoreDoubleword, 0x1234567800000000, riscv64::A1, +0, riscv64::T8);
  __ StoreConstToOffset(riscv64::kStoreDoubleword, 0x0000000012345678, riscv64::A1, +0, riscv64::T8);

  __ StoreConstToOffset(riscv64::kStoreWord, 0, riscv64::T8, +0, riscv64::T8);
  __ StoreConstToOffset(riscv64::kStoreWord, 0x12345678, riscv64::T8, +0, riscv64::T8);

  __ StoreConstToOffset(riscv64::kStoreWord, 0, riscv64::A1, -0xFFF0, riscv64::T8);
  __ StoreConstToOffset(riscv64::kStoreWord, 0x12345678, riscv64::A1, +0xFFF0, riscv64::T8);

  __ StoreConstToOffset(riscv64::kStoreWord, 0, riscv64::T8, -0xFFF0, riscv64::T8);
  __ StoreConstToOffset(riscv64::kStoreWord, 0x12345678, riscv64::T8, +0xFFF0, riscv64::T8);

  const char* expected =
      "ori $t8, $zero, 0xFF\n"
      "sb $t8, 0($a1)\n"
      "ori $t8, $zero, 0xFFFF\n"
      "sh $t8, 0($a1)\n"
      "lui $t8, 0x1234\n"
      "ori $t8, $t8,0x5678\n"
      "sw $t8, 0($a1)\n"
      "lui $t8, 0x9abc\n"
      "ori $t8, $t8,0xdef0\n"
      "dahi $t8, $t8, 0x5679\n"
      "dati $t8, $t8, 0x1234\n"
      "sd $t8, 0($a1)\n"
      "sb $zero, 0($a1)\n"
      "sh $zero, 0($a1)\n"
      "sw $zero, 0($a1)\n"
      "sd $zero, 0($a1)\n"
      "lui $t8, 0x1234\n"
      "ori $t8, $t8,0x5678\n"
      "dins $t8, $t8, 0x20, 0x20\n"
      "sd $t8, 0($a1)\n"
      "lui $t8, 0x246\n"
      "ori $t8, $t8, 0x8acf\n"
      "dsll32 $t8, $t8, 0x3\n"
      "sd $t8, 0($a1)\n"
      "lui $t8, 0x1234\n"
      "ori $t8, $t8, 0x5678\n"
      "sd $t8, 0($a1)\n"
      "sw $zero, 0($t8)\n"
      "lui $at,0x1234\n"
      "ori $at, $at, 0x5678\n"
      "sw  $at, 0($t8)\n"
      "daddiu $at, $a1, -32760 # 0x8008\n"
      "sw $zero, -32760($at) # 0x8008\n"
      "daddiu $at, $a1, 32760 # 0x7FF8\n"
      "lui $t8, 4660 # 0x1234\n"
      "ori $t8, $t8, 22136 # 0x5678\n"
      "sw $t8, 32760($at) # 0x7FF8\n"
      "daddiu $at, $t8, -32760 # 0x8008\n"
      "sw $zero, -32760($at) # 0x8008\n"
      "daddiu $at, $t8, 32760 # 0x7FF8\n"
      "lui $t8, 4660 # 0x1234\n"
      "ori $t8, $t8, 22136 # 0x5678\n"
      "sw $t8, 32760($at) # 0x7FF8\n";
  DriverStr(expected, "StoreConstToOffset");
}
//////////////////////////////
// Loading/adding Constants //
//////////////////////////////

TEST_F(AssemblerRISCV64Test, LoadConst32) {
  // IsUint<16>(value)
  __ LoadConst32(riscv64::V0, 0);
  __ LoadConst32(riscv64::V0, 65535);
  // IsInt<16>(value)
  __ LoadConst32(riscv64::V0, -1);
  __ LoadConst32(riscv64::V0, -32768);
  // Everything else
  __ LoadConst32(riscv64::V0, 65536);
  __ LoadConst32(riscv64::V0, 65537);
  __ LoadConst32(riscv64::V0, 2147483647);
  __ LoadConst32(riscv64::V0, -32769);
  __ LoadConst32(riscv64::V0, -65536);
  __ LoadConst32(riscv64::V0, -65537);
  __ LoadConst32(riscv64::V0, -2147483647);
  __ LoadConst32(riscv64::V0, -2147483648);

  const char* expected =
      // IsUint<16>(value)
      "ori $v0, $zero, 0\n"         // __ LoadConst32(riscv64::V0, 0);
      "ori $v0, $zero, 65535\n"     // __ LoadConst32(riscv64::V0, 65535);
      // IsInt<16>(value)
      "addiu $v0, $zero, -1\n"      // __ LoadConst32(riscv64::V0, -1);
      "addiu $v0, $zero, -32768\n"  // __ LoadConst32(riscv64::V0, -32768);
      // Everything else
      "lui $v0, 1\n"                // __ LoadConst32(riscv64::V0, 65536);
      "lui $v0, 1\n"                // __ LoadConst32(riscv64::V0, 65537);
      "ori $v0, 1\n"                //                 "
      "lui $v0, 32767\n"            // __ LoadConst32(riscv64::V0, 2147483647);
      "ori $v0, 65535\n"            //                 "
      "lui $v0, 65535\n"            // __ LoadConst32(riscv64::V0, -32769);
      "ori $v0, 32767\n"            //                 "
      "lui $v0, 65535\n"            // __ LoadConst32(riscv64::V0, -65536);
      "lui $v0, 65534\n"            // __ LoadConst32(riscv64::V0, -65537);
      "ori $v0, 65535\n"            //                 "
      "lui $v0, 32768\n"            // __ LoadConst32(riscv64::V0, -2147483647);
      "ori $v0, 1\n"                //                 "
      "lui $v0, 32768\n";           // __ LoadConst32(riscv64::V0, -2147483648);
  DriverStr(expected, "LoadConst32");
}

TEST_F(AssemblerRISCV64Test, Addiu32) {
  __ Addiu32(riscv64::A1, riscv64::A2, -0x8000);
  __ Addiu32(riscv64::A1, riscv64::A2, +0);
  __ Addiu32(riscv64::A1, riscv64::A2, +0x7FFF);
  __ Addiu32(riscv64::A1, riscv64::A2, -0x8001);
  __ Addiu32(riscv64::A1, riscv64::A2, +0x8000);
  __ Addiu32(riscv64::A1, riscv64::A2, -0x10000);
  __ Addiu32(riscv64::A1, riscv64::A2, +0x10000);
  __ Addiu32(riscv64::A1, riscv64::A2, +0x12345678);

  const char* expected =
      "addiu $a1, $a2, -0x8000\n"
      "addiu $a1, $a2, 0\n"
      "addiu $a1, $a2, 0x7FFF\n"
      "aui $a1, $a2, 0xFFFF\n"
      "addiu $a1, $a1, 0x7FFF\n"
      "aui $a1, $a2, 1\n"
      "addiu $a1, $a1, -0x8000\n"
      "aui $a1, $a2, 0xFFFF\n"
      "aui $a1, $a2, 1\n"
      "aui $a1, $a2, 0x1234\n"
      "addiu $a1, $a1, 0x5678\n";
  DriverStr(expected, "Addiu32");
}

static uint64_t SignExtend16To64(uint16_t n) {
  return static_cast<int16_t>(n);
}

// The art::riscv64::Riscv64Assembler::LoadConst64() method uses a template
// to minimize the number of instructions needed to load a 64-bit constant
// value into a register. The template calls various methods which emit
// MIPS machine instructions. This struct (class) uses the same template
// but overrides the definitions of the methods which emit MIPS instructions
// to use methods which simulate the operation of the corresponding MIPS
// instructions. After invoking LoadConst64() the target register should
// contain the same 64-bit value as was input to LoadConst64(). If the
// simulated register doesn't contain the correct value then there is probably
// an error in the template function.
struct LoadConst64Tester {
  LoadConst64Tester() {
    // Initialize all of the registers for simulation to zero.
    for (int r = 0; r < 32; r++) {
      regs_[r] = 0;
    }
    // Clear all of the path flags.
    loadconst64_paths_ = art::riscv64::kLoadConst64PathZero;
  }
  void Addiu(riscv64::GpuRegister rd, riscv64::GpuRegister rs, uint16_t c) {
    regs_[rd] = static_cast<int32_t>(regs_[rs] + SignExtend16To64(c));
  }
  void Daddiu(riscv64::GpuRegister rd, riscv64::GpuRegister rs, uint16_t c) {
    regs_[rd] = regs_[rs] + SignExtend16To64(c);
  }
  void Dahi(riscv64::GpuRegister rd, uint16_t c) {
    regs_[rd] += SignExtend16To64(c) << 32;
  }
  void Dati(riscv64::GpuRegister rd, uint16_t c) {
    regs_[rd] += SignExtend16To64(c) << 48;
  }
  void Dinsu(riscv64::GpuRegister rt, riscv64::GpuRegister rs, int pos, int size) {
    CHECK(IsUint<5>(pos - 32)) << pos;
    CHECK(IsUint<5>(size - 1)) << size;
    CHECK(IsUint<5>(pos + size - 33)) << pos << " + " << size;
    uint64_t src_mask = (UINT64_C(1) << size) - 1;
    uint64_t dsk_mask = ~(src_mask << pos);

    regs_[rt] = (regs_[rt] & dsk_mask) | ((regs_[rs] & src_mask) << pos);
  }
  void Dsll(riscv64::GpuRegister rd, riscv64::GpuRegister rt, int shamt) {
    regs_[rd] = regs_[rt] << (shamt & 0x1f);
  }
  void Dsll32(riscv64::GpuRegister rd, riscv64::GpuRegister rt, int shamt) {
    regs_[rd] = regs_[rt] << (32 + (shamt & 0x1f));
  }
  void Dsrl(riscv64::GpuRegister rd, riscv64::GpuRegister rt, int shamt) {
    regs_[rd] = regs_[rt] >> (shamt & 0x1f);
  }
  void Dsrl32(riscv64::GpuRegister rd, riscv64::GpuRegister rt, int shamt) {
    regs_[rd] = regs_[rt] >> (32 + (shamt & 0x1f));
  }
  void Lui(riscv64::GpuRegister rd, uint16_t c) {
    regs_[rd] = SignExtend16To64(c) << 16;
  }
  void Ori(riscv64::GpuRegister rd, riscv64::GpuRegister rs, uint16_t c) {
    regs_[rd] = regs_[rs] | c;
  }
  void LoadConst32(riscv64::GpuRegister rd, int32_t c) {
    CHECK_NE(rd, 0);
    riscv64::TemplateLoadConst32<LoadConst64Tester>(this, rd, c);
    CHECK_EQ(regs_[rd], static_cast<uint64_t>(c));
  }
  void LoadConst64(riscv64::GpuRegister rd, int64_t c) {
    CHECK_NE(rd, 0);
    riscv64::TemplateLoadConst64<LoadConst64Tester>(this, rd, c);
    CHECK_EQ(regs_[rd], static_cast<uint64_t>(c));
  }
  uint64_t regs_[32];

  // Getter function for loadconst64_paths_.
  int GetPathsCovered() {
    return loadconst64_paths_;
  }

  void RecordLoadConst64Path(int value) {
    loadconst64_paths_ |= value;
  }

 private:
  // This variable holds a bitmask to tell us which paths were taken
  // through the template function which loads 64-bit values.
  int loadconst64_paths_;
};

TEST_F(AssemblerRISCV64Test, LoadConst64) {
  const uint16_t imms[] = {
      0, 1, 2, 3, 4, 0x33, 0x66, 0x55, 0x99, 0xaa, 0xcc, 0xff, 0x5500, 0x5555,
      0x7ffc, 0x7ffd, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0x8002, 0x8003, 0x8004,
      0xaaaa, 0xfffc, 0xfffd, 0xfffe, 0xffff
  };
  unsigned d0, d1, d2, d3;
  LoadConst64Tester tester;

  union {
    int64_t v64;
    uint16_t v16[4];
  } u;

  for (d3 = 0; d3 < sizeof imms / sizeof imms[0]; d3++) {
    u.v16[3] = imms[d3];

    for (d2 = 0; d2 < sizeof imms / sizeof imms[0]; d2++) {
      u.v16[2] = imms[d2];

      for (d1 = 0; d1 < sizeof imms / sizeof imms[0]; d1++) {
        u.v16[1] = imms[d1];

        for (d0 = 0; d0 < sizeof imms / sizeof imms[0]; d0++) {
          u.v16[0] = imms[d0];

          tester.LoadConst64(riscv64::V0, u.v64);
        }
      }
    }
  }

  // Verify that we tested all paths through the "load 64-bit value"
  // function template.
  EXPECT_EQ(tester.GetPathsCovered(), art::riscv64::kLoadConst64PathAllPaths);
}

TEST_F(AssemblerRISCV64Test, LoadFarthestNearLabelAddress) {
  riscv64::Riscv64Label label;
  __ LoadLabelAddress(riscv64::V0, &label);
  constexpr uint32_t kAdduCount = 0x3FFDE;
  for (uint32_t i = 0; i != kAdduCount; ++i) {
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
  }
  __ Bind(&label);

  std::string expected =
      "lapc $v0, 1f\n" +
      RepeatInsn(kAdduCount, "addu $zero, $zero, $zero\n") +
      "1:\n";
  DriverStr(expected, "LoadFarthestNearLabelAddress");
  EXPECT_EQ(__ GetLabelLocation(&label), (1 + kAdduCount) * 4);
}

TEST_F(AssemblerRISCV64Test, LoadNearestFarLabelAddress) {
  riscv64::Riscv64Label label;
  __ LoadLabelAddress(riscv64::V0, &label);
  constexpr uint32_t kAdduCount = 0x3FFDF;
  for (uint32_t i = 0; i != kAdduCount; ++i) {
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
  }
  __ Bind(&label);

  std::string expected =
      "1:\n"
      "auipc $at, %hi(2f - 1b)\n"
      "daddiu $v0, $at, %lo(2f - 1b)\n" +
      RepeatInsn(kAdduCount, "addu $zero, $zero, $zero\n") +
      "2:\n";
  DriverStr(expected, "LoadNearestFarLabelAddress");
  EXPECT_EQ(__ GetLabelLocation(&label), (2 + kAdduCount) * 4);
}

TEST_F(AssemblerRISCV64Test, LoadFarthestNearLiteral) {
  riscv64::Literal* literal = __ NewLiteral<uint32_t>(0x12345678);
  __ LoadLiteral(riscv64::V0, riscv64::kLoadWord, literal);
  constexpr uint32_t kAdduCount = 0x3FFDE;
  for (uint32_t i = 0; i != kAdduCount; ++i) {
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
  }

  std::string expected =
      "lwpc $v0, 1f\n" +
      RepeatInsn(kAdduCount, "addu $zero, $zero, $zero\n") +
      "1:\n"
      ".word 0x12345678\n";
  DriverStr(expected, "LoadFarthestNearLiteral");
  EXPECT_EQ(__ GetLabelLocation(literal->GetLabel()), (1 + kAdduCount) * 4);
}

TEST_F(AssemblerRISCV64Test, LoadNearestFarLiteral) {
  riscv64::Literal* literal = __ NewLiteral<uint32_t>(0x12345678);
  __ LoadLiteral(riscv64::V0, riscv64::kLoadWord, literal);
  constexpr uint32_t kAdduCount = 0x3FFDF;
  for (uint32_t i = 0; i != kAdduCount; ++i) {
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
  }

  std::string expected =
      "1:\n"
      "auipc $at, %hi(2f - 1b)\n"
      "lw $v0, %lo(2f - 1b)($at)\n" +
      RepeatInsn(kAdduCount, "addu $zero, $zero, $zero\n") +
      "2:\n"
      ".word 0x12345678\n";
  DriverStr(expected, "LoadNearestFarLiteral");
  EXPECT_EQ(__ GetLabelLocation(literal->GetLabel()), (2 + kAdduCount) * 4);
}

TEST_F(AssemblerRISCV64Test, LoadFarthestNearLiteralUnsigned) {
  riscv64::Literal* literal = __ NewLiteral<uint32_t>(0x12345678);
  __ LoadLiteral(riscv64::V0, riscv64::kLoadUnsignedWord, literal);
  constexpr uint32_t kAdduCount = 0x3FFDE;
  for (uint32_t i = 0; i != kAdduCount; ++i) {
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
  }

  std::string expected =
      "lwupc $v0, 1f\n" +
      RepeatInsn(kAdduCount, "addu $zero, $zero, $zero\n") +
      "1:\n"
      ".word 0x12345678\n";
  DriverStr(expected, "LoadFarthestNearLiteralUnsigned");
  EXPECT_EQ(__ GetLabelLocation(literal->GetLabel()), (1 + kAdduCount) * 4);
}

TEST_F(AssemblerRISCV64Test, LoadNearestFarLiteralUnsigned) {
  riscv64::Literal* literal = __ NewLiteral<uint32_t>(0x12345678);
  __ LoadLiteral(riscv64::V0, riscv64::kLoadUnsignedWord, literal);
  constexpr uint32_t kAdduCount = 0x3FFDF;
  for (uint32_t i = 0; i != kAdduCount; ++i) {
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
  }

  std::string expected =
      "1:\n"
      "auipc $at, %hi(2f - 1b)\n"
      "lwu $v0, %lo(2f - 1b)($at)\n" +
      RepeatInsn(kAdduCount, "addu $zero, $zero, $zero\n") +
      "2:\n"
      ".word 0x12345678\n";
  DriverStr(expected, "LoadNearestFarLiteralUnsigned");
  EXPECT_EQ(__ GetLabelLocation(literal->GetLabel()), (2 + kAdduCount) * 4);
}

TEST_F(AssemblerRISCV64Test, LoadFarthestNearLiteralLong) {
  riscv64::Literal* literal = __ NewLiteral<uint64_t>(UINT64_C(0x0123456789ABCDEF));
  __ LoadLiteral(riscv64::V0, riscv64::kLoadDoubleword, literal);
  constexpr uint32_t kAdduCount = 0x3FFDD;
  for (uint32_t i = 0; i != kAdduCount; ++i) {
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
  }

  std::string expected =
      "ldpc $v0, 1f\n" +
      RepeatInsn(kAdduCount, "addu $zero, $zero, $zero\n") +
      "1:\n"
      ".dword 0x0123456789ABCDEF\n";
  DriverStr(expected, "LoadFarthestNearLiteralLong");
  EXPECT_EQ(__ GetLabelLocation(literal->GetLabel()), (1 + kAdduCount) * 4);
}

TEST_F(AssemblerRISCV64Test, LoadNearestFarLiteralLong) {
  riscv64::Literal* literal = __ NewLiteral<uint64_t>(UINT64_C(0x0123456789ABCDEF));
  __ LoadLiteral(riscv64::V0, riscv64::kLoadDoubleword, literal);
  constexpr uint32_t kAdduCount = 0x3FFDE;
  for (uint32_t i = 0; i != kAdduCount; ++i) {
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
  }

  std::string expected =
      "1:\n"
      "auipc $at, %hi(2f - 1b)\n"
      "ld $v0, %lo(2f - 1b)($at)\n" +
      RepeatInsn(kAdduCount, "addu $zero, $zero, $zero\n") +
      "2:\n"
      ".dword 0x0123456789ABCDEF\n";
  DriverStr(expected, "LoadNearestFarLiteralLong");
  EXPECT_EQ(__ GetLabelLocation(literal->GetLabel()), (2 + kAdduCount) * 4);
}

TEST_F(AssemblerRISCV64Test, LongLiteralAlignmentNop) {
  riscv64::Literal* literal1 = __ NewLiteral<uint64_t>(UINT64_C(0x0123456789ABCDEF));
  riscv64::Literal* literal2 = __ NewLiteral<uint64_t>(UINT64_C(0x5555555555555555));
  riscv64::Literal* literal3 = __ NewLiteral<uint64_t>(UINT64_C(0xAAAAAAAAAAAAAAAA));
  __ LoadLiteral(riscv64::A1, riscv64::kLoadDoubleword, literal1);
  __ LoadLiteral(riscv64::A2, riscv64::kLoadDoubleword, literal2);
  __ LoadLiteral(riscv64::A3, riscv64::kLoadDoubleword, literal3);
  __ LoadLabelAddress(riscv64::V0, literal1->GetLabel());
  __ LoadLabelAddress(riscv64::V1, literal2->GetLabel());
  // A nop will be inserted here before the 64-bit literals.

  std::string expected =
      "ldpc $a1, 1f\n"
      // The GNU assembler incorrectly requires the ldpc instruction to be located
      // at an address that's a multiple of 8. TODO: Remove this workaround if/when
      // the assembler is fixed.
      // "ldpc $a2, 2f\n"
      ".word 0xECD80004\n"
      "ldpc $a3, 3f\n"
      "lapc $v0, 1f\n"
      "lapc $v1, 2f\n"
      "nop\n"
      "1:\n"
      ".dword 0x0123456789ABCDEF\n"
      "2:\n"
      ".dword 0x5555555555555555\n"
      "3:\n"
      ".dword 0xAAAAAAAAAAAAAAAA\n";
  DriverStr(expected, "LongLiteralAlignmentNop");
  EXPECT_EQ(__ GetLabelLocation(literal1->GetLabel()), 6 * 4u);
  EXPECT_EQ(__ GetLabelLocation(literal2->GetLabel()), 8 * 4u);
  EXPECT_EQ(__ GetLabelLocation(literal3->GetLabel()), 10 * 4u);
}

TEST_F(AssemblerRISCV64Test, LongLiteralAlignmentNoNop) {
  riscv64::Literal* literal1 = __ NewLiteral<uint64_t>(UINT64_C(0x0123456789ABCDEF));
  riscv64::Literal* literal2 = __ NewLiteral<uint64_t>(UINT64_C(0x5555555555555555));
  __ LoadLiteral(riscv64::A1, riscv64::kLoadDoubleword, literal1);
  __ LoadLiteral(riscv64::A2, riscv64::kLoadDoubleword, literal2);
  __ LoadLabelAddress(riscv64::V0, literal1->GetLabel());
  __ LoadLabelAddress(riscv64::V1, literal2->GetLabel());

  std::string expected =
      "ldpc $a1, 1f\n"
      // The GNU assembler incorrectly requires the ldpc instruction to be located
      // at an address that's a multiple of 8. TODO: Remove this workaround if/when
      // the assembler is fixed.
      // "ldpc $a2, 2f\n"
      ".word 0xECD80003\n"
      "lapc $v0, 1f\n"
      "lapc $v1, 2f\n"
      "1:\n"
      ".dword 0x0123456789ABCDEF\n"
      "2:\n"
      ".dword 0x5555555555555555\n";
  DriverStr(expected, "LongLiteralAlignmentNoNop");
  EXPECT_EQ(__ GetLabelLocation(literal1->GetLabel()), 4 * 4u);
  EXPECT_EQ(__ GetLabelLocation(literal2->GetLabel()), 6 * 4u);
}

TEST_F(AssemblerRISCV64Test, FarLongLiteralAlignmentNop) {
  riscv64::Literal* literal = __ NewLiteral<uint64_t>(UINT64_C(0x0123456789ABCDEF));
  __ LoadLiteral(riscv64::V0, riscv64::kLoadDoubleword, literal);
  __ LoadLabelAddress(riscv64::V1, literal->GetLabel());
  constexpr uint32_t kAdduCount = 0x3FFDF;
  for (uint32_t i = 0; i != kAdduCount; ++i) {
    __ Addu(riscv64::ZERO, riscv64::ZERO, riscv64::ZERO);
  }
  // A nop will be inserted here before the 64-bit literal.

  std::string expected =
      "1:\n"
      "auipc $at, %hi(3f - 1b)\n"
      "ld $v0, %lo(3f - 1b)($at)\n"
      "2:\n"
      "auipc $at, %hi(3f - 2b)\n"
      "daddiu $v1, $at, %lo(3f - 2b)\n" +
      RepeatInsn(kAdduCount, "addu $zero, $zero, $zero\n") +
      "nop\n"
      "3:\n"
      ".dword 0x0123456789ABCDEF\n";
  DriverStr(expected, "FarLongLiteralAlignmentNop");
  EXPECT_EQ(__ GetLabelLocation(literal->GetLabel()), (5 + kAdduCount) * 4);
}

// MSA instructions.

TEST_F(AssemblerRISCV64Test, AndV) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::AndV, "and.v ${reg1}, ${reg2}, ${reg3}"), "and.v");
}

TEST_F(AssemblerRISCV64Test, OrV) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::OrV, "or.v ${reg1}, ${reg2}, ${reg3}"), "or.v");
}

TEST_F(AssemblerRISCV64Test, NorV) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::NorV, "nor.v ${reg1}, ${reg2}, ${reg3}"), "nor.v");
}

TEST_F(AssemblerRISCV64Test, XorV) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::XorV, "xor.v ${reg1}, ${reg2}, ${reg3}"), "xor.v");
}

TEST_F(AssemblerRISCV64Test, AddvB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::AddvB, "addv.b ${reg1}, ${reg2}, ${reg3}"),
            "addv.b");
}

TEST_F(AssemblerRISCV64Test, AddvH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::AddvH, "addv.h ${reg1}, ${reg2}, ${reg3}"),
            "addv.h");
}

TEST_F(AssemblerRISCV64Test, AddvW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::AddvW, "addv.w ${reg1}, ${reg2}, ${reg3}"),
            "addv.w");
}

TEST_F(AssemblerRISCV64Test, AddvD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::AddvD, "addv.d ${reg1}, ${reg2}, ${reg3}"),
            "addv.d");
}

TEST_F(AssemblerRISCV64Test, SubvB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SubvB, "subv.b ${reg1}, ${reg2}, ${reg3}"),
            "subv.b");
}

TEST_F(AssemblerRISCV64Test, SubvH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SubvH, "subv.h ${reg1}, ${reg2}, ${reg3}"),
            "subv.h");
}

TEST_F(AssemblerRISCV64Test, SubvW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SubvW, "subv.w ${reg1}, ${reg2}, ${reg3}"),
            "subv.w");
}

TEST_F(AssemblerRISCV64Test, SubvD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SubvD, "subv.d ${reg1}, ${reg2}, ${reg3}"),
            "subv.d");
}

TEST_F(AssemblerRISCV64Test, Asub_sB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Asub_sB, "asub_s.b ${reg1}, ${reg2}, ${reg3}"),
            "asub_s.b");
}

TEST_F(AssemblerRISCV64Test, Asub_sH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Asub_sH, "asub_s.h ${reg1}, ${reg2}, ${reg3}"),
            "asub_s.h");
}

TEST_F(AssemblerRISCV64Test, Asub_sW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Asub_sW, "asub_s.w ${reg1}, ${reg2}, ${reg3}"),
            "asub_s.w");
}

TEST_F(AssemblerRISCV64Test, Asub_sD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Asub_sD, "asub_s.d ${reg1}, ${reg2}, ${reg3}"),
            "asub_s.d");
}

TEST_F(AssemblerRISCV64Test, Asub_uB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Asub_uB, "asub_u.b ${reg1}, ${reg2}, ${reg3}"),
            "asub_u.b");
}

TEST_F(AssemblerRISCV64Test, Asub_uH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Asub_uH, "asub_u.h ${reg1}, ${reg2}, ${reg3}"),
            "asub_u.h");
}

TEST_F(AssemblerRISCV64Test, Asub_uW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Asub_uW, "asub_u.w ${reg1}, ${reg2}, ${reg3}"),
            "asub_u.w");
}

TEST_F(AssemblerRISCV64Test, Asub_uD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Asub_uD, "asub_u.d ${reg1}, ${reg2}, ${reg3}"),
            "asub_u.d");
}

TEST_F(AssemblerRISCV64Test, MulvB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::MulvB, "mulv.b ${reg1}, ${reg2}, ${reg3}"),
            "mulv.b");
}

TEST_F(AssemblerRISCV64Test, MulvH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::MulvH, "mulv.h ${reg1}, ${reg2}, ${reg3}"),
            "mulv.h");
}

TEST_F(AssemblerRISCV64Test, MulvW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::MulvW, "mulv.w ${reg1}, ${reg2}, ${reg3}"),
            "mulv.w");
}

TEST_F(AssemblerRISCV64Test, MulvD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::MulvD, "mulv.d ${reg1}, ${reg2}, ${reg3}"),
            "mulv.d");
}

TEST_F(AssemblerRISCV64Test, Div_sB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Div_sB, "div_s.b ${reg1}, ${reg2}, ${reg3}"),
            "div_s.b");
}

TEST_F(AssemblerRISCV64Test, Div_sH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Div_sH, "div_s.h ${reg1}, ${reg2}, ${reg3}"),
            "div_s.h");
}

TEST_F(AssemblerRISCV64Test, Div_sW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Div_sW, "div_s.w ${reg1}, ${reg2}, ${reg3}"),
            "div_s.w");
}

TEST_F(AssemblerRISCV64Test, Div_sD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Div_sD, "div_s.d ${reg1}, ${reg2}, ${reg3}"),
            "div_s.d");
}

TEST_F(AssemblerRISCV64Test, Div_uB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Div_uB, "div_u.b ${reg1}, ${reg2}, ${reg3}"),
            "div_u.b");
}

TEST_F(AssemblerRISCV64Test, Div_uH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Div_uH, "div_u.h ${reg1}, ${reg2}, ${reg3}"),
            "div_u.h");
}

TEST_F(AssemblerRISCV64Test, Div_uW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Div_uW, "div_u.w ${reg1}, ${reg2}, ${reg3}"),
            "div_u.w");
}

TEST_F(AssemblerRISCV64Test, Div_uD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Div_uD, "div_u.d ${reg1}, ${reg2}, ${reg3}"),
            "div_u.d");
}

TEST_F(AssemblerRISCV64Test, Mod_sB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Mod_sB, "mod_s.b ${reg1}, ${reg2}, ${reg3}"),
            "mod_s.b");
}

TEST_F(AssemblerRISCV64Test, Mod_sH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Mod_sH, "mod_s.h ${reg1}, ${reg2}, ${reg3}"),
            "mod_s.h");
}

TEST_F(AssemblerRISCV64Test, Mod_sW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Mod_sW, "mod_s.w ${reg1}, ${reg2}, ${reg3}"),
            "mod_s.w");
}

TEST_F(AssemblerRISCV64Test, Mod_sD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Mod_sD, "mod_s.d ${reg1}, ${reg2}, ${reg3}"),
            "mod_s.d");
}

TEST_F(AssemblerRISCV64Test, Mod_uB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Mod_uB, "mod_u.b ${reg1}, ${reg2}, ${reg3}"),
            "mod_u.b");
}

TEST_F(AssemblerRISCV64Test, Mod_uH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Mod_uH, "mod_u.h ${reg1}, ${reg2}, ${reg3}"),
            "mod_u.h");
}

TEST_F(AssemblerRISCV64Test, Mod_uW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Mod_uW, "mod_u.w ${reg1}, ${reg2}, ${reg3}"),
            "mod_u.w");
}

TEST_F(AssemblerRISCV64Test, Mod_uD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Mod_uD, "mod_u.d ${reg1}, ${reg2}, ${reg3}"),
            "mod_u.d");
}

TEST_F(AssemblerRISCV64Test, Add_aB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Add_aB, "add_a.b ${reg1}, ${reg2}, ${reg3}"),
            "add_a.b");
}

TEST_F(AssemblerRISCV64Test, Add_aH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Add_aH, "add_a.h ${reg1}, ${reg2}, ${reg3}"),
            "add_a.h");
}

TEST_F(AssemblerRISCV64Test, Add_aW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Add_aW, "add_a.w ${reg1}, ${reg2}, ${reg3}"),
            "add_a.w");
}

TEST_F(AssemblerRISCV64Test, Add_aD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Add_aD, "add_a.d ${reg1}, ${reg2}, ${reg3}"),
            "add_a.d");
}

TEST_F(AssemblerRISCV64Test, Ave_sB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Ave_sB, "ave_s.b ${reg1}, ${reg2}, ${reg3}"),
            "ave_s.b");
}

TEST_F(AssemblerRISCV64Test, Ave_sH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Ave_sH, "ave_s.h ${reg1}, ${reg2}, ${reg3}"),
            "ave_s.h");
}

TEST_F(AssemblerRISCV64Test, Ave_sW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Ave_sW, "ave_s.w ${reg1}, ${reg2}, ${reg3}"),
            "ave_s.w");
}

TEST_F(AssemblerRISCV64Test, Ave_sD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Ave_sD, "ave_s.d ${reg1}, ${reg2}, ${reg3}"),
            "ave_s.d");
}

TEST_F(AssemblerRISCV64Test, Ave_uB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Ave_uB, "ave_u.b ${reg1}, ${reg2}, ${reg3}"),
            "ave_u.b");
}

TEST_F(AssemblerRISCV64Test, Ave_uH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Ave_uH, "ave_u.h ${reg1}, ${reg2}, ${reg3}"),
            "ave_u.h");
}

TEST_F(AssemblerRISCV64Test, Ave_uW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Ave_uW, "ave_u.w ${reg1}, ${reg2}, ${reg3}"),
            "ave_u.w");
}

TEST_F(AssemblerRISCV64Test, Ave_uD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Ave_uD, "ave_u.d ${reg1}, ${reg2}, ${reg3}"),
            "ave_u.d");
}

TEST_F(AssemblerRISCV64Test, Aver_sB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Aver_sB, "aver_s.b ${reg1}, ${reg2}, ${reg3}"),
            "aver_s.b");
}

TEST_F(AssemblerRISCV64Test, Aver_sH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Aver_sH, "aver_s.h ${reg1}, ${reg2}, ${reg3}"),
            "aver_s.h");
}

TEST_F(AssemblerRISCV64Test, Aver_sW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Aver_sW, "aver_s.w ${reg1}, ${reg2}, ${reg3}"),
            "aver_s.w");
}

TEST_F(AssemblerRISCV64Test, Aver_sD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Aver_sD, "aver_s.d ${reg1}, ${reg2}, ${reg3}"),
            "aver_s.d");
}

TEST_F(AssemblerRISCV64Test, Aver_uB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Aver_uB, "aver_u.b ${reg1}, ${reg2}, ${reg3}"),
            "aver_u.b");
}

TEST_F(AssemblerRISCV64Test, Aver_uH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Aver_uH, "aver_u.h ${reg1}, ${reg2}, ${reg3}"),
            "aver_u.h");
}

TEST_F(AssemblerRISCV64Test, Aver_uW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Aver_uW, "aver_u.w ${reg1}, ${reg2}, ${reg3}"),
            "aver_u.w");
}

TEST_F(AssemblerRISCV64Test, Aver_uD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Aver_uD, "aver_u.d ${reg1}, ${reg2}, ${reg3}"),
            "aver_u.d");
}

TEST_F(AssemblerRISCV64Test, Max_sB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Max_sB, "max_s.b ${reg1}, ${reg2}, ${reg3}"),
            "max_s.b");
}

TEST_F(AssemblerRISCV64Test, Max_sH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Max_sH, "max_s.h ${reg1}, ${reg2}, ${reg3}"),
            "max_s.h");
}

TEST_F(AssemblerRISCV64Test, Max_sW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Max_sW, "max_s.w ${reg1}, ${reg2}, ${reg3}"),
            "max_s.w");
}

TEST_F(AssemblerRISCV64Test, Max_sD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Max_sD, "max_s.d ${reg1}, ${reg2}, ${reg3}"),
            "max_s.d");
}

TEST_F(AssemblerRISCV64Test, Max_uB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Max_uB, "max_u.b ${reg1}, ${reg2}, ${reg3}"),
            "max_u.b");
}

TEST_F(AssemblerRISCV64Test, Max_uH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Max_uH, "max_u.h ${reg1}, ${reg2}, ${reg3}"),
            "max_u.h");
}

TEST_F(AssemblerRISCV64Test, Max_uW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Max_uW, "max_u.w ${reg1}, ${reg2}, ${reg3}"),
            "max_u.w");
}

TEST_F(AssemblerRISCV64Test, Max_uD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Max_uD, "max_u.d ${reg1}, ${reg2}, ${reg3}"),
            "max_u.d");
}

TEST_F(AssemblerRISCV64Test, Min_sB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Min_sB, "min_s.b ${reg1}, ${reg2}, ${reg3}"),
            "min_s.b");
}

TEST_F(AssemblerRISCV64Test, Min_sH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Min_sH, "min_s.h ${reg1}, ${reg2}, ${reg3}"),
            "min_s.h");
}

TEST_F(AssemblerRISCV64Test, Min_sW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Min_sW, "min_s.w ${reg1}, ${reg2}, ${reg3}"),
            "min_s.w");
}

TEST_F(AssemblerRISCV64Test, Min_sD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Min_sD, "min_s.d ${reg1}, ${reg2}, ${reg3}"),
            "min_s.d");
}

TEST_F(AssemblerRISCV64Test, Min_uB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Min_uB, "min_u.b ${reg1}, ${reg2}, ${reg3}"),
            "min_u.b");
}

TEST_F(AssemblerRISCV64Test, Min_uH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Min_uH, "min_u.h ${reg1}, ${reg2}, ${reg3}"),
            "min_u.h");
}

TEST_F(AssemblerRISCV64Test, Min_uW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Min_uW, "min_u.w ${reg1}, ${reg2}, ${reg3}"),
            "min_u.w");
}

TEST_F(AssemblerRISCV64Test, Min_uD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Min_uD, "min_u.d ${reg1}, ${reg2}, ${reg3}"),
            "min_u.d");
}

TEST_F(AssemblerRISCV64Test, FaddW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FaddW, "fadd.w ${reg1}, ${reg2}, ${reg3}"),
            "fadd.w");
}

TEST_F(AssemblerRISCV64Test, FaddD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FaddD, "fadd.d ${reg1}, ${reg2}, ${reg3}"),
            "fadd.d");
}

TEST_F(AssemblerRISCV64Test, FsubW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FsubW, "fsub.w ${reg1}, ${reg2}, ${reg3}"),
            "fsub.w");
}

TEST_F(AssemblerRISCV64Test, FsubD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FsubD, "fsub.d ${reg1}, ${reg2}, ${reg3}"),
            "fsub.d");
}

TEST_F(AssemblerRISCV64Test, FmulW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FmulW, "fmul.w ${reg1}, ${reg2}, ${reg3}"),
            "fmul.w");
}

TEST_F(AssemblerRISCV64Test, FmulD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FmulD, "fmul.d ${reg1}, ${reg2}, ${reg3}"),
            "fmul.d");
}

TEST_F(AssemblerRISCV64Test, FdivW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FdivW, "fdiv.w ${reg1}, ${reg2}, ${reg3}"),
            "fdiv.w");
}

TEST_F(AssemblerRISCV64Test, FdivD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FdivD, "fdiv.d ${reg1}, ${reg2}, ${reg3}"),
            "fdiv.d");
}

TEST_F(AssemblerRISCV64Test, FmaxW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FmaxW, "fmax.w ${reg1}, ${reg2}, ${reg3}"),
            "fmax.w");
}

TEST_F(AssemblerRISCV64Test, FmaxD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FmaxD, "fmax.d ${reg1}, ${reg2}, ${reg3}"),
            "fmax.d");
}

TEST_F(AssemblerRISCV64Test, FminW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FminW, "fmin.w ${reg1}, ${reg2}, ${reg3}"),
            "fmin.w");
}

TEST_F(AssemblerRISCV64Test, FminD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FminD, "fmin.d ${reg1}, ${reg2}, ${reg3}"),
            "fmin.d");
}

TEST_F(AssemblerRISCV64Test, Ffint_sW) {
  DriverStr(RepeatVV(&riscv64::Riscv64Assembler::Ffint_sW, "ffint_s.w ${reg1}, ${reg2}"),
            "ffint_s.w");
}

TEST_F(AssemblerRISCV64Test, Ffint_sD) {
  DriverStr(RepeatVV(&riscv64::Riscv64Assembler::Ffint_sD, "ffint_s.d ${reg1}, ${reg2}"),
            "ffint_s.d");
}

TEST_F(AssemblerRISCV64Test, Ftint_sW) {
  DriverStr(RepeatVV(&riscv64::Riscv64Assembler::Ftint_sW, "ftint_s.w ${reg1}, ${reg2}"),
            "ftint_s.w");
}

TEST_F(AssemblerRISCV64Test, Ftint_sD) {
  DriverStr(RepeatVV(&riscv64::Riscv64Assembler::Ftint_sD, "ftint_s.d ${reg1}, ${reg2}"),
            "ftint_s.d");
}

TEST_F(AssemblerRISCV64Test, SllB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SllB, "sll.b ${reg1}, ${reg2}, ${reg3}"), "sll.b");
}

TEST_F(AssemblerRISCV64Test, SllH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SllH, "sll.h ${reg1}, ${reg2}, ${reg3}"), "sll.h");
}

TEST_F(AssemblerRISCV64Test, SllW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SllW, "sll.w ${reg1}, ${reg2}, ${reg3}"), "sll.w");
}

TEST_F(AssemblerRISCV64Test, SllD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SllD, "sll.d ${reg1}, ${reg2}, ${reg3}"), "sll.d");
}

TEST_F(AssemblerRISCV64Test, SraB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SraB, "sra.b ${reg1}, ${reg2}, ${reg3}"), "sra.b");
}

TEST_F(AssemblerRISCV64Test, SraH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SraH, "sra.h ${reg1}, ${reg2}, ${reg3}"), "sra.h");
}

TEST_F(AssemblerRISCV64Test, SraW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SraW, "sra.w ${reg1}, ${reg2}, ${reg3}"), "sra.w");
}

TEST_F(AssemblerRISCV64Test, SraD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SraD, "sra.d ${reg1}, ${reg2}, ${reg3}"), "sra.d");
}

TEST_F(AssemblerRISCV64Test, SrlB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SrlB, "srl.b ${reg1}, ${reg2}, ${reg3}"), "srl.b");
}

TEST_F(AssemblerRISCV64Test, SrlH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SrlH, "srl.h ${reg1}, ${reg2}, ${reg3}"), "srl.h");
}

TEST_F(AssemblerRISCV64Test, SrlW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SrlW, "srl.w ${reg1}, ${reg2}, ${reg3}"), "srl.w");
}

TEST_F(AssemblerRISCV64Test, SrlD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::SrlD, "srl.d ${reg1}, ${reg2}, ${reg3}"), "srl.d");
}

TEST_F(AssemblerRISCV64Test, SlliB) {
  DriverStr(RepeatVVIb(&riscv64::Riscv64Assembler::SlliB, 3, "slli.b ${reg1}, ${reg2}, {imm}"),
            "slli.b");
}

TEST_F(AssemblerRISCV64Test, SlliH) {
  DriverStr(RepeatVVIb(&riscv64::Riscv64Assembler::SlliH, 4, "slli.h ${reg1}, ${reg2}, {imm}"),
            "slli.h");
}

TEST_F(AssemblerRISCV64Test, SlliW) {
  DriverStr(RepeatVVIb(&riscv64::Riscv64Assembler::SlliW, 5, "slli.w ${reg1}, ${reg2}, {imm}"),
            "slli.w");
}

TEST_F(AssemblerRISCV64Test, SlliD) {
  DriverStr(RepeatVVIb(&riscv64::Riscv64Assembler::SlliD, 6, "slli.d ${reg1}, ${reg2}, {imm}"),
            "slli.d");
}

TEST_F(AssemblerRISCV64Test, MoveV) {
  DriverStr(RepeatVV(&riscv64::Riscv64Assembler::MoveV, "move.v ${reg1}, ${reg2}"), "move.v");
}

TEST_F(AssemblerRISCV64Test, SplatiB) {
  DriverStr(RepeatVVIb(&riscv64::Riscv64Assembler::SplatiB, 4, "splati.b ${reg1}, ${reg2}[{imm}]"),
            "splati.b");
}

TEST_F(AssemblerRISCV64Test, SplatiH) {
  DriverStr(RepeatVVIb(&riscv64::Riscv64Assembler::SplatiH, 3, "splati.h ${reg1}, ${reg2}[{imm}]"),
            "splati.h");
}

TEST_F(AssemblerRISCV64Test, SplatiW) {
  DriverStr(RepeatVVIb(&riscv64::Riscv64Assembler::SplatiW, 2, "splati.w ${reg1}, ${reg2}[{imm}]"),
            "splati.w");
}

TEST_F(AssemblerRISCV64Test, SplatiD) {
  DriverStr(RepeatVVIb(&riscv64::Riscv64Assembler::SplatiD, 1, "splati.d ${reg1}, ${reg2}[{imm}]"),
            "splati.d");
}

TEST_F(AssemblerRISCV64Test, Copy_sB) {
  DriverStr(RepeatRVIb(&riscv64::Riscv64Assembler::Copy_sB, 4, "copy_s.b ${reg1}, ${reg2}[{imm}]"),
            "copy_s.b");
}

TEST_F(AssemblerRISCV64Test, Copy_sH) {
  DriverStr(RepeatRVIb(&riscv64::Riscv64Assembler::Copy_sH, 3, "copy_s.h ${reg1}, ${reg2}[{imm}]"),
            "copy_s.h");
}

TEST_F(AssemblerRISCV64Test, Copy_sW) {
  DriverStr(RepeatRVIb(&riscv64::Riscv64Assembler::Copy_sW, 2, "copy_s.w ${reg1}, ${reg2}[{imm}]"),
            "copy_s.w");
}

TEST_F(AssemblerRISCV64Test, Copy_sD) {
  DriverStr(RepeatRVIb(&riscv64::Riscv64Assembler::Copy_sD, 1, "copy_s.d ${reg1}, ${reg2}[{imm}]"),
            "copy_s.d");
}

TEST_F(AssemblerRISCV64Test, Copy_uB) {
  DriverStr(RepeatRVIb(&riscv64::Riscv64Assembler::Copy_uB, 4, "copy_u.b ${reg1}, ${reg2}[{imm}]"),
            "copy_u.b");
}

TEST_F(AssemblerRISCV64Test, Copy_uH) {
  DriverStr(RepeatRVIb(&riscv64::Riscv64Assembler::Copy_uH, 3, "copy_u.h ${reg1}, ${reg2}[{imm}]"),
            "copy_u.h");
}

TEST_F(AssemblerRISCV64Test, Copy_uW) {
  DriverStr(RepeatRVIb(&riscv64::Riscv64Assembler::Copy_uW, 2, "copy_u.w ${reg1}, ${reg2}[{imm}]"),
            "copy_u.w");
}

TEST_F(AssemblerRISCV64Test, InsertB) {
  DriverStr(RepeatVRIb(&riscv64::Riscv64Assembler::InsertB, 4, "insert.b ${reg1}[{imm}], ${reg2}"),
            "insert.b");
}

TEST_F(AssemblerRISCV64Test, InsertH) {
  DriverStr(RepeatVRIb(&riscv64::Riscv64Assembler::InsertH, 3, "insert.h ${reg1}[{imm}], ${reg2}"),
            "insert.h");
}

TEST_F(AssemblerRISCV64Test, InsertW) {
  DriverStr(RepeatVRIb(&riscv64::Riscv64Assembler::InsertW, 2, "insert.w ${reg1}[{imm}], ${reg2}"),
            "insert.w");
}

TEST_F(AssemblerRISCV64Test, InsertD) {
  DriverStr(RepeatVRIb(&riscv64::Riscv64Assembler::InsertD, 1, "insert.d ${reg1}[{imm}], ${reg2}"),
            "insert.d");
}

TEST_F(AssemblerRISCV64Test, FillB) {
  DriverStr(RepeatVR(&riscv64::Riscv64Assembler::FillB, "fill.b ${reg1}, ${reg2}"), "fill.b");
}

TEST_F(AssemblerRISCV64Test, FillH) {
  DriverStr(RepeatVR(&riscv64::Riscv64Assembler::FillH, "fill.h ${reg1}, ${reg2}"), "fill.h");
}

TEST_F(AssemblerRISCV64Test, FillW) {
  DriverStr(RepeatVR(&riscv64::Riscv64Assembler::FillW, "fill.w ${reg1}, ${reg2}"), "fill.w");
}

TEST_F(AssemblerRISCV64Test, FillD) {
  DriverStr(RepeatVR(&riscv64::Riscv64Assembler::FillD, "fill.d ${reg1}, ${reg2}"), "fill.d");
}

TEST_F(AssemblerRISCV64Test, PcntB) {
  DriverStr(RepeatVV(&riscv64::Riscv64Assembler::PcntB, "pcnt.b ${reg1}, ${reg2}"), "pcnt.b");
}

TEST_F(AssemblerRISCV64Test, PcntH) {
  DriverStr(RepeatVV(&riscv64::Riscv64Assembler::PcntH, "pcnt.h ${reg1}, ${reg2}"), "pcnt.h");
}

TEST_F(AssemblerRISCV64Test, PcntW) {
  DriverStr(RepeatVV(&riscv64::Riscv64Assembler::PcntW, "pcnt.w ${reg1}, ${reg2}"), "pcnt.w");
}

TEST_F(AssemblerRISCV64Test, PcntD) {
  DriverStr(RepeatVV(&riscv64::Riscv64Assembler::PcntD, "pcnt.d ${reg1}, ${reg2}"), "pcnt.d");
}

TEST_F(AssemblerRISCV64Test, LdiB) {
  DriverStr(RepeatVIb(&riscv64::Riscv64Assembler::LdiB, -8, "ldi.b ${reg}, {imm}"), "ldi.b");
}

TEST_F(AssemblerRISCV64Test, LdiH) {
  DriverStr(RepeatVIb(&riscv64::Riscv64Assembler::LdiH, -10, "ldi.h ${reg}, {imm}"), "ldi.h");
}

TEST_F(AssemblerRISCV64Test, LdiW) {
  DriverStr(RepeatVIb(&riscv64::Riscv64Assembler::LdiW, -10, "ldi.w ${reg}, {imm}"), "ldi.w");
}

TEST_F(AssemblerRISCV64Test, LdiD) {
  DriverStr(RepeatVIb(&riscv64::Riscv64Assembler::LdiD, -10, "ldi.d ${reg}, {imm}"), "ldi.d");
}

TEST_F(AssemblerRISCV64Test, LdB) {
  DriverStr(RepeatVRIb(&riscv64::Riscv64Assembler::LdB, -10, "ld.b ${reg1}, {imm}(${reg2})"), "ld.b");
}

TEST_F(AssemblerRISCV64Test, LdH) {
  DriverStr(RepeatVRIb(&riscv64::Riscv64Assembler::LdH, -10, "ld.h ${reg1}, {imm}(${reg2})", 0, 2),
            "ld.h");
}

TEST_F(AssemblerRISCV64Test, LdW) {
  DriverStr(RepeatVRIb(&riscv64::Riscv64Assembler::LdW, -10, "ld.w ${reg1}, {imm}(${reg2})", 0, 4),
            "ld.w");
}

TEST_F(AssemblerRISCV64Test, LdD) {
  DriverStr(RepeatVRIb(&riscv64::Riscv64Assembler::LdD, -10, "ld.d ${reg1}, {imm}(${reg2})", 0, 8),
            "ld.d");
}

TEST_F(AssemblerRISCV64Test, StB) {
  DriverStr(RepeatVRIb(&riscv64::Riscv64Assembler::StB, -10, "st.b ${reg1}, {imm}(${reg2})"), "st.b");
}

TEST_F(AssemblerRISCV64Test, StH) {
  DriverStr(RepeatVRIb(&riscv64::Riscv64Assembler::StH, -10, "st.h ${reg1}, {imm}(${reg2})", 0, 2),
            "st.h");
}

TEST_F(AssemblerRISCV64Test, StW) {
  DriverStr(RepeatVRIb(&riscv64::Riscv64Assembler::StW, -10, "st.w ${reg1}, {imm}(${reg2})", 0, 4),
            "st.w");
}

TEST_F(AssemblerRISCV64Test, StD) {
  DriverStr(RepeatVRIb(&riscv64::Riscv64Assembler::StD, -10, "st.d ${reg1}, {imm}(${reg2})", 0, 8),
            "st.d");
}

TEST_F(AssemblerRISCV64Test, IlvlB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvlB, "ilvl.b ${reg1}, ${reg2}, ${reg3}"),
            "ilvl.b");
}

TEST_F(AssemblerRISCV64Test, IlvlH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvlH, "ilvl.h ${reg1}, ${reg2}, ${reg3}"),
            "ilvl.h");
}

TEST_F(AssemblerRISCV64Test, IlvlW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvlW, "ilvl.w ${reg1}, ${reg2}, ${reg3}"),
            "ilvl.w");
}

TEST_F(AssemblerRISCV64Test, IlvlD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvlD, "ilvl.d ${reg1}, ${reg2}, ${reg3}"),
            "ilvl.d");
}

TEST_F(AssemblerRISCV64Test, IlvrB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvrB, "ilvr.b ${reg1}, ${reg2}, ${reg3}"),
            "ilvr.b");
}

TEST_F(AssemblerRISCV64Test, IlvrH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvrH, "ilvr.h ${reg1}, ${reg2}, ${reg3}"),
            "ilvr.h");
}

TEST_F(AssemblerRISCV64Test, IlvrW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvrW, "ilvr.w ${reg1}, ${reg2}, ${reg3}"),
            "ilvr.w");
}

TEST_F(AssemblerRISCV64Test, IlvrD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvrD, "ilvr.d ${reg1}, ${reg2}, ${reg3}"),
            "ilvr.d");
}

TEST_F(AssemblerRISCV64Test, IlvevB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvevB, "ilvev.b ${reg1}, ${reg2}, ${reg3}"),
            "ilvev.b");
}

TEST_F(AssemblerRISCV64Test, IlvevH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvevH, "ilvev.h ${reg1}, ${reg2}, ${reg3}"),
            "ilvev.h");
}

TEST_F(AssemblerRISCV64Test, IlvevW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvevW, "ilvev.w ${reg1}, ${reg2}, ${reg3}"),
            "ilvev.w");
}

TEST_F(AssemblerRISCV64Test, IlvevD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvevD, "ilvev.d ${reg1}, ${reg2}, ${reg3}"),
            "ilvev.d");
}

TEST_F(AssemblerRISCV64Test, IlvodB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvodB, "ilvod.b ${reg1}, ${reg2}, ${reg3}"),
            "ilvod.b");
}

TEST_F(AssemblerRISCV64Test, IlvodH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvodH, "ilvod.h ${reg1}, ${reg2}, ${reg3}"),
            "ilvod.h");
}

TEST_F(AssemblerRISCV64Test, IlvodW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvodW, "ilvod.w ${reg1}, ${reg2}, ${reg3}"),
            "ilvod.w");
}

TEST_F(AssemblerRISCV64Test, IlvodD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::IlvodD, "ilvod.d ${reg1}, ${reg2}, ${reg3}"),
            "ilvod.d");
}

TEST_F(AssemblerRISCV64Test, MaddvB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::MaddvB, "maddv.b ${reg1}, ${reg2}, ${reg3}"),
            "maddv.b");
}

TEST_F(AssemblerRISCV64Test, MaddvH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::MaddvH, "maddv.h ${reg1}, ${reg2}, ${reg3}"),
            "maddv.h");
}

TEST_F(AssemblerRISCV64Test, MaddvW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::MaddvW, "maddv.w ${reg1}, ${reg2}, ${reg3}"),
            "maddv.w");
}

TEST_F(AssemblerRISCV64Test, MaddvD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::MaddvD, "maddv.d ${reg1}, ${reg2}, ${reg3}"),
            "maddv.d");
}

TEST_F(AssemblerRISCV64Test, Hadd_sH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Hadd_sH, "hadd_s.h ${reg1}, ${reg2}, ${reg3}"),
            "hadd_s.h");
}

TEST_F(AssemblerRISCV64Test, Hadd_sW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Hadd_sW, "hadd_s.w ${reg1}, ${reg2}, ${reg3}"),
            "hadd_s.w");
}

TEST_F(AssemblerRISCV64Test, Hadd_sD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Hadd_sD, "hadd_s.d ${reg1}, ${reg2}, ${reg3}"),
            "hadd_s.d");
}

TEST_F(AssemblerRISCV64Test, Hadd_uH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Hadd_uH, "hadd_u.h ${reg1}, ${reg2}, ${reg3}"),
            "hadd_u.h");
}

TEST_F(AssemblerRISCV64Test, Hadd_uW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Hadd_uW, "hadd_u.w ${reg1}, ${reg2}, ${reg3}"),
            "hadd_u.w");
}

TEST_F(AssemblerRISCV64Test, Hadd_uD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::Hadd_uD, "hadd_u.d ${reg1}, ${reg2}, ${reg3}"),
            "hadd_u.d");
}

TEST_F(AssemblerRISCV64Test, MsubvB) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::MsubvB, "msubv.b ${reg1}, ${reg2}, ${reg3}"),
            "msubv.b");
}

TEST_F(AssemblerRISCV64Test, MsubvH) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::MsubvH, "msubv.h ${reg1}, ${reg2}, ${reg3}"),
            "msubv.h");
}

TEST_F(AssemblerRISCV64Test, MsubvW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::MsubvW, "msubv.w ${reg1}, ${reg2}, ${reg3}"),
            "msubv.w");
}

TEST_F(AssemblerRISCV64Test, MsubvD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::MsubvD, "msubv.d ${reg1}, ${reg2}, ${reg3}"),
            "msubv.d");
}

TEST_F(AssemblerRISCV64Test, FmaddW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FmaddW, "fmadd.w ${reg1}, ${reg2}, ${reg3}"),
            "fmadd.w");
}

TEST_F(AssemblerRISCV64Test, FmaddD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FmaddD, "fmadd.d ${reg1}, ${reg2}, ${reg3}"),
            "fmadd.d");
}

TEST_F(AssemblerRISCV64Test, FmsubW) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FmsubW, "fmsub.w ${reg1}, ${reg2}, ${reg3}"),
            "fmsub.w");
}

TEST_F(AssemblerRISCV64Test, FmsubD) {
  DriverStr(RepeatVVV(&riscv64::Riscv64Assembler::FmsubD, "fmsub.d ${reg1}, ${reg2}, ${reg3}"),
            "fmsub.d");
}
#endif
#if 1
#if  TEST_RV32I_R
TEST_F(AssemblerRISCV64Test, Add) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Add, "add {reg1}, {reg2}, {reg3}"), "Add");
}

TEST_F(AssemblerRISCV64Test, Sub) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Sub, "sub {reg1}, {reg2}, {reg3}"), "Sub");
}

TEST_F(AssemblerRISCV64Test, Sll) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Sll, "sll {reg1}, {reg2}, {reg3}"), "Sll");
}

TEST_F(AssemblerRISCV64Test, Slt) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Slt, "slt {reg1}, {reg2}, {reg3}"), "Slt");
}

TEST_F(AssemblerRISCV64Test, Sltu) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Sltu, "sltu {reg1}, {reg2}, {reg3}"), "Sltu");
}

TEST_F(AssemblerRISCV64Test, Xor) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Xor, "xor {reg1}, {reg2}, {reg3}"), "Xor");
}

TEST_F(AssemblerRISCV64Test, Srl) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Srl, "srl {reg1}, {reg2}, {reg3}"), "Srl");
}

TEST_F(AssemblerRISCV64Test, Sra) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Sra, "sra {reg1}, {reg2}, {reg3}"), "Sra");
}

TEST_F(AssemblerRISCV64Test, Or) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Or, "or {reg1}, {reg2}, {reg3}"), "Or");
}

TEST_F(AssemblerRISCV64Test, And) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::And, "and {reg1}, {reg2}, {reg3}"), "And");
}
#endif

#if TEST_RV32I_I
TEST_F(AssemblerRISCV64Test, Jalr) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Jalr, -11, "jalr {reg1}, {imm}({reg2})"), "Jalr");
}
TEST_F(AssemblerRISCV64Test, Lb) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Lb, -11, "lb {reg1}, {imm}({reg2})"), "Lb");
}

TEST_F(AssemblerRISCV64Test, Lh) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Lh, -11, "lh {reg1}, {imm}({reg2})"), "Lh");
}

TEST_F(AssemblerRISCV64Test, Lw) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Lw, -11, "lw {reg1}, {imm}({reg2})"), "Lw");
}

TEST_F(AssemblerRISCV64Test, Lbu) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Lbu, -11, "lbu {reg1}, {imm}({reg2})"), "Lbu");
}

TEST_F(AssemblerRISCV64Test, Lhu) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Lhu, -11, "lhu {reg1}, {imm}({reg2})"), "Lhu");
}

TEST_F(AssemblerRISCV64Test, Addi) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Addi, -11, "addi {reg1}, {reg2}, {imm}"), "Addi");
}

TEST_F(AssemblerRISCV64Test, Slti) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Slti, -11, "slti {reg1}, {reg2}, {imm}"), "Slti");
}

TEST_F(AssemblerRISCV64Test, Sltiu) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Sltiu, -11, "sltiu {reg1}, {reg2}, {imm}"), "Sltiu");
}

TEST_F(AssemblerRISCV64Test, Xori) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Xori, -11, "xori {reg1}, {reg2}, {imm}"), "Xori");
}

TEST_F(AssemblerRISCV64Test, Ori) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Ori, -11, "ori {reg1}, {reg2}, {imm}"), "Ori");
}

TEST_F(AssemblerRISCV64Test, Andi) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Andi, -11, "andi {reg1}, {reg2}, {imm}"), "Andi");
}


TEST_F(AssemblerRISCV64Test, Fence) {
  __ Fence(0x02, 0x03);
  DriverStr("fence r, rw", "Fence");
}

TEST_F(AssemblerRISCV64Test, FenceI) {
  __ FenceI();
  DriverStr("fence.i", "FenceI");
}

TEST_F(AssemblerRISCV64Test, Ecall) {
  __ Ecall();
  DriverStr("ecall", "Ecall");
}

TEST_F(AssemblerRISCV64Test, Ebreak) {
  __ Ebreak();
  DriverStr("ebreak", "Ebreak");
}

TEST_F(AssemblerRISCV64Test, Csrrw) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Csrrw, 12, "csrrw {reg1}, {imm}, {reg2}"), "Csrrw");
}

TEST_F(AssemblerRISCV64Test, Csrrs) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Csrrs, 12, "csrrs {reg1}, {imm}, {reg2}"), "Csrrs");
}

TEST_F(AssemblerRISCV64Test, Csrrc) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Csrrc, 12, "csrrc {reg1}, {imm}, {reg2}"), "Csrrc");
}

TEST_F(AssemblerRISCV64Test, Csrrwi) {
  __ Csrrwi(riscv64::A0, 0, 1);
  DriverStr("csrrwi a0, ustatus, 1\n"
    , "Csrrwi");
}

TEST_F(AssemblerRISCV64Test, Csrrsi) {
  __ Csrrsi(riscv64::A0, 0, 1);
  DriverStr("csrrsi a0, ustatus, 1\n"
    , "Csrrsi");
}

TEST_F(AssemblerRISCV64Test, Csrrci) {
  __ Csrrci(riscv64::A0, 0, 1);
  DriverStr("csrrci a0, ustatus, 1\n"
    , "Csrrci");
}
#endif

#if TEST_RV32I_S
TEST_F(AssemblerRISCV64Test, Sb) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Sb, -11, "sb {reg1}, {imm}({reg2})"), "Sb");
}

TEST_F(AssemblerRISCV64Test, Sh) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Sh, -11, "sh {reg1}, {imm}({reg2})"), "Sh");
}

TEST_F(AssemblerRISCV64Test, Sw) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Sw, -11, "sw {reg1}, {imm}({reg2})"), "Sw");
}
#endif

#if TEST_RV32I_B
TEST_F(AssemblerRISCV64Test, Beq) {
  BranchCondTwoRegsHelper1(&riscv64::Riscv64Assembler::Beq, "Beq", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, Bne) {
  BranchCondTwoRegsHelper1(&riscv64::Riscv64Assembler::Bne, "Bne", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, Blt) {
  BranchCondTwoRegsHelper1(&riscv64::Riscv64Assembler::Blt, "Blt", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, Bge) {
  BranchCondTwoRegsHelper1(&riscv64::Riscv64Assembler::Bge, "Bge", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, Bltu) {
  BranchCondTwoRegsHelper1(&riscv64::Riscv64Assembler::Bltu, "Bltu", /* is_bare= */ true);
}

TEST_F(AssemblerRISCV64Test, Bgeu) {
  BranchCondTwoRegsHelper1(&riscv64::Riscv64Assembler::Bgeu, "Bgeu", /* is_bare= */ true);
}
#endif

#if TEST_RV32I_U
TEST_F(AssemblerRISCV64Test, Lui) {
  DriverStr(RepeatrIb(&riscv64::Riscv64Assembler::Lui, 20, "lui {reg}, {imm}"), "Lui");
}

TEST_F(AssemblerRISCV64Test, Auipc) {
  DriverStr(RepeatrIb(&riscv64::Riscv64Assembler::Auipc, 20, "auipc {reg}, {imm}"), "Auipc");
}
#endif

#if TEST_RV32I_J
TEST_F(AssemblerRISCV64Test, Jal) {
  BranchHelper1(&riscv64::Riscv64Assembler::Jal, "Jal");
}
#endif

#if TEST_RV64I_R
TEST_F(AssemblerRISCV64Test, Addw) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Addw, "addw {reg1}, {reg2}, {reg3}"), "Addw");
}

TEST_F(AssemblerRISCV64Test, Subw) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Subw, "subw {reg1}, {reg2}, {reg3}"), "Subw");
}

TEST_F(AssemblerRISCV64Test, Sllw) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Sllw, "sllw {reg1}, {reg2}, {reg3}"), "Sllw");
}

TEST_F(AssemblerRISCV64Test, Srlw) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Srlw, "srlw {reg1}, {reg2}, {reg3}"), "Srlw");
}

TEST_F(AssemblerRISCV64Test, Sraw) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Sraw, "sraw {reg1}, {reg2}, {reg3}"), "Sraw");
}
#endif

#if TEST_RV64I_I
TEST_F(AssemblerRISCV64Test, Lwu) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Lwu, -11, "lwu {reg1}, {imm}({reg2})"), "Lwu");
}

TEST_F(AssemblerRISCV64Test, Ld) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Ld, -11, "ld {reg1}, {imm}({reg2})"), "Ld");
}

TEST_F(AssemblerRISCV64Test, Slli) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Slli, 6, "slli {reg1}, {reg2}, {imm}"), "Slli");
}

TEST_F(AssemblerRISCV64Test, Srli) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Srli, 6, "srli {reg1}, {reg2}, {imm}"), "Srli");
}

TEST_F(AssemblerRISCV64Test, Srai) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Srai, 6, "srai {reg1}, {reg2}, {imm}"), "Srai");
}

TEST_F(AssemblerRISCV64Test, Addiw) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Addiw, -11, "addiw {reg1}, {reg2}, {imm}"), "Addiw");
}

TEST_F(AssemblerRISCV64Test, Slliw) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Slliw, 5, "slliw {reg1}, {reg2}, {imm}"), "Slliw");
}

TEST_F(AssemblerRISCV64Test, Srliw) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Srliw, 5, "srliw {reg1}, {reg2}, {imm}"), "Srliw");
}

TEST_F(AssemblerRISCV64Test, Sraiw) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Sraiw, 5, "sraiw {reg1}, {reg2}, {imm}"), "Sraiw");
}
#endif

#if TEST_RV64I_S
TEST_F(AssemblerRISCV64Test, Sd) {
  DriverStr(RepeatrrIb(&riscv64::Riscv64Assembler::Sd, -11, "sd {reg1}, {imm}({reg2})"), "Sd");
}
#endif

#if TEST_RV32M_R
TEST_F(AssemblerRISCV64Test, Mul) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Mul, "mul {reg1}, {reg2}, {reg3}"), "Mul");
}

TEST_F(AssemblerRISCV64Test, Mulh) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Mulh, "mulh {reg1}, {reg2}, {reg3}"), "Mulh");
}

TEST_F(AssemblerRISCV64Test, Mulhsu) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Mulhsu, "mulhsu {reg1}, {reg2}, {reg3}"), "Mulhsu");
}

TEST_F(AssemblerRISCV64Test, Mulhu) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Mulhu, "mulhu {reg1}, {reg2}, {reg3}"), "Mulhu");
}

TEST_F(AssemblerRISCV64Test, Div) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Div, "div {reg1}, {reg2}, {reg3}"), "Div");
}

TEST_F(AssemblerRISCV64Test, Divu) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Divu, "divu {reg1}, {reg2}, {reg3}"), "Divu");
}

TEST_F(AssemblerRISCV64Test, Rem) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Rem, "rem {reg1}, {reg2}, {reg3}"), "Rem");
}

TEST_F(AssemblerRISCV64Test, Remu) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Remu, "remu {reg1}, {reg2}, {reg3}"), "Remu");
}
#endif

#if TEST_RV64M_R
TEST_F(AssemblerRISCV64Test, Mulw) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Mulw, "mulw {reg1}, {reg2}, {reg3}"), "Mulw");
}

TEST_F(AssemblerRISCV64Test, Divw) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Divw, "divw {reg1}, {reg2}, {reg3}"), "Divw");
}

TEST_F(AssemblerRISCV64Test, Divuw) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Divuw, "divuw {reg1}, {reg2}, {reg3}"), "Divuw");
}

TEST_F(AssemblerRISCV64Test, Remw) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Remw, "remw {reg1}, {reg2}, {reg3}"), "Remw");
}

TEST_F(AssemblerRISCV64Test, Remuw) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::Remuw, "remuw {reg1}, {reg2}, {reg3}"), "Remuw");
}
#endif

#if TEST_RV32A_R
TEST_F(AssemblerRISCV64Test, LrW) {
  DriverStr(Repeatrr(&riscv64::Riscv64Assembler::LrW, "lr.w {reg1}, ({reg2})"), "LrW");
}

TEST_F(AssemblerRISCV64Test, ScW) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::ScW, "sc.w {reg1}, {reg2}, ({reg3})"), "ScW");
}

TEST_F(AssemblerRISCV64Test, AmoSwapW) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoSwapW, "amoswap.w {reg1}, {reg2}, ({reg3})"), "AmoSwapW");
}

TEST_F(AssemblerRISCV64Test, AmoAddW) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoAddW, "amoadd.w {reg1}, {reg2}, ({reg3})"), "AmoAddW");
}

TEST_F(AssemblerRISCV64Test, AmoXorW) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoXorW, "amoxor.w {reg1}, {reg2}, ({reg3})"), "AmoXorW");
}

TEST_F(AssemblerRISCV64Test, AmoAndW) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoAndW, "amoand.w {reg1}, {reg2}, ({reg3})"), "AmoAndW");
}

TEST_F(AssemblerRISCV64Test, AmoOrW) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoOrW, "amoor.w {reg1}, {reg2}, ({reg3})"), "AmoOrW");
}

TEST_F(AssemblerRISCV64Test, AmoMinW) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoMinW, "amomin.w {reg1}, {reg2}, ({reg3})"), "AmoMinW");
}

TEST_F(AssemblerRISCV64Test, AmoMaxW) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoMaxW, "amomax.w {reg1}, {reg2}, ({reg3})"), "AmoMaxW");
}

TEST_F(AssemblerRISCV64Test, AmoMinuW) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoMinuW, "amominu.w {reg1}, {reg2}, ({reg3})"), "AmoMinuW");
}

TEST_F(AssemblerRISCV64Test, AmoMaxuW) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoMaxuW, "amomaxu.w {reg1}, {reg2}, ({reg3})"), "AmoMaxuW");
}
#endif

#if TEST_RV64A_R
TEST_F(AssemblerRISCV64Test, LrD) {
  DriverStr(Repeatrr(&riscv64::Riscv64Assembler::LrD, "lr.d {reg1}, ({reg2})"), "LrD");
}

TEST_F(AssemblerRISCV64Test, ScD) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::ScD, "sc.d {reg1}, {reg2}, ({reg3})"), "ScD");
}

TEST_F(AssemblerRISCV64Test, AmoSwapD) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoSwapD, "amoswap.d {reg1}, {reg2}, ({reg3})"), "AmoSwapD");
}

TEST_F(AssemblerRISCV64Test, AmoAddD) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoAddD, "amoadd.d {reg1}, {reg2}, ({reg3})"), "AmoAddD");
}

TEST_F(AssemblerRISCV64Test, AmoXorD) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoXorD, "amoxor.d {reg1}, {reg2}, ({reg3})"), "AmoXorD");
}

TEST_F(AssemblerRISCV64Test, AmoAndD) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoAndD, "amoand.d {reg1}, {reg2}, ({reg3})"), "AmoAndD");
}

TEST_F(AssemblerRISCV64Test, AmoOrD) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoOrD, "amoor.d {reg1}, {reg2}, ({reg3})"), "AmoOrD");
}

TEST_F(AssemblerRISCV64Test, AmoMinD) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoMinD, "amomin.d {reg1}, {reg2}, ({reg3})"), "AmoMinD");
}

TEST_F(AssemblerRISCV64Test, AmoMaxD) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoMaxD, "amomax.d {reg1}, {reg2}, ({reg3})"), "AmoMaxD");
}

TEST_F(AssemblerRISCV64Test, AmoMinuD) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoMinuD, "amominu.d {reg1}, {reg2}, ({reg3})"), "AmoMinuD");
}

TEST_F(AssemblerRISCV64Test, AmoMaxuD) {
  DriverStr(Repeatrrr(&riscv64::Riscv64Assembler::AmoMaxuD, "amomaxu.d {reg1}, {reg2}, ({reg3})"), "AmoMaxuD");
}
#endif

#if TEST_RV32F_I
TEST_F(AssemblerRISCV64Test, FLw) {
  DriverStr(RepeatFrIb(&riscv64::Riscv64Assembler::FLw, -11, "flw {reg1}, {imm}({reg2})"), "FLw");
}
#endif

#if TEST_RV32F_S
TEST_F(AssemblerRISCV64Test, FSw) {
  DriverStr(RepeatFrIb(&riscv64::Riscv64Assembler::FSw, 2, "fsw {reg1}, {imm}({reg2})"), "FSw");
}
#endif

#if TEST_RV32F_R
# if 1
// takes too long time, close it for quick testing
TEST_F(AssemblerRISCV64Test, FMAddS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FMAddS, "fmadd.s {reg1}, {reg2}, {reg3}, {reg4}"), "FMAddS");
}

TEST_F(AssemblerRISCV64Test, FMSubS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FMSubS, "fmsub.s {reg1}, {reg2}, {reg3}, {reg4}"), "FMSubS");
}

TEST_F(AssemblerRISCV64Test, FNMSubS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FNMSubS, "fnmsub.s {reg1}, {reg2}, {reg3}, {reg4}"), "FNMSubS");
}

TEST_F(AssemblerRISCV64Test, FNMAddS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FNMAddS, "fnmadd.s {reg1}, {reg2}, {reg3}, {reg4}"), "FNMAddS");
}
#endif

TEST_F(AssemblerRISCV64Test, FAddS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FAddS, "fadd.s {reg1}, {reg2}, {reg3}"), "FAddS");
}

TEST_F(AssemblerRISCV64Test, FSubS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FSubS, "fsub.s {reg1}, {reg2}, {reg3}"), "FSubS");
}

TEST_F(AssemblerRISCV64Test, FMulS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FMulS, "fmul.s {reg1}, {reg2}, {reg3}"), "FMulS");
}

TEST_F(AssemblerRISCV64Test, FDivS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FDivS, "fdiv.s {reg1}, {reg2}, {reg3}"), "FDivS");
}

TEST_F(AssemblerRISCV64Test, FSqrtS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::FSqrtS, "fsqrt.s {reg1}, {reg2}"), "FSqrtS");
}

TEST_F(AssemblerRISCV64Test, FSgnjS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FSgnjS, "fsgnj.s {reg1}, {reg2}, {reg3}"), "FSgnjS");
}

TEST_F(AssemblerRISCV64Test, FSgnjnS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FSgnjnS, "fsgnjn.s {reg1}, {reg2}, {reg3}"), "FSgnjnS");
}

TEST_F(AssemblerRISCV64Test, FSgnjxS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FSgnjxS, "fsgnjx.s {reg1}, {reg2}, {reg3}"), "FSgnjxS");
}

TEST_F(AssemblerRISCV64Test, FMinS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FMinS, "fmin.s {reg1}, {reg2}, {reg3}"), "FMinS");
}

TEST_F(AssemblerRISCV64Test, FMaxS) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FMaxS, "fmax.s {reg1}, {reg2}, {reg3}"), "FMaxS");
}

TEST_F(AssemblerRISCV64Test, FCvtWS) {
  // DriverStr(RepeatrF(&riscv64::Riscv64Assembler::FCvtWS, "fcvt.w.s {reg1}, {reg2}, {reg3}"), "FCvtWS");
}

TEST_F(AssemblerRISCV64Test, FCvtWuS) {
  DriverStr(RepeatrF(&riscv64::Riscv64Assembler::FCvtWuS, "fcvt.wu.s {reg1}, {reg2}"), "FCvtWuS");
}

TEST_F(AssemblerRISCV64Test, FMvXW) {
  DriverStr(RepeatrF(&riscv64::Riscv64Assembler::FMvXW, "fmv.x.w {reg1}, {reg2}"), "FMvXW");
}

TEST_F(AssemblerRISCV64Test, FEqS) {
  DriverStr(RepeatrFF(&riscv64::Riscv64Assembler::FEqS, "feq.s {reg1}, {reg2}, {reg3}"), "FEqS");
}

TEST_F(AssemblerRISCV64Test, FLtS) {
  DriverStr(RepeatrFF(&riscv64::Riscv64Assembler::FLtS, "flt.s {reg1}, {reg2}, {reg3}"), "FLtS");
}

TEST_F(AssemblerRISCV64Test, FLeS) {
  DriverStr(RepeatrFF(&riscv64::Riscv64Assembler::FLeS, "fle.s {reg1}, {reg2}, {reg3}"), "FLeS");
}

TEST_F(AssemblerRISCV64Test, FClassS) {
  DriverStr(RepeatrF(&riscv64::Riscv64Assembler::FClassS, "fclass.s {reg1}, {reg2}"), "FClassS");
}

TEST_F(AssemblerRISCV64Test, FCvtSW) {
  DriverStr(RepeatFr(&riscv64::Riscv64Assembler::FCvtSW, "fcvt.s.w {reg1}, {reg2}"), "FCvtSW");
}

TEST_F(AssemblerRISCV64Test, FCvtSWu) {
  DriverStr(RepeatFr(&riscv64::Riscv64Assembler::FCvtSWu, "fcvt.s.wu {reg1}, {reg2}"), "FCvtSWu");
}

TEST_F(AssemblerRISCV64Test, FMvWX) {
  DriverStr(RepeatFr(&riscv64::Riscv64Assembler::FMvWX, "fmv.w.x {reg1}, {reg2}"), "FMvWX");
}
#endif

#if TEST_RV64F_R
TEST_F(AssemblerRISCV64Test, FCvtLS) {
  // DriverStr(RepeatrF(&riscv64::Riscv64Assembler::FCvtLS, "fcvt.l.s {reg1}, {reg2}, {reg3}"), "FCvtLS");
}
TEST_F(AssemblerRISCV64Test, FCvtLuS) {
  DriverStr(RepeatrF(&riscv64::Riscv64Assembler::FCvtLuS, "fcvt.lu.s {reg1}, {reg2}"), "FCvtLuS");
}
TEST_F(AssemblerRISCV64Test, FCvtSL) {
  DriverStr(RepeatFr(&riscv64::Riscv64Assembler::FCvtSL, "fcvt.s.l {reg1}, {reg2}"), "FCvtSL");
}
TEST_F(AssemblerRISCV64Test, FCvtSLu) {
  DriverStr(RepeatFr(&riscv64::Riscv64Assembler::FCvtSLu, "fcvt.s.lu {reg1}, {reg2}"), "FCvtSLu");
}
#endif

#if TEST_RV32D_I
TEST_F(AssemblerRISCV64Test, FLd) {
  DriverStr(RepeatFrIb(&riscv64::Riscv64Assembler::FLd, -11, "fld {reg1}, {imm}({reg2})"), "FLw");
}
#endif

#if TEST_RV32D_S
TEST_F(AssemblerRISCV64Test, FSd) {
  DriverStr(RepeatFrIb(&riscv64::Riscv64Assembler::FSd, 2, "fsd {reg1}, {imm}({reg2})"), "FSw");
}
#endif

#if TEST_RV32D_R
# if 1
// takes too long time, close it for quick testing
TEST_F(AssemblerRISCV64Test, FMAddD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FMAddD, "fmadd.d {reg1}, {reg2}, {reg3}, {reg4}"), "FMAddD");
}

TEST_F(AssemblerRISCV64Test, FMSubD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FMSubD, "fmsub.d {reg1}, {reg2}, {reg3}, {reg4}"), "FMSubD");
}

TEST_F(AssemblerRISCV64Test, FNMSubD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FNMSubD, "fnmsub.d {reg1}, {reg2}, {reg3}, {reg4}"), "FNMSubD");
}

TEST_F(AssemblerRISCV64Test, FNMAddD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FNMAddD, "fnmadd.d {reg1}, {reg2}, {reg3}, {reg4}"), "FNMAddD");
}
#endif

TEST_F(AssemblerRISCV64Test, FAddD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FAddD, "fadd.d {reg1}, {reg2}, {reg3}"), "FAddD");
}

TEST_F(AssemblerRISCV64Test, FSubD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FSubD, "fsub.d {reg1}, {reg2}, {reg3}"), "FSubD");
}

TEST_F(AssemblerRISCV64Test, FMulD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FMulD, "fmul.d {reg1}, {reg2}, {reg3}"), "FMulD");
}

TEST_F(AssemblerRISCV64Test, FDivD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FDivD, "fdiv.d {reg1}, {reg2}, {reg3}"), "FDivD");
}

TEST_F(AssemblerRISCV64Test, FSqrtD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::FSqrtD, "fsqrt.d {reg1}, {reg2}"), "FSqrtD");
}

TEST_F(AssemblerRISCV64Test, FSgnjD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FSgnjD, "fsgnj.d {reg1}, {reg2}, {reg3}"), "FSgnjD");
}

TEST_F(AssemblerRISCV64Test, FSgnjnD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FSgnjnD, "fsgnjn.d {reg1}, {reg2}, {reg3}"), "FSgnjnD");
}

TEST_F(AssemblerRISCV64Test, FSgnjxD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FSgnjxD, "fsgnjx.d {reg1}, {reg2}, {reg3}"), "FSgnjxD");
}

TEST_F(AssemblerRISCV64Test, FMinD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FMinD, "fmin.d {reg1}, {reg2}, {reg3}"), "FMinD");
}

TEST_F(AssemblerRISCV64Test, FMaxD) {
  DriverStr(RepeatFFF(&riscv64::Riscv64Assembler::FMaxD, "fmax.d {reg1}, {reg2}, {reg3}"), "FMaxD");
}

TEST_F(AssemblerRISCV64Test, FCvtSD) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::FCvtSD, "fcvt.s.d {reg1}, {reg2}"), "FCvtSD");
}

TEST_F(AssemblerRISCV64Test, FCvtDS) {
  DriverStr(RepeatFF(&riscv64::Riscv64Assembler::FCvtDS, "fcvt.d.s {reg1}, {reg2}"), "FCvtDS");
}

TEST_F(AssemblerRISCV64Test, FEqD) {
  DriverStr(RepeatrFF(&riscv64::Riscv64Assembler::FEqD, "feq.d {reg1}, {reg2}, {reg3}"), "FEqD");
}

TEST_F(AssemblerRISCV64Test, FLtD) {
  DriverStr(RepeatrFF(&riscv64::Riscv64Assembler::FLtD, "flt.d {reg1}, {reg2}, {reg3}"), "FLtD");
}

TEST_F(AssemblerRISCV64Test, FLeD) {
  DriverStr(RepeatrFF(&riscv64::Riscv64Assembler::FLeD, "fle.d {reg1}, {reg2}, {reg3}"), "FLeD");
}

TEST_F(AssemblerRISCV64Test, FClassD) {
  DriverStr(RepeatrF(&riscv64::Riscv64Assembler::FClassD, "fclass.d {reg1}, {reg2}"), "FClassD");
}

TEST_F(AssemblerRISCV64Test, FCvtWD) {
  // DriverStr(RepeatrF(&riscv64::Riscv64Assembler::FCvtWD, "fcvt.w.d {reg1}, {reg2}, {reg3}"), "FCvtWD");
}

TEST_F(AssemblerRISCV64Test, FCvtWuD) {
  DriverStr(RepeatrF(&riscv64::Riscv64Assembler::FCvtWuD, "fcvt.wu.d {reg1}, {reg2}"), "FCvtWuD");
}

TEST_F(AssemblerRISCV64Test, FCvtDW) {
  DriverStr(RepeatFr(&riscv64::Riscv64Assembler::FCvtDW, "fcvt.d.w {reg1}, {reg2}"), "FCvtDW");
}

TEST_F(AssemblerRISCV64Test, FCvtDWu) {
  DriverStr(RepeatFr(&riscv64::Riscv64Assembler::FCvtDWu, "fcvt.d.wu {reg1}, {reg2}"), "FCvtDWu");
}
#endif

#if TEST_RV64D_R
TEST_F(AssemblerRISCV64Test, FCvtLD) {
  // DriverStr(RepeatrF(&riscv64::Riscv64Assembler::FCvtLD, "fcvt.l.d {reg1}, {reg2}, {reg3}"), "FCvtLD");
}
TEST_F(AssemblerRISCV64Test, FCvtLuD) {
  DriverStr(RepeatrF(&riscv64::Riscv64Assembler::FCvtLuD, "fcvt.lu.d {reg1}, {reg2}"), "FCvtLuD");
}
TEST_F(AssemblerRISCV64Test, FMvXD) {
  DriverStr(RepeatrF(&riscv64::Riscv64Assembler::FMvXD, "fmv.x.d {reg1}, {reg2}"), "FMvXD");
}
TEST_F(AssemblerRISCV64Test, FCvtDL) {
  DriverStr(RepeatFr(&riscv64::Riscv64Assembler::FCvtDL, "fcvt.d.l {reg1}, {reg2}"), "FCvtDL");
}
TEST_F(AssemblerRISCV64Test, FCvtDLu) {
  DriverStr(RepeatFr(&riscv64::Riscv64Assembler::FCvtDLu, "fcvt.d.lu {reg1}, {reg2}"), "FCvtDLu");
}
TEST_F(AssemblerRISCV64Test, FMvDX) {
  DriverStr(RepeatFr(&riscv64::Riscv64Assembler::FMvDX, "fmv.d.x {reg1}, {reg2}"), "FMvDX");
}
#endif

#endif

#undef __

}  // namespace art
