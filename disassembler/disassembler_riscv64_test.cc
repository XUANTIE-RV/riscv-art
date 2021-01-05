/*
 * Copyright (C) 2012 The Android Open Source Project
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
#include "gtest/gtest.h"

#include "base/malloc_arena_pool.h"
#include "utils/riscv64/assembler_riscv64.h"
#include "disassembler_riscv64.h"
#include "utils/riscv64/constants_riscv64.h"

#include "android-base/stringprintf.h"

using android::base::StringPrintf;  // NOLINT(build/namespaces)

#define COMPARE(ASM, EXP)                                                \
  do {                                                                   \
    GetAssembler()->ASM;                                                 \
    GetAssembler()->FinalizeCode();                                      \
    std::vector<uint8_t> data(GetAssembler()->CodeSize());               \
    MemoryRegion code(data.data(), data.size());                         \
    GetAssembler()->FinalizeInstructions(code);                          \
    uint32_t encoding = *reinterpret_cast<uint32_t*>(&data[0]);          \
    std::string instruction = GetDisassembler()->DumpInstruction(encoding); \
    EXPECT_EQ(instruction, EXP); \
  } while (0)

#define __ GetAssembler()->

void ThreadOffsetNameFunctionDumy(std::ostream&, uint32_t) {
}

namespace art {
namespace riscv64 {
class DisassemblerRiscv64Test : public ::testing::Test {
 public:
  using Ass = riscv64::Riscv64Assembler;
  using Dss = riscv64::DisassemblerRiscv64;

  Ass* GetAssembler() {
    return assembler_.get();
  }

  Dss* GetDisassembler() {
    return disassembler_.get();
  }

 protected:
  void SetUp() override {
    allocator_.reset(new ArenaAllocator(&pool_));
    assembler_.reset(CreateAssembler(allocator_.get()));
    disassembler_.reset(CreateDisassembler());
  }

 private:
  virtual Ass* CreateAssembler(ArenaAllocator* allocator) {
    return new (allocator) Ass(allocator);
  }

  virtual Dss* CreateDisassembler() {
    return new Dss(new DisassemblerOptions(false, nullptr, nullptr, false,
        &ThreadOffsetNameFunctionDumy));
  }

  MallocArenaPool pool_;
  std::unique_ptr<ArenaAllocator> allocator_;
  std::unique_ptr<Ass> assembler_;
  std::unique_ptr<Dss> disassembler_;
};

#if 1
TEST_F(DisassemblerRiscv64Test, Add) {
  COMPARE(Add(A0, A1, A2), "add a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Sub) {
  COMPARE(Sub(A0, A1, A2), "sub a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Sll) {
  COMPARE(Sll(A0, A1, A2), "sll a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Slt) {
  COMPARE(Slt(A0, A1, A2), "slt a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Sltu) {
  COMPARE(Sltu(A0, A1, A2), "sltu a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Xor) {
  COMPARE(Xor(A0, A1, A2), "xor a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Srl) {
  COMPARE(Srl(A0, A1, A2), "srl a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Sra) {
  COMPARE(Sra(A0, A1, A2), "sra a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Or) {
  COMPARE(Or(A0, A1, A2), "or a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, And) {
  COMPARE(And(A0, A1, A2), "and a0,a1,a2");
}

#endif
// RV32I-S
TEST_F(DisassemblerRiscv64Test, Sb) {
  COMPARE(Sb(A0, A1, -255), "sb a0,-255(a1)");
}

TEST_F(DisassemblerRiscv64Test, Sh) {
  COMPARE(Sh(A0, A1, -255), "sh a0,-255(a1)");
}

TEST_F(DisassemblerRiscv64Test, Sw) {
  COMPARE(Sw(A0, A1, -255), "sw a0,-255(a1)");
}

// RV32I-B
TEST_F(DisassemblerRiscv64Test, Beq) {
  COMPARE(Beq(A0, A1, -256), "beq a0,a1,-256");
}

TEST_F(DisassemblerRiscv64Test, Bne) {
  COMPARE(Bne(A0, A1, -256), "bne a0,a1,-256");
}

TEST_F(DisassemblerRiscv64Test, Blt) {
  COMPARE(Blt(A0, A1, -256), "blt a0,a1,-256");
}

TEST_F(DisassemblerRiscv64Test, Bge) {
  COMPARE(Bge(A0, A1, -256), "bge a0,a1,-256");
}

TEST_F(DisassemblerRiscv64Test, Bltu) {
  COMPARE(Bltu(A0, A1, -256), "bltu a0,a1,-256");
}

TEST_F(DisassemblerRiscv64Test, Bgeu) {
  COMPARE(Bgeu(A0, A1, -256), "bgeu a0,a1,-256");
}

// RV32I-I
TEST_F(DisassemblerRiscv64Test, Jalr) {
  COMPARE(Jalr(A0, A1, -2047), "jalr a0,-2047(a1)");
}

TEST_F(DisassemblerRiscv64Test, Lb) {
  COMPARE(Lb(A0, A1, -256), "lb a0,-256(a1)");
}

TEST_F(DisassemblerRiscv64Test, Lh) {
  COMPARE(Lh(A0, A1, -256), "lh a0,-256(a1)");
}

TEST_F(DisassemblerRiscv64Test, Lw) {
  COMPARE(Lw(A0, A1, -256), "lw a0,-256(a1)");
}

TEST_F(DisassemblerRiscv64Test, Lbu) {
  COMPARE(Lbu(A0, A1, -256), "lbu a0,-256(a1)");
}

TEST_F(DisassemblerRiscv64Test, Lhu) {
  COMPARE(Lhu(A0, A1, -256), "lhu a0,-256(a1)");
}

TEST_F(DisassemblerRiscv64Test, Addi) {
  COMPARE(Addi(A0, A1, -256), "addi a0,a1,-256");
}

TEST_F(DisassemblerRiscv64Test, Fence) {
  COMPARE(Fence(15, 15), "fence iorw,iorw");
}

TEST_F(DisassemblerRiscv64Test, FenceI) {
  COMPARE(FenceI(), "fence.i ");
}

TEST_F(DisassemblerRiscv64Test, Ecall) {
  COMPARE(Ecall(), "ecall ");
}

TEST_F(DisassemblerRiscv64Test, Ebreak) {
  COMPARE(Ebreak(), "ebreak ");
}

TEST_F(DisassemblerRiscv64Test, Csrrw) {
  COMPARE(Csrrw(A0, A1, 0x1F), "csrrw a0,31,a1");
}

TEST_F(DisassemblerRiscv64Test, Csrrs) {
  COMPARE(Csrrs(A0, A1, 0x1F), "csrrs a0,31,a1");
}

TEST_F(DisassemblerRiscv64Test, Csrrc) {
  COMPARE(Csrrc(A0, A1, 0x1F), "csrrc a0,31,a1");
}

TEST_F(DisassemblerRiscv64Test, Csrrwi) {
  COMPARE(Csrrwi(A0, 0x001, 1), "csrrwi a0,fflags,1");
}

TEST_F(DisassemblerRiscv64Test, Csrrsi) {
  COMPARE(Csrrsi(A0, 0x001, 1), "csrrsi a0,fflags,1");
}

TEST_F(DisassemblerRiscv64Test, Csrrci) {
  COMPARE(Csrrci(A0, 0x001, 1), "csrrci a0,fflags,1");
}

// RV32I-
TEST_F(DisassemblerRiscv64Test, Slti) {
  COMPARE(Slti(A0, A1, -256), "slti a0,a1,-256");
}

TEST_F(DisassemblerRiscv64Test, Sltiu) {
  COMPARE(Sltiu(A0, A1, -256), "sltiu a0,a1,-256");
}

TEST_F(DisassemblerRiscv64Test, Xori) {
  COMPARE(Xori(A0, A1, -256), "xori a0,a1,-256");
}

TEST_F(DisassemblerRiscv64Test, Ori) {
  COMPARE(Ori(A0, A1, -256), "ori a0,a1,-256");
}

TEST_F(DisassemblerRiscv64Test, Andi) {
  COMPARE(Andi(A0, A1, -256), "andi a0,a1,-256");
}

TEST_F(DisassemblerRiscv64Test, Slli) {
  COMPARE(Slli(A0, A1, 16), "slli a0,a1,16");
}

TEST_F(DisassemblerRiscv64Test, Srli) {
  COMPARE(Srli(A0, A1, 16), "srli a0,a1,16");
}

TEST_F(DisassemblerRiscv64Test, Srai) {
  COMPARE(Srai(A0, A1, 16), "srai a0,a1,16");
}

TEST_F(DisassemblerRiscv64Test, Lui) {
  COMPARE(Lui(A0, -524287), "lui a0,-524287");
}

TEST_F(DisassemblerRiscv64Test, Auipc) {
  COMPARE(Auipc(A0, -524287), "auipc a0,-524287");
}

// RV32I-J
TEST_F(DisassemblerRiscv64Test, Jal) {
  COMPARE(Jal(A0, -1048574), "jal a0,-1048574");
}

// RV64I-I
TEST_F(DisassemblerRiscv64Test, Lwu) {
  COMPARE(Lwu(A0, A1, -2047), "lwu a0,-2047(a1)");
}

TEST_F(DisassemblerRiscv64Test, Ld) {
  COMPARE(Ld(A0, A1, -2047), "ld a0,-2047(a1)");
}

TEST_F(DisassemblerRiscv64Test, Addiw) {
  COMPARE(Addiw(A0, A1, -2047), "addiw a0,a1,-2047");
}

// RV64I-S
TEST_F(DisassemblerRiscv64Test, Sd) {
  COMPARE(Sd(A0, A1, -2047), "sd a0,-2047(a1)");
}

// RV64I-R
TEST_F(DisassemblerRiscv64Test, Slliw) {
  COMPARE(Slliw(A0, A1, 16), "slliw a0,a1,16");
}

TEST_F(DisassemblerRiscv64Test, Srliw) {
  COMPARE(Srliw(A0, A1, 16), "srliw a0,a1,16");
}

TEST_F(DisassemblerRiscv64Test, Sraiw) {
  COMPARE(Sraiw(A0, A1, 16), "sraiw a0,a1,16");
}

TEST_F(DisassemblerRiscv64Test, Addw) {
  COMPARE(Addw(A0, A1, A2), "addw a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Subw) {
  COMPARE(Subw(A0, A1, A2), "subw a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Sllw) {
  COMPARE(Sllw(A0, A1, A2), "sllw a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Srlw) {
  COMPARE(Srlw(A0, A1, A2), "srlw a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Sraw) {
  COMPARE(Sraw(A0, A1, A2), "sraw a0,a1,a2");
}

// RV32M-R
TEST_F(DisassemblerRiscv64Test, Mul) {
  COMPARE(Mul(A0, A1, A2), "mul a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Mulh) {
  COMPARE(Mulh(A0, A1, A2), "mulh a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Mulhsu) {
  COMPARE(Mulhsu(A0, A1, A2), "mulhsu a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Mulhu) {
  COMPARE(Mulhu(A0, A1, A2), "mulhu a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Div) {
  COMPARE(Div(A0, A1, A2), "div a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Divu) {
  COMPARE(Divu(A0, A1, A2), "divu a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Rem) {
  COMPARE(Rem(A0, A1, A2), "rem a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Remu) {
  COMPARE(Remu(A0, A1, A2), "remu a0,a1,a2");
}

// RV64M-R
TEST_F(DisassemblerRiscv64Test, Mulw) {
  COMPARE(Mulw(A0, A1, A2), "mulw a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Divw) {
  COMPARE(Divw(A0, A1, A2), "divw a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Divuw) {
  COMPARE(Divuw(A0, A1, A2), "divuw a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Remw) {
  COMPARE(Remw(A0, A1, A2), "remw a0,a1,a2");
}

TEST_F(DisassemblerRiscv64Test, Remuw) {
  COMPARE(Remuw(A0, A1, A2), "remuw a0,a1,a2");
}

// RV32A-R
TEST_F(DisassemblerRiscv64Test, LrW) {
  COMPARE(LrW(A0, A1), "lr.w a0,(a1)");
}

TEST_F(DisassemblerRiscv64Test, ScW) {
  COMPARE(ScW(A0, A1, A2), "sc.w a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoSwapW) {
  COMPARE(AmoSwapW(A0, A1, A2), "amoswap.w a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoAddW) {
  COMPARE(AmoAddW(A0, A1, A2), "amoadd.w a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoXorW) {
  COMPARE(AmoXorW(A0, A1, A2), "amoxor.w a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoAndW) {
  COMPARE(AmoAndW(A0, A1, A2), "amoand.w a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoOrW) {
  COMPARE(AmoOrW(A0, A1, A2), "amoor.w a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoMinW) {
  COMPARE(AmoMinW(A0, A1, A2), "amomin.w a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoMaxW) {
  COMPARE(AmoMaxW(A0, A1, A2), "amomax.w a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoMinuW) {
  COMPARE(AmoMinuW(A0, A1, A2), "amominu.w a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoMaxuW) {
  COMPARE(AmoMaxuW(A0, A1, A2), "amomaxu.w a0,a1,(a2)");
}

// RV64A-R
TEST_F(DisassemblerRiscv64Test, LrD) {
  COMPARE(LrD(A0, A1), "lr.d a0,(a1)");
}

TEST_F(DisassemblerRiscv64Test, ScD) {
  COMPARE(ScD(A0, A1, A2), "sc.d a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoSwapD) {
  COMPARE(AmoSwapD(A0, A1, A2), "amoswap.d a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoAddD) {
  COMPARE(AmoAddD(A0, A1, A2), "amoadd.d a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoXorD) {
  COMPARE(AmoXorD(A0, A1, A2), "amoxor.d a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoAndD) {
  COMPARE(AmoAndD(A0, A1, A2), "amoand.d a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoOrD) {
  COMPARE(AmoOrD(A0, A1, A2), "amoor.d a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoMinD) {
  COMPARE(AmoMinD(A0, A1, A2), "amomin.d a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoMaxD) {
  COMPARE(AmoMaxD(A0, A1, A2), "amomax.d a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoMinuD) {
  COMPARE(AmoMinuD(A0, A1, A2), "amominu.d a0,a1,(a2)");
}

TEST_F(DisassemblerRiscv64Test, AmoMaxuD) {
  COMPARE(AmoMaxuD(A0, A1, A2), "amomaxu.d a0,a1,(a2)");
}

// RV32F-I
TEST_F(DisassemblerRiscv64Test, FLw) {
  COMPARE(FLw(FT0, A1, -2047), "flw ft0,-2047(a1)");
}

// RV32F-S
TEST_F(DisassemblerRiscv64Test, FSw) {
  COMPARE(FSw(FT0, A1, -2047), "fsw ft0,-2047(a1)");
}

// RV32F-R
TEST_F(DisassemblerRiscv64Test, FMAddS) {
  COMPARE(FMAddS(FT0, FT1, FT2, FT3), "fmadd.s ft0,ft1,ft2,ft3");
}

TEST_F(DisassemblerRiscv64Test, FMSubS) {
  COMPARE(FMSubS(FT0, FT1, FT2, FT3), "fmsub.s ft0,ft1,ft2,ft3");
}

TEST_F(DisassemblerRiscv64Test, FNMSubS) {
  COMPARE(FNMSubS(FT0, FT1, FT2, FT3), "fnmsub.s ft0,ft1,ft2,ft3");
}

TEST_F(DisassemblerRiscv64Test, FNMAddS) {
  COMPARE(FNMAddS(FT0, FT1, FT2, FT3), "fnmadd.s ft0,ft1,ft2,ft3");
}

TEST_F(DisassemblerRiscv64Test, FAddS) {
  COMPARE(FAddS(FT0, FT1, FT2), "fadd.s ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FSubS) {
  COMPARE(FSubS(FT0, FT1, FT2), "fsub.s ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FMulS) {
  COMPARE(FMulS(FT0, FT1, FT2), "fmul.s ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FDivS) {
  COMPARE(FDivS(FT0, FT1, FT2), "fdiv.s ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FSqrtS) {
  COMPARE(FSqrtS(FT0, FT1), "fsqrt.s ft0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FSgnjS) {
  COMPARE(FSgnjS(FT0, FT1, FT2), "fsgnj.s ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FSgnjnS) {
  COMPARE(FSgnjnS(FT0, FT1, FT2), "fsgnjn.s ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FSgnjxS) {
  COMPARE(FSgnjxS(FT0, FT1, FT2), "fsgnjx.s ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FMinS) {
  COMPARE(FMinS(FT0, FT1, FT2), "fmin.s ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FMaxS) {
  COMPARE(FMaxS(FT0, FT1, FT2), "fmax.s ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FCvtWS) {
  COMPARE(FCvtWS(A0, FT1), "fcvt.w.s a0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FCvtWuS) {
  COMPARE(FCvtWuS(A0, FT1), "fcvt.wu.s a0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FMvXW) {
  COMPARE(FMvXW(A0, FT1), "fmv.x.w a0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FEqS) {
  COMPARE(FEqS(A0, FT1, FT2), "feq.s a0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FClassS) {
  COMPARE(FClassS(A0, FT1), "fclass.s a0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FCvtSW) {
  COMPARE(FCvtSW(FT0, A1), "fcvt.s.w ft0,a1");
}

TEST_F(DisassemblerRiscv64Test, FCvtSWu) {
  COMPARE(FCvtSWu(FT0, A1), "fcvt.s.wu ft0,a1");
}

TEST_F(DisassemblerRiscv64Test, FMvWX) {
  COMPARE(FMvWX(FT0, A1), "fmv.w.x ft0,a1");
}

// RV64F-R
TEST_F(DisassemblerRiscv64Test, FCvtLS) {
  COMPARE(FCvtLS(A0, FT1), "fcvt.l.s a0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FCvtLuS) {
  COMPARE(FCvtLuS(A0, FT1), "fcvt.lu.s a0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FCvtSL) {
  COMPARE(FCvtSL(FT0, A1), "fcvt.s.l ft0,a1");
}

TEST_F(DisassemblerRiscv64Test, FCvtSLu) {
  COMPARE(FCvtSLu(FT0, A1), "fcvt.s.lu ft0,a1");
}

// RV32D-I
TEST_F(DisassemblerRiscv64Test, FLd) {
  COMPARE(FLd(FT0, A1, -2047), "fld ft0,-2047(a1)");
}

// RV32D-S
TEST_F(DisassemblerRiscv64Test, FSd) {
  COMPARE(FSd(FT0, A1, -2047), "fsd ft0,-2047(a1)");
}

// RV32D-R
TEST_F(DisassemblerRiscv64Test, FMAddD) {
  COMPARE(FMAddD(FT0, FT1, FT2, FT3), "fmadd.d ft0,ft1,ft2,ft3");
}

TEST_F(DisassemblerRiscv64Test, FMSubD) {
  COMPARE(FMSubD(FT0, FT1, FT2, FT3), "fmsub.d ft0,ft1,ft2,ft3");
}

TEST_F(DisassemblerRiscv64Test, FNMSubD) {
  COMPARE(FNMSubD(FT0, FT1, FT2, FT3), "fnmsub.d ft0,ft1,ft2,ft3");
}

TEST_F(DisassemblerRiscv64Test, FNMAddD) {
  COMPARE(FNMAddD(FT0, FT1, FT2, FT3), "fnmadd.d ft0,ft1,ft2,ft3");
}

TEST_F(DisassemblerRiscv64Test, FAddD) {
  COMPARE(FAddD(FT0, FT1, FT2), "fadd.d ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FSubD) {
  COMPARE(FSubD(FT0, FT1, FT2), "fsub.d ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FMulD) {
  COMPARE(FMulD(FT0, FT1, FT2), "fmul.d ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FDivD) {
  COMPARE(FDivD(FT0, FT1, FT2), "fdiv.d ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FSqrtD) {
  COMPARE(FSqrtD(FT0, FT1), "fsqrt.d ft0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FSgnjD) {
  COMPARE(FSgnjD(FT0, FT1, FT2), "fsgnj.d ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FSgnjnD) {
  COMPARE(FSgnjnD(FT0, FT1, FT2), "fsgnjn.d ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FSgnjxD) {
  COMPARE(FSgnjxD(FT0, FT1, FT2), "fsgnjx.d ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FMinD) {
  COMPARE(FMinD(FT0, FT1, FT2), "fmin.d ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FMaxD) {
  COMPARE(FMaxD(FT0, FT1, FT2), "fmax.d ft0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FCvtSD) {
  COMPARE(FCvtSD(FT0, FT1), "fcvt.s.d ft0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FCvtDS) {
  COMPARE(FCvtDS(FT0, FT1), "fcvt.d.s ft0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FEqD) {
  COMPARE(FEqD(A0, FT1, FT2), "feq.d a0,ft1,ft2");
}

TEST_F(DisassemblerRiscv64Test, FClassD) {
  COMPARE(FClassD(A0, FT1), "fclass.d a0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FCvtWD) {
  COMPARE(FCvtWD(A0, FT1), "fcvt.w.d a0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FCvtWuD) {
  COMPARE(FCvtWuD(A0, FT1), "fcvt.wu.d a0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FCvtDW) {
  COMPARE(FCvtDW(FT0, A1), "fcvt.d.w ft0,a1");
}

TEST_F(DisassemblerRiscv64Test, FCvtDWu) {
  COMPARE(FCvtDWu(FT0, A1), "fcvt.d.wu ft0,a1");
}

// RV64D-R
TEST_F(DisassemblerRiscv64Test, FCvtLD) {
  COMPARE(FCvtLD(A0, FT1), "fcvt.l.d a0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FCvtLuD) {
  COMPARE(FCvtLuD(A0, FT1), "fcvt.lu.d a0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FMvXD) {
  COMPARE(FMvXD(A0, FT1), "fmv.x.d a0,ft1");
}

TEST_F(DisassemblerRiscv64Test, FCvtDL) {
  COMPARE(FCvtDL(FT0, A1), "fcvt.d.l ft0,a1");
}

TEST_F(DisassemblerRiscv64Test, FCvtDLu) {
  COMPARE(FCvtDLu(FT0, A1), "fcvt.d.lu ft0,a1");
}

TEST_F(DisassemblerRiscv64Test, FMvDX) {
  COMPARE(FMvDX(FT0, A1), "fmv.d.x ft0,a1");
}

}  // namespace riscv64
}  // namespace art
