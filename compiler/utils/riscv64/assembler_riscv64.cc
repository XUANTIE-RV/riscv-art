/*
 * Copyright (C) 2014 The Android Open Source Project
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

#include "base/bit_utils.h"
#include "base/casts.h"
#include "base/memory_region.h"
#include "entrypoints/quick/quick_entrypoints.h"
#include "entrypoints/quick/quick_entrypoints_enum.h"
#include "thread.h"

namespace art {
namespace riscv64 {

static_assert(static_cast<size_t>(kRiscv64PointerSize) == kRiscv64DoublewordSize,
              "Unexpected Riscv64 pointer size.");
static_assert(kRiscv64PointerSize == PointerSize::k64, "Unexpected Riscv64 pointer size.");


void Riscv64Assembler::FinalizeCode() {
  for (auto& exception_block : exception_blocks_) {
    EmitExceptionPoll(&exception_block);
  }
  ReserveJumpTableSpace();
  EmitLiterals();
  PromoteBranches();
}

void Riscv64Assembler::FinalizeInstructions(const MemoryRegion& region) {
  EmitBranches();
  EmitJumpTables();
  Assembler::FinalizeInstructions(region);
  PatchCFI();
}

void Riscv64Assembler::PatchCFI() {
  if (cfi().NumberOfDelayedAdvancePCs() == 0u) {
    return;
  }

  using DelayedAdvancePC = DebugFrameOpCodeWriterForAssembler::DelayedAdvancePC;
  const auto data = cfi().ReleaseStreamAndPrepareForDelayedAdvancePC();
  const std::vector<uint8_t>& old_stream = data.first;
  const std::vector<DelayedAdvancePC>& advances = data.second;

  // Refill our data buffer with patched opcodes.
  cfi().ReserveCFIStream(old_stream.size() + advances.size() + 16);
  size_t stream_pos = 0;
  for (const DelayedAdvancePC& advance : advances) {
    DCHECK_GE(advance.stream_pos, stream_pos);
    // Copy old data up to the point where advance was issued.
    cfi().AppendRawData(old_stream, stream_pos, advance.stream_pos);
    stream_pos = advance.stream_pos;
    // Insert the advance command with its final offset.
    size_t final_pc = GetAdjustedPosition(advance.pc);
    cfi().AdvancePC(final_pc);
  }
  // Copy the final segment if any.
  cfi().AppendRawData(old_stream, stream_pos, old_stream.size());
}

void Riscv64Assembler::EmitBranches() {
  CHECK(!overwriting_);
  // Switch from appending instructions at the end of the buffer to overwriting
  // existing instructions (branch placeholders) in the buffer.
  overwriting_ = true;
  for (auto& branch : branches_) {
    EmitBranch(&branch);
  }
  overwriting_ = false;
}

void Riscv64Assembler::Emit(uint32_t value) {
  if (overwriting_) {
    // Branches to labels are emitted into their placeholders here.
    buffer_.Store<uint32_t>(overwrite_location_, value);
    overwrite_location_ += sizeof(uint32_t);
  } else {
    // Other instructions are simply appended at the end here.
    AssemblerBuffer::EnsureCapacity ensured(&buffer_);
    buffer_.Emit<uint32_t>(value);
  }
}

void Riscv64Assembler::EmitRsd(int opcode, GpuRegister rs, GpuRegister rd,
                              int shamt, int funct) {
  CHECK_NE(rs, kNoGpuRegister);
  CHECK_NE(rd, kNoGpuRegister);
  uint32_t encoding = static_cast<uint32_t>(opcode) << kOpcodeShift |
                      static_cast<uint32_t>(rs) << kRsShift |
                      static_cast<uint32_t>(ZERO) << kRtShift |
                      static_cast<uint32_t>(rd) << kRdShift |
                      shamt << kShamtShift |
                      funct;
  Emit(encoding);
}

void Riscv64Assembler::EmitRtd(int opcode, GpuRegister rt, GpuRegister rd,
                              int shamt, int funct) {
  CHECK_NE(rt, kNoGpuRegister);
  CHECK_NE(rd, kNoGpuRegister);
  uint32_t encoding = static_cast<uint32_t>(opcode) << kOpcodeShift |
                      static_cast<uint32_t>(ZERO) << kRsShift |
                      static_cast<uint32_t>(rt) << kRtShift |
                      static_cast<uint32_t>(rd) << kRdShift |
                      shamt << kShamtShift |
                      funct;
  Emit(encoding);
}

void Riscv64Assembler::EmitI(int opcode, GpuRegister rs, GpuRegister rt, uint16_t imm) {
  CHECK_NE(rs, kNoGpuRegister);
  CHECK_NE(rt, kNoGpuRegister);
  uint32_t encoding = static_cast<uint32_t>(opcode) << kOpcodeShift |
                      static_cast<uint32_t>(rs) << kRsShift |
                      static_cast<uint32_t>(rt) << kRtShift |
                      imm;
  Emit(encoding);
}

void Riscv64Assembler::EmitI21(int opcode, GpuRegister rs, uint32_t imm21) {
  CHECK_NE(rs, kNoGpuRegister);
  CHECK(IsUint<21>(imm21)) << imm21;
  uint32_t encoding = static_cast<uint32_t>(opcode) << kOpcodeShift |
                      static_cast<uint32_t>(rs) << kRsShift |
                      imm21;
  Emit(encoding);
}

void Riscv64Assembler::EmitI26(int opcode, uint32_t imm26) {
  CHECK(IsUint<26>(imm26)) << imm26;
  uint32_t encoding = static_cast<uint32_t>(opcode) << kOpcodeShift | imm26;
  Emit(encoding);
}

void Riscv64Assembler::EmitFR(int opcode, int fmt, FpuRegister ft, FpuRegister fs, FpuRegister fd,
                             int funct) {
  CHECK_NE(ft, kNoFpuRegister);
  CHECK_NE(fs, kNoFpuRegister);
  CHECK_NE(fd, kNoFpuRegister);
  uint32_t encoding = static_cast<uint32_t>(opcode) << kOpcodeShift |
                      fmt << kFmtShift |
                      static_cast<uint32_t>(ft) << kFtShift |
                      static_cast<uint32_t>(fs) << kFsShift |
                      static_cast<uint32_t>(fd) << kFdShift |
                      funct;
  Emit(encoding);
}

void Riscv64Assembler::EmitFI(int opcode, int fmt, FpuRegister ft, uint16_t imm) {
  CHECK_NE(ft, kNoFpuRegister);
  uint32_t encoding = static_cast<uint32_t>(opcode) << kOpcodeShift |
                      fmt << kFmtShift |
                      static_cast<uint32_t>(ft) << kFtShift |
                      imm;
  Emit(encoding);
}

void Riscv64Assembler::EmitMsa3R(int operation,
                                int df,
                                VectorRegister wt,
                                VectorRegister ws,
                                VectorRegister wd,
                                int minor_opcode) {
  CHECK_NE(wt, kNoVectorRegister);
  CHECK_NE(ws, kNoVectorRegister);
  CHECK_NE(wd, kNoVectorRegister);
  uint32_t encoding = static_cast<uint32_t>(kMsaMajorOpcode) << kOpcodeShift |
                      operation << kMsaOperationShift |
                      df << kDfShift |
                      static_cast<uint32_t>(wt) << kWtShift |
                      static_cast<uint32_t>(ws) << kWsShift |
                      static_cast<uint32_t>(wd) << kWdShift |
                      minor_opcode;
  Emit(encoding);
}

void Riscv64Assembler::EmitMsaBIT(int operation,
                                 int df_m,
                                 VectorRegister ws,
                                 VectorRegister wd,
                                 int minor_opcode) {
  CHECK_NE(ws, kNoVectorRegister);
  CHECK_NE(wd, kNoVectorRegister);
  uint32_t encoding = static_cast<uint32_t>(kMsaMajorOpcode) << kOpcodeShift |
                      operation << kMsaOperationShift |
                      df_m << kDfMShift |
                      static_cast<uint32_t>(ws) << kWsShift |
                      static_cast<uint32_t>(wd) << kWdShift |
                      minor_opcode;
  Emit(encoding);
}

void Riscv64Assembler::EmitMsaELM(int operation,
                                 int df_n,
                                 VectorRegister ws,
                                 VectorRegister wd,
                                 int minor_opcode) {
  CHECK_NE(ws, kNoVectorRegister);
  CHECK_NE(wd, kNoVectorRegister);
  uint32_t encoding = static_cast<uint32_t>(kMsaMajorOpcode) << kOpcodeShift |
                      operation << kMsaELMOperationShift |
                      df_n << kDfNShift |
                      static_cast<uint32_t>(ws) << kWsShift |
                      static_cast<uint32_t>(wd) << kWdShift |
                      minor_opcode;
  Emit(encoding);
}

void Riscv64Assembler::EmitMsaMI10(int s10,
                                  GpuRegister rs,
                                  VectorRegister wd,
                                  int minor_opcode,
                                  int df) {
  CHECK_NE(rs, kNoGpuRegister);
  CHECK_NE(wd, kNoVectorRegister);
  CHECK(IsUint<10>(s10)) << s10;
  uint32_t encoding = static_cast<uint32_t>(kMsaMajorOpcode) << kOpcodeShift |
                      s10 << kS10Shift |
                      static_cast<uint32_t>(rs) << kWsShift |
                      static_cast<uint32_t>(wd) << kWdShift |
                      minor_opcode << kS10MinorShift |
                      df;
  Emit(encoding);
}

void Riscv64Assembler::EmitMsaI10(int operation,
                                 int df,
                                 int i10,
                                 VectorRegister wd,
                                 int minor_opcode) {
  CHECK_NE(wd, kNoVectorRegister);
  CHECK(IsUint<10>(i10)) << i10;
  uint32_t encoding = static_cast<uint32_t>(kMsaMajorOpcode) << kOpcodeShift |
                      operation << kMsaOperationShift |
                      df << kDfShift |
                      i10 << kI10Shift |
                      static_cast<uint32_t>(wd) << kWdShift |
                      minor_opcode;
  Emit(encoding);
}

void Riscv64Assembler::EmitMsa2R(int operation,
                                int df,
                                VectorRegister ws,
                                VectorRegister wd,
                                int minor_opcode) {
  CHECK_NE(ws, kNoVectorRegister);
  CHECK_NE(wd, kNoVectorRegister);
  uint32_t encoding = static_cast<uint32_t>(kMsaMajorOpcode) << kOpcodeShift |
                      operation << kMsa2ROperationShift |
                      df << kDf2RShift |
                      static_cast<uint32_t>(ws) << kWsShift |
                      static_cast<uint32_t>(wd) << kWdShift |
                      minor_opcode;
  Emit(encoding);
}

void Riscv64Assembler::EmitMsa2RF(int operation,
                                 int df,
                                 VectorRegister ws,
                                 VectorRegister wd,
                                 int minor_opcode) {
  CHECK_NE(ws, kNoVectorRegister);
  CHECK_NE(wd, kNoVectorRegister);
  uint32_t encoding = static_cast<uint32_t>(kMsaMajorOpcode) << kOpcodeShift |
                      operation << kMsa2RFOperationShift |
                      df << kDf2RShift |
                      static_cast<uint32_t>(ws) << kWsShift |
                      static_cast<uint32_t>(wd) << kWdShift |
                      minor_opcode;
  Emit(encoding);
}

void Riscv64Assembler::Addu(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  Addw(rd, rs, rt);
}

void Riscv64Assembler::Addiu(GpuRegister rd, GpuRegister rs, int16_t imm16) {
  if (IsInt<12>(imm16)) {
    Addiw(rd, rs, imm16);
  } else {
    int32_t l = imm16 & 0xFFF;
    int32_t h = imm16 >> 12;
    if ((l & 0x800) != 0) {
      h += 1;
    }
    // rs and rd may be same or be TMP, use TMP2 here.
    Lui(TMP2, h);
    Addiw(TMP2, TMP2, l);
    Addu(rd, TMP2, rs);
  }
}

void Riscv64Assembler::Daddu(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  Add(rd, rs, rt);;
}

void Riscv64Assembler::Daddiu(GpuRegister rd, GpuRegister rs, int16_t imm16) {
  if (IsInt<12>(imm16)) {
    Addi(rd, rs, imm16);
  } else {
    int32_t l = imm16 & 0xFFF;
    int32_t h = imm16 >> 12;
    if ((l & 0x800) != 0) {
      h += 1;  // overflow ?
    }
    // rs and rd may be same or be TMP, use TMP2 here.
    Lui(TMP2, h);
    Addiw(TMP2, TMP2, l);
    Daddu(rd, TMP2, rs);
  }
}

void Riscv64Assembler::Subu(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  Subw(rd, rs, rt);
}

void Riscv64Assembler::Dsubu(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  Sub(rd, rs, rt);
}

void Riscv64Assembler::MulR6(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  // MulR6 --> Mulw in Riscv64
  Mulw(rd, rs, rt);
}

void Riscv64Assembler::MuhR6(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  // There's no instruction in Riscv64 can get the high 32bit of 32-bit Multiplication.
  // Shift left 32 for both of source operands
  // Use TMP2 and T6 here
  Slli(TMP2, rs, 32);
  Slli(T6, rt, 32);
  Mul(rd, TMP2, T6);   // rd <-- (rs x rt)'s 64-bit result
  Srai(rd, rd, 32);  // get the high 32bit result and keep sign
}

void Riscv64Assembler::DivR6(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  // DivR6 --> Divw in Riscv64
  Divw(rd, rs, rt);
}

void Riscv64Assembler::ModR6(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  Remw(rd, rs, rt);
}

void Riscv64Assembler::DivuR6(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  // DivuR6 --> Divuw in Riscv64
  Divuw(rd, rs, rt);
}

void Riscv64Assembler::ModuR6(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  Remuw(rd, rs, rt);
}

void Riscv64Assembler::Dmul(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  // Dmul --> Mul in Riscv64
  Mul(rd, rs, rt);
}

void Riscv64Assembler::Dmuh(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  Mulh(rd, rs, rt);
}

void Riscv64Assembler::Ddiv(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  Div(rd, rs, rt);
}

void Riscv64Assembler::Dmod(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  Rem(rd, rs, rt);
}

void Riscv64Assembler::Ddivu(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  Divu(rd, rs, rt);
}

void Riscv64Assembler::Dmodu(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  Remu(rd, rs, rt);
}

void Riscv64Assembler::Bitswap(GpuRegister rd, GpuRegister rt) {
  assert(0);
  EmitRtd(0x1f, rt, rd, 0x0, 0x20);
}

void Riscv64Assembler::Dbitswap(GpuRegister rd, GpuRegister rt) {
  assert(0);
  EmitRtd(0x1f, rt, rd, 0x0, 0x24);
}

void Riscv64Assembler::Seb(GpuRegister rd, GpuRegister rt) {
  Srliw(rd, rt, 24);  // Sign extend bit 7 to hight 32-bit
  Srai(rd, rd, 24);
}

void Riscv64Assembler::Seh(GpuRegister rd, GpuRegister rt) {
  Srliw(rd, rt, 16);  // Sign extend bit 15 to hight 32-bit
  Srai(rd, rd, 16);
}

void Riscv64Assembler::Dsbh(GpuRegister rd, GpuRegister rt) {
  assert(0);
  EmitRtd(0x1f, rt, rd, 0x2, 0x24);
}

void Riscv64Assembler::Dshd(GpuRegister rd, GpuRegister rt) {
  assert(0);
  EmitRtd(0x1f, rt, rd, 0x5, 0x24);
}

void Riscv64Assembler::Dext(GpuRegister rt, GpuRegister rs, int pos, int size) {
  CHECK(IsUint<5>(pos)) << pos;
  CHECK(IsUint<5>(size - 1)) << size;
  Srli(rt, rs, pos);
  Slli(rt, rt, (64-size));
  Srli(rt, rt, (64-size));
}

void Riscv64Assembler::Ins(GpuRegister rd, GpuRegister rt, int pos, int size) {
  assert(0);
  CHECK(IsUint<5>(pos)) << pos;
  CHECK(IsUint<5>(size - 1)) << size;
  CHECK(IsUint<5>(pos + size - 1)) << pos << " + " << size;
  EmitR(0x1f, rt, rd, static_cast<GpuRegister>(pos + size - 1), pos, 0x04);
}

void Riscv64Assembler::Dinsm(GpuRegister rt, GpuRegister rs, int pos, int size) {
  assert(0);
  CHECK(IsUint<5>(pos)) << pos;
  CHECK(2 <= size && size <= 64) << size;
  CHECK(IsUint<5>(pos + size - 33)) << pos << " + " << size;
  EmitR(0x1f, rs, rt, static_cast<GpuRegister>(pos + size - 33), pos, 0x5);
}

void Riscv64Assembler::Dinsu(GpuRegister rt, GpuRegister rs, int pos, int size) {
  assert(0);
  CHECK(IsUint<5>(pos - 32)) << pos;
  CHECK(IsUint<5>(size - 1)) << size;
  CHECK(IsUint<5>(pos + size - 33)) << pos << " + " << size;
  EmitR(0x1f, rs, rt, static_cast<GpuRegister>(pos + size - 33), pos - 32, 0x6);
}

void Riscv64Assembler::Dins(GpuRegister rt, GpuRegister rs, int pos, int size) {
  assert(0);
  CHECK(IsUint<5>(pos)) << pos;
  CHECK(IsUint<5>(size - 1)) << size;
  CHECK(IsUint<5>(pos + size - 1)) << pos << " + " << size;
  EmitR(0x1f, rs, rt, static_cast<GpuRegister>(pos + size - 1), pos, 0x7);
}

void Riscv64Assembler::DblIns(GpuRegister rt, GpuRegister rs, int pos, int size) {
  assert(0);
  if (pos >= 32) {
    Dinsu(rt, rs, pos, size);
  } else if ((static_cast<int64_t>(pos) + size - 1) >= 32) {
    Dinsm(rt, rs, pos, size);
  } else {
    Dins(rt, rs, pos, size);
  }
}

void Riscv64Assembler::Lsa(GpuRegister rd, GpuRegister rs, GpuRegister rt, int saPlusOne) {
  CHECK(1 <= saPlusOne && saPlusOne <= 4) << saPlusOne;
  Slli(TMP2, rs, saPlusOne);
  Addw(rd, TMP2, rt);
}

void Riscv64Assembler::Dlsa(GpuRegister rd, GpuRegister rs, GpuRegister rt, int saPlusOne) {
  CHECK(1 <= saPlusOne && saPlusOne <= 4) << saPlusOne;
  Slli(TMP2, rs, saPlusOne);
  Add(rd, TMP2, rt);
}

void Riscv64Assembler::Wsbh(GpuRegister rd, GpuRegister rt) {
  assert(0);
  EmitRtd(0x1f, rt, rd, 2, 0x20);
}

void Riscv64Assembler::Sc(GpuRegister rt, GpuRegister base, int16_t imm9) {
  CHECK(IsInt<9>(imm9));
  if (imm9 != 0) {
    Addi(TMP2, base, imm9);
    ScW(rt, rt, TMP2);   // todo: sucess, 0; fail, 1, which is opposite to mips;
  } else {
    ScW(rt, rt, base);
  }
}

void Riscv64Assembler::Scd(GpuRegister rt, GpuRegister base, int16_t imm9) {
  CHECK(IsInt<9>(imm9));
  if (imm9 != 0) {
    Addi(TMP2, base, imm9);
    ScD(rt, rt, TMP2);   // todo: sucess, 0; fail, 1, which is opposite to mips;
  } else {
    ScD(rt, rt, base);
  }
}

void Riscv64Assembler::Ll(GpuRegister rt, GpuRegister base, int16_t imm9) {
  CHECK(IsInt<9>(imm9));
  if (imm9 != 0) {
    Addi(TMP2, base, imm9);
    LrW(rt, TMP2);   // todo: sucess, 0; fail, 1, which is opposite to mips;
  } else {
    LrW(rt, base);
  }
}

void Riscv64Assembler::Lld(GpuRegister rt, GpuRegister base, int16_t imm9) {
  CHECK(IsInt<9>(imm9));
  if (imm9 != 0) {
    Addi(TMP2, base, imm9);
    LrD(rt, TMP2);   // todo: sucess, 0; fail, 1, which is opposite to mips;
  } else {
    LrD(rt, base);
  }
}

void Riscv64Assembler::Sll(GpuRegister rd, GpuRegister rt, int shamt) {
  Slliw(rd, rt, shamt);
}

void Riscv64Assembler::Srl(GpuRegister rd, GpuRegister rt, int shamt) {
  Srliw(rd, rt, shamt);
}

void Riscv64Assembler::Rotr(GpuRegister rd, GpuRegister rt, int shamt) {
  CHECK(0 <= shamt < 32) << shamt;
  // Riscv64 codegen don't use the blocked registers for rd, rt, rs till now.
  // It's safe to use scratch registers here.
  Srliw(TMP, rt, shamt);
  Slliw(rd, rt, 32-shamt);  // logical shift left (32 -shamt)
  Or(rd, rd, TMP);
}

void Riscv64Assembler::Sra(GpuRegister rd, GpuRegister rt, int shamt) {
  Sraiw(rd, rt, shamt);
}

void Riscv64Assembler::Sllv(GpuRegister rd, GpuRegister rt, GpuRegister rs) {
  Sllw(rd, rt, rs);
}

void Riscv64Assembler::Rotrv(GpuRegister rd, GpuRegister rt, GpuRegister rs) {
  // Riscv64 codegen don't use the blocked registers for rd, rt, rs till now.
  // It's safe to use TMP scratch registers here.
  Srlw(TMP, rt, rs);
  Subw(TMP2, ZERO, rs);  // TMP2 = -rs
  Addiw(TMP2, TMP2, 32);   // TMP2 = 32 -rs
  Andi(TMP2, TMP2, 0x1F);
  Sllw(rd, rt, TMP2);
  Or(rd, rd, TMP);
}

void Riscv64Assembler::Srlv(GpuRegister rd, GpuRegister rt, GpuRegister rs) {
  Srlw(rd, rt, rs);
}

void Riscv64Assembler::Srav(GpuRegister rd, GpuRegister rt, GpuRegister rs) {
  Sraw(rd, rt, rs);
}

void Riscv64Assembler::Dsll(GpuRegister rd, GpuRegister rt, int shamt) {
  Slli(rd, rt, shamt);
}

void Riscv64Assembler::Dsrl(GpuRegister rd, GpuRegister rt, int shamt) {
  Srli(rd, rt, shamt);
}

void Riscv64Assembler::Drotr(GpuRegister rd, GpuRegister rt, int shamt) {
  CHECK(0 <= shamt < 32) << shamt;
  // Riscv64 codegen don't use the blocked registers for rd, rt, rs till now.
  // It's safe to use scratch registers here.
  Srli(TMP, rt, shamt);
  Slli(rd, rt, (64 - shamt));
  Or(rd, rd, TMP);
}

void Riscv64Assembler::Dsra(GpuRegister rd, GpuRegister rt, int shamt) {
  Srai(rd, rt, shamt);
}

void Riscv64Assembler::Dsll32(GpuRegister rd, GpuRegister rt, int shamt) {
  CHECK(0 <= shamt < 32) << shamt;

  Slli(rd, rt, shamt+32);
}

void Riscv64Assembler::Dsrl32(GpuRegister rd, GpuRegister rt, int shamt) {
  CHECK(0 <= shamt < 32) << shamt;

  Srli(rd, rt, shamt+32);
}

void Riscv64Assembler::Drotr32(GpuRegister rd, GpuRegister rt, int shamt) {
  CHECK(0 <= shamt < 32) << shamt;
  Drotr(rd, rt, 32+shamt);
}

void Riscv64Assembler::Dsra32(GpuRegister rd, GpuRegister rt, int shamt) {
  CHECK(0 <= shamt < 32) << shamt;

  Srai(rd, rt, shamt+32);
}

void Riscv64Assembler::Dsllv(GpuRegister rd, GpuRegister rt, GpuRegister rs) {
  Sll(rd, rt, rs);
}

void Riscv64Assembler::Dsrlv(GpuRegister rd, GpuRegister rt, GpuRegister rs) {
  Srl(rd, rt, rs);
}

void Riscv64Assembler::Drotrv(GpuRegister rd, GpuRegister rt, GpuRegister rs) {
  // Riscv64 codegen don't use the blocked registers for rd, rt, rs till now.
  // It's safe to use scratch registers here.
  Srl(TMP, rt, rs);
  Sub(TMP2, ZERO, rs);  // TMP2 = -rs
  Addi(TMP2, TMP2, 64);   // TMP2 = 64 -rs
  Sll(rd, rt, TMP2);
  Or(rd, rd, TMP);
}

void Riscv64Assembler::Dsrav(GpuRegister rd, GpuRegister rt, GpuRegister rs) {
  Sra(rd, rt, rs);
}

void Riscv64Assembler::Lwpc(GpuRegister rs, uint32_t imm19) {
  Auipc(rs, (imm19<<2)>>12);
  Lw(rs, rs, (imm19<<2)&0xFFF);
}

void Riscv64Assembler::Lwupc(GpuRegister rs, uint32_t imm19) {
  Auipc(rs, (imm19<<2)>>12);
  Lwu(rs, rs, (imm19<<2)&0xFFF);
}

void Riscv64Assembler::Ldpc(GpuRegister rs, uint32_t imm18) {
  Auipc(rs, (imm18<<2)>>12);
  Ld(rs, rs, (imm18<<2)&0xFFF);
}

void Riscv64Assembler::Aui(GpuRegister rt, GpuRegister rs, uint16_t imm16) {
  int32_t l = imm16 & 0xFFF;
  int32_t h = imm16 >> 12;
  if ((l & 0x800) != 0) {
    h += 1;  // overflow ?
  }

  // rs and rd may be same or be TMP, use TMP2 here.
  Lui(TMP2, h);
  Addiw(TMP2, TMP2, l);
  Slliw(TMP2, TMP2, 16);
  Addw(rt, rs, TMP2);
}

void Riscv64Assembler::Daui(GpuRegister rt, GpuRegister rs, uint16_t imm16) {
  int32_t l = imm16 & 0xFFF;
  int32_t h = imm16 >> 12;
  if ((l & 0x800) != 0) {
    h += 1;  // overflow ?
  }

  // rs and rd may be same or be TMP, use TMP2 here.
  Lui(TMP2, h);
  Addi(TMP2, TMP2, l);
  Slli(TMP2, TMP2, 16);
  Add(rt, rs, TMP2);
}

void Riscv64Assembler::Dahi(GpuRegister rs, uint16_t imm16) {
  int32_t l = imm16 & 0xFFF;
  int32_t h = imm16 >> 12;
  if ((l & 0x800) != 0) {
    h += 1;  // overflow ?
  }

  // rs and rd may be same or be TMP, use TMP2 here.
  Lui(TMP2, h);
  Addi(TMP2, TMP2, l);
  Slli(TMP2, TMP2, 32);
  Add(rs, rs, TMP2);
}

void Riscv64Assembler::Dati(GpuRegister rs, uint16_t imm16) {
  int32_t l = imm16 & 0xFFF;
  int32_t h = imm16 >> 12;
  if ((l & 0x800) != 0) {
    h += 1;  // overflow ?
  }

  // rs and rd may be same or be TMP, use TMP2 here.
  Lui(TMP2, h);
  Addi(TMP2, TMP2, l);
  Slli(TMP2, TMP2, 48);
  Add(rs, rs, TMP2);
}

void Riscv64Assembler::Sync(uint32_t stype) {
  // FIXME: T-HEAD, just for simplify, use normal fence here for all types.
  Fence(0xf, 0xf);
}

void Riscv64Assembler::Seleqz(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  Move(TMP2, rt);
  Move(rd, rs);
  Beq(TMP2, ZERO, 8);
  Move(rd, ZERO);
}

void Riscv64Assembler::Selnez(GpuRegister rd, GpuRegister rs, GpuRegister rt) {
  Move(TMP2, rt);
  Move(rd, rs);
  Bne(TMP2, ZERO, 8);
  Move(rd, ZERO);
}

void Riscv64Assembler::Clz(GpuRegister rd, GpuRegister rs) {
  assert(0);
  EmitRsd(0, rs, rd, 0x01, 0x10);
}

void Riscv64Assembler::Clo(GpuRegister rd, GpuRegister rs) {
  assert(0);
  EmitRsd(0, rs, rd, 0x01, 0x11);
}

void Riscv64Assembler::Dclz(GpuRegister rd, GpuRegister rs) {
  assert(0);
  EmitRsd(0, rs, rd, 0x01, 0x12);
}

void Riscv64Assembler::Dclo(GpuRegister rd, GpuRegister rs) {
  assert(0);
  EmitRsd(0, rs, rd, 0x01, 0x13);
}

void Riscv64Assembler::Jalr(GpuRegister rd, GpuRegister rs) {
  Jalr(rd, rs, 0);
}

void Riscv64Assembler::Jalr(GpuRegister rs) {
  Jalr(RA, rs, 0);
}

void Riscv64Assembler::Jr(GpuRegister rs) {
  Jalr(ZERO, rs, 0);
}

void Riscv64Assembler::Addiupc(GpuRegister rs, uint32_t imm19) {
  CHECK(IsUint<19>(imm19)) << imm19;
  Auipc(rs, (imm19<<2)>>12);
  Addi(rs, rs, (imm19<<2)&0xFFF);
}

void Riscv64Assembler::Bc(uint32_t imm20) {
  Jal(ZERO, imm20);
}

void Riscv64Assembler::Balc(uint32_t imm20) {
  Jal(RA, imm20);
}

void Riscv64Assembler::Jic(GpuRegister rt, uint16_t imm16) {
  Jalr(ZERO, rt, imm16);
}

void Riscv64Assembler::Jialc(GpuRegister rt, uint16_t imm16) {
  Jalr(RA, rt, imm16);
}

void Riscv64Assembler::Bltc(GpuRegister rs, GpuRegister rt, uint16_t imm12) {
  CHECK_NE(rs, ZERO);
  CHECK_NE(rt, ZERO);
  CHECK_NE(rs, rt);
  Blt(rs, rt, imm12);
}

void Riscv64Assembler::Bltzc(GpuRegister rt, uint16_t imm12) {
  CHECK_NE(rt, ZERO);
  Blt(rt, ZERO, imm12);
}

void Riscv64Assembler::Bgtzc(GpuRegister rt, uint16_t imm12) {
  CHECK_NE(rt, ZERO);
  Blt(ZERO, rt, imm12);
}

void Riscv64Assembler::Bgec(GpuRegister rs, GpuRegister rt, uint16_t imm12) {
  CHECK_NE(rs, ZERO);
  CHECK_NE(rt, ZERO);
  CHECK_NE(rs, rt);
  Bge(rs, rt, imm12);
}

void Riscv64Assembler::Bgezc(GpuRegister rt, uint16_t imm12) {
  CHECK_NE(rt, ZERO);
  Bge(rt, ZERO, imm12);
}

void Riscv64Assembler::Blezc(GpuRegister rt, uint16_t imm12) {
  CHECK_NE(rt, ZERO);
  Bge(ZERO, rt, imm12);
}

void Riscv64Assembler::Bltuc(GpuRegister rs, GpuRegister rt, uint16_t imm12) {
  CHECK_NE(rs, ZERO);
  CHECK_NE(rt, ZERO);
  CHECK_NE(rs, rt);
  Bltu(rs, rt, imm12);
}

void Riscv64Assembler::Bgeuc(GpuRegister rs, GpuRegister rt, uint16_t imm12) {
  CHECK_NE(rs, ZERO);
  CHECK_NE(rt, ZERO);
  CHECK_NE(rs, rt);
  Bgeu(rs, rt, imm12);
}

void Riscv64Assembler::Beqc(GpuRegister rs, GpuRegister rt, uint16_t imm12) {
  CHECK_NE(rs, ZERO);
  CHECK_NE(rt, ZERO);
  CHECK_NE(rs, rt);
  Beq(rs, rt, imm12);
}

void Riscv64Assembler::Bnec(GpuRegister rs, GpuRegister rt, uint16_t imm12) {
  CHECK_NE(rs, ZERO);
  CHECK_NE(rt, ZERO);
  CHECK_NE(rs, rt);
  Bne(rs, rt, imm12);
}

void Riscv64Assembler::Beqzc(GpuRegister rs, uint32_t imm12) {
  CHECK_NE(rs, ZERO);
  Beq(rs, ZERO, imm12);
}

void Riscv64Assembler::Bnezc(GpuRegister rs, uint32_t imm12) {
  CHECK_NE(rs, ZERO);
  Bne(rs, ZERO, imm12);
}

void Riscv64Assembler::EmitBcond(BranchCondition cond,
                                  GpuRegister rs,
                                  GpuRegister rt,
                                  uint32_t imm16_21) {
  switch (cond) {
    case kCondLT:
      Bltc(rs, rt, imm16_21);
      break;
    case kCondGE:
      Bgec(rs, rt, imm16_21);
      break;
    case kCondLE:
      Bgec(rt, rs, imm16_21);
      break;
    case kCondGT:
      Bltc(rt, rs, imm16_21);
      break;
    case kCondLTZ:
      CHECK_EQ(rt, ZERO);
      Bltzc(rs, imm16_21);
      break;
    case kCondGEZ:
      CHECK_EQ(rt, ZERO);
      Bgezc(rs, imm16_21);
      break;
    case kCondLEZ:
      CHECK_EQ(rt, ZERO);
      Blezc(rs, imm16_21);
      break;
    case kCondGTZ:
      CHECK_EQ(rt, ZERO);
      Bgtzc(rs, imm16_21);
      break;
    case kCondEQ:
      Beqc(rs, rt, imm16_21);
      break;
    case kCondNE:
      Bnec(rs, rt, imm16_21);
      break;
    case kCondEQZ:
      CHECK_EQ(rt, ZERO);
      Beqzc(rs, imm16_21);
      break;
    case kCondNEZ:
      CHECK_EQ(rt, ZERO);
      Bnezc(rs, imm16_21);
      break;
    case kCondLTU:
      Bltuc(rs, rt, imm16_21);
      break;
    case kCondGEU:
      Bgeuc(rs, rt, imm16_21);
      break;
    case kUncond:
      // LOG(FATAL) << "Unexpected branch condition " << cond;
      LOG(FATAL) << "Unexpected branch condition ";
      UNREACHABLE();
  }
}

void Riscv64Assembler::AddS(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FAddS(fd, fs, ft);
}

void Riscv64Assembler::SubS(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FSubS(fd, fs, ft);
}

void Riscv64Assembler::MulS(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FMulS(fd, fs, ft);
}

void Riscv64Assembler::DivS(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FDivS(fd, fs, ft);
}

void Riscv64Assembler::AddD(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FAddD(fd, fs, ft);
}

void Riscv64Assembler::SubD(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FSubD(fd, fs, ft);
}

void Riscv64Assembler::MulD(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FMulD(fd, fs, ft);
}

void Riscv64Assembler::DivD(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FDivD(fd, fs, ft);
}

void Riscv64Assembler::SqrtS(FpuRegister fd, FpuRegister fs) {
  FSqrtS(fd, fs);
}

void Riscv64Assembler::SqrtD(FpuRegister fd, FpuRegister fs) {
  FSqrtD(fd, fs);
}

void Riscv64Assembler::AbsS(FpuRegister fd, FpuRegister fs) {
  FSgnjxS(fd, fs, fs);
}

void Riscv64Assembler::AbsD(FpuRegister fd, FpuRegister fs) {
  FSgnjxD(fd, fs, fs);
}

void Riscv64Assembler::MovS(FpuRegister fd, FpuRegister fs) {
  FSgnjS(fd, fs, fs);
}

void Riscv64Assembler::MovD(FpuRegister fd, FpuRegister fs) {
  FSgnjD(fd, fs, fs);
}

void Riscv64Assembler::NegS(FpuRegister fd, FpuRegister fs) {
  FSgnjnS(fd, fs, fs);
}

void Riscv64Assembler::NegD(FpuRegister fd, FpuRegister fs) {
  FSgnjnD(fd, fs, fs);
}

void Riscv64Assembler::RoundLS(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x10, static_cast<FpuRegister>(0), fs, fd, 0x8);
}

void Riscv64Assembler::RoundLD(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x11, static_cast<FpuRegister>(0), fs, fd, 0x8);
}

void Riscv64Assembler::RoundWS(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x10, static_cast<FpuRegister>(0), fs, fd, 0xc);
}

void Riscv64Assembler::RoundWD(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x11, static_cast<FpuRegister>(0), fs, fd, 0xc);
}

void Riscv64Assembler::TruncLS(GpuRegister rd, FpuRegister fs) {
  Xor(rd, rd, rd);
  FEqS(TMP, fs, fs);
  riscv64::Riscv64Label label;
  Beqzc(TMP, &label);
  FCvtLS(rd, fs, kFPRoundingModeRTZ);
  Bind(&label);
}

void Riscv64Assembler::TruncLD(GpuRegister rd, FpuRegister fs) {
  Xor(rd, rd, rd);
  FEqD(TMP, fs, fs);
  riscv64::Riscv64Label label;
  Beqzc(TMP, &label);
  FCvtLD(rd, fs, kFPRoundingModeRTZ);
  Bind(&label);
}

void Riscv64Assembler::TruncWS(GpuRegister rd, FpuRegister fs) {
  Xor(rd, rd, rd);
  FEqS(TMP, fs, fs);
  riscv64::Riscv64Label label;
  Beqzc(TMP, &label);
  FCvtWS(rd, fs, kFPRoundingModeRTZ);
  Bind(&label);
}

void Riscv64Assembler::TruncWD(GpuRegister rd, FpuRegister fs) {
  Xor(rd, rd, rd);
  FEqD(TMP, fs, fs);
  riscv64::Riscv64Label label;
  Beqzc(TMP, &label);
  FCvtWD(rd, fs, kFPRoundingModeRTZ);
  Bind(&label);
}

void Riscv64Assembler::CeilLS(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x10, static_cast<FpuRegister>(0), fs, fd, 0xa);
}

void Riscv64Assembler::CeilLD(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x11, static_cast<FpuRegister>(0), fs, fd, 0xa);
}

void Riscv64Assembler::CeilWS(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x10, static_cast<FpuRegister>(0), fs, fd, 0xe);
}

void Riscv64Assembler::CeilWD(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x11, static_cast<FpuRegister>(0), fs, fd, 0xe);
}

void Riscv64Assembler::FloorLS(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x10, static_cast<FpuRegister>(0), fs, fd, 0xb);
}

void Riscv64Assembler::FloorLD(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x11, static_cast<FpuRegister>(0), fs, fd, 0xb);
}

void Riscv64Assembler::FloorWS(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x10, static_cast<FpuRegister>(0), fs, fd, 0xf);
}

void Riscv64Assembler::FloorWD(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x11, static_cast<FpuRegister>(0), fs, fd, 0xf);
}

void Riscv64Assembler::SelS(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FMvXW(TMP, fd);
  Andi(TMP, TMP, 1);
  Beq(TMP, ZERO, 12);
  FSgnjS(fd, ft, ft);
  Jal(ZERO, 8);

  FSgnjS(fd, fs, fs);
}

void Riscv64Assembler::SelD(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FMvXD(TMP, fd);
  Andi(TMP, TMP, 1);
  Beq(TMP, ZERO, 12);
  FSgnjD(fd, ft, ft);
  Jal(ZERO, 8);

  FSgnjD(fd, fs, fs);
}

void Riscv64Assembler::SeleqzS(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FMvXW(TMP, ft);
  Andi(TMP, TMP, 1);
  Beq(TMP, ZERO, 16);
  Addiw(TMP, ZERO, 0);
  FCvtSW(fd, TMP);
  Jal(ZERO, 8);

  FSgnjS(fd, fs, fs);
}

void Riscv64Assembler::SeleqzD(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FMvXD(TMP, ft);
  Andi(TMP, TMP, 1);
  Beq(TMP, ZERO, 16);
  Addi(TMP, ZERO, 0);
  FCvtDL(fd, TMP);
  Jal(ZERO, 8);

  FSgnjD(fd, fs, fs);
}

void Riscv64Assembler::SelnezS(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FMvXW(TMP, ft);
  Andi(TMP, TMP, 1);
  Bne(TMP, ZERO, 16);
  Addiw(TMP, ZERO, 0);
  FCvtSW(fd, TMP);
  Jal(ZERO, 8);

  FSgnjS(fd, fs, fs);
}

void Riscv64Assembler::SelnezD(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FMvXD(TMP, ft);
  Andi(TMP, TMP, 1);
  Bne(TMP, ZERO, 16);
  Addi(TMP, ZERO, 0);
  FCvtDL(fd, TMP);
  Jal(ZERO, 8);

  FSgnjD(fd, fs, fs);
}

void Riscv64Assembler::RintS(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x10, static_cast<FpuRegister>(0), fs, fd, 0x1a);
}

void Riscv64Assembler::RintD(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x11, static_cast<FpuRegister>(0), fs, fd, 0x1a);
}

void Riscv64Assembler::ClassS(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x10, static_cast<FpuRegister>(0), fs, fd, 0x1b);
}

void Riscv64Assembler::ClassD(FpuRegister fd, FpuRegister fs) {
  assert(0);
  EmitFR(0x11, 0x11, static_cast<FpuRegister>(0), fs, fd, 0x1b);
}

void Riscv64Assembler::MinS(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FMinS(fd, fs, ft);
}

void Riscv64Assembler::MinD(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FMinD(fd, fs, ft);
}

void Riscv64Assembler::MaxS(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FMaxS(fd, fs, ft);
}

void Riscv64Assembler::MaxD(FpuRegister fd, FpuRegister fs, FpuRegister ft) {
  FMaxD(fd, fs, ft);
}

void Riscv64Assembler::CmpUnS(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FClassS(TMP, fs);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 24);

  FClassS(TMP, ft);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 12);

  Addi(rd, ZERO, 0);  // unordered false;
  Jal(ZERO, 8);

  Addi(rd, ZERO, 1);  // unordered true;
}

void Riscv64Assembler::CmpEqS(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FEqS(rd, fs, ft);
}

void Riscv64Assembler::CmpUeqS(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FClassS(TMP, fs);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 24);

  FClassS(TMP, ft);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 12);

  FEqS(rd, fs, ft);
  Jal(ZERO, 8);

  Addi(rd, ZERO, 1);  // unordered true;
}

void Riscv64Assembler::CmpLtS(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FLtS(rd, fs, ft);
}

void Riscv64Assembler::CmpUltS(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FClassS(TMP, fs);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 24);

  FClassS(TMP, ft);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 12);

  FLtS(rd, fs, ft);
  Jal(ZERO, 8);

  Addi(rd, ZERO, 1);  // unordered true;
}

void Riscv64Assembler::CmpLeS(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FLeS(rd, fs, ft);
}

void Riscv64Assembler::CmpUleS(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FClassS(TMP, fs);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 24);

  FClassS(TMP, ft);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 12);

  FLeS(rd, fs, ft);
  Jal(ZERO, 8);

  Addi(rd, ZERO, 1);  // unordered true;
}

void Riscv64Assembler::CmpOrS(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  CmpUnS(rd, fs, ft);
  Sub(rd, ZERO, rd);
}

void Riscv64Assembler::CmpUneS(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FClassS(TMP, fs);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 28);

  FClassS(TMP, ft);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 16);

  FEqS(TMP, fs, ft);
  Sltiu(rd, TMP, 1);
  Jal(ZERO, 8);

  Addi(rd, ZERO, 1);  // unordered true;
}

void Riscv64Assembler::CmpNeS(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FEqS(rd, fs, ft);
  Sltiu(rd, rd, 1);
}

void Riscv64Assembler::CmpUnD(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FClassD(TMP, fs);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 24);

  FClassD(TMP, ft);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 12);

  Addi(rd, ZERO, 0);  // unordered false;
  Jal(ZERO, 8);

  Addi(rd, ZERO, 1);  // unordered true;
}

void Riscv64Assembler::CmpEqD(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FEqD(rd, fs, ft);
}

void Riscv64Assembler::CmpUeqD(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FClassD(TMP, fs);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 24);

  FClassD(TMP, ft);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 12);

  FEqD(rd, fs, ft);
  Jal(ZERO, 8);

  Addi(rd, ZERO, 1);  // unordered true;
}

void Riscv64Assembler::CmpLtD(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FLtD(rd, fs, ft);
}

void Riscv64Assembler::CmpUltD(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FClassD(TMP, fs);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 24);

  FClassD(TMP, ft);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 12);

  FLtD(rd, fs, ft);
  Jal(ZERO, 8);

  Addi(rd, ZERO, 1);  // unordered true;
}

void Riscv64Assembler::CmpLeD(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FLeD(rd, fs, ft);
}

void Riscv64Assembler::CmpUleD(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FClassD(TMP, fs);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 24);

  FClassD(TMP, ft);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 12);

  FLeD(rd, fs, ft);
  Jal(ZERO, 8);

  Addi(rd, ZERO, 1);  // unordered true;
}

void Riscv64Assembler::CmpOrD(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  CmpUnD(rd, fs, ft);
  Sltiu(rd, rd, 1);
}

void Riscv64Assembler::CmpUneD(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FClassD(TMP, fs);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 28);

  FClassD(TMP, ft);
  Srli(TMP, TMP, 8);
  Bne(TMP, ZERO, 16);

  FEqD(TMP, fs, ft);
  Sltiu(rd, rd, 1);
  Jal(ZERO, 8);

  Addi(rd, ZERO, 1);  // unordered true;
}

void Riscv64Assembler::CmpNeD(GpuRegister rd, FpuRegister fs, FpuRegister ft) {
  FEqD(rd, fs, ft);
  Sltiu(rd, rd, 1);
}

void Riscv64Assembler::Cvtsw(FpuRegister fd, FpuRegister fs) {
  assert(0);
}

void Riscv64Assembler::Cvtdw(FpuRegister fd, FpuRegister fs) {
  assert(0);
}

void Riscv64Assembler::Cvtsd(FpuRegister fd, FpuRegister fs) {
  FCvtSD(fd, fs);
}

void Riscv64Assembler::Cvtds(FpuRegister fd, FpuRegister fs) {
  FCvtDS(fd, fs);
}

void Riscv64Assembler::Cvtsl(FpuRegister fd, FpuRegister fs) {
  assert(0);
}

void Riscv64Assembler::Cvtdl(FpuRegister fd, FpuRegister fs) {
  assert(0);
}

void Riscv64Assembler::Mfc1(GpuRegister rt, FpuRegister fs) {
  // Move float32 in fs to rt
  FMvXW(rt, fs);
}

void Riscv64Assembler::Mfhc1(GpuRegister rt, FpuRegister fs) {
  FMvXD(rt, fs);
  Srli(rt, rt, 32);
}

void Riscv64Assembler::Mtc1(GpuRegister rt, FpuRegister fs) {
  // Move float32 in rt to fs
  FMvWX(fs, rt);
}

void Riscv64Assembler::Mthc1(GpuRegister rt, FpuRegister fs) {
  FMvXD(TMP, fs);
  Slli(TMP, TMP, 32);
  Srli(TMP, TMP, 32);
  Slli(rt, rt, 32);
  Or(rt, rt, TMP);
  FMvDX(fs, rt);
}

void Riscv64Assembler::Dmfc1(GpuRegister rt, FpuRegister fs) {
  // Move double in fs to rt
  FMvXD(rt, fs);
}

void Riscv64Assembler::Dmtc1(GpuRegister rt, FpuRegister fs) {
  // Move double in rt to fs
  FMvDX(fs, rt);
}

void Riscv64Assembler::Lwc1(FpuRegister ft, GpuRegister rs, uint16_t imm12) {
  FLw(ft, rs, imm12);  // warning: lw offset max is 12bits, not 16bits
}

void Riscv64Assembler::Ldc1(FpuRegister ft, GpuRegister rs, uint16_t imm12) {
  FLd(ft, rs, imm12);  // warning: lw offset max is 12bits, not 16bits
}

void Riscv64Assembler::Swc1(FpuRegister ft, GpuRegister rs, uint16_t imm12) {
  FSw(ft, rs, imm12);  // warning: lw offset max is 12bits, not 16bits
}

void Riscv64Assembler::Sdc1(FpuRegister ft, GpuRegister rs, uint16_t imm12) {
  FSd(ft, rs, imm12);  // warning: lw offset max is 12bits, not 16bits
}

void Riscv64Assembler::Break() {
  Ebreak();
}

void Riscv64Assembler::Nop() {
  Addi(ZERO, ZERO, 0);
}

void Riscv64Assembler::Move(GpuRegister rd, GpuRegister rs) {
  Or(rd, rs, ZERO);
}

void Riscv64Assembler::Clear(GpuRegister rd) {
  Move(rd, ZERO);
}

void Riscv64Assembler::Not(GpuRegister rd, GpuRegister rs) {
  Xori(rd, rs, -1);
}

void Riscv64Assembler::AndV(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x0, wt, ws, wd, 0x1e);
}

void Riscv64Assembler::OrV(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x1, wt, ws, wd, 0x1e);
}

void Riscv64Assembler::NorV(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x2, wt, ws, wd, 0x1e);
}

void Riscv64Assembler::XorV(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x3, wt, ws, wd, 0x1e);
}

void Riscv64Assembler::AddvB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x0, wt, ws, wd, 0xe);
}

void Riscv64Assembler::AddvH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x1, wt, ws, wd, 0xe);
}

void Riscv64Assembler::AddvW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x2, wt, ws, wd, 0xe);
}

void Riscv64Assembler::AddvD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x3, wt, ws, wd, 0xe);
}

void Riscv64Assembler::SubvB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x0, wt, ws, wd, 0xe);
}

void Riscv64Assembler::SubvH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x1, wt, ws, wd, 0xe);
}

void Riscv64Assembler::SubvW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x2, wt, ws, wd, 0xe);
}

void Riscv64Assembler::SubvD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x3, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Asub_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x0, wt, ws, wd, 0x11);
}

void Riscv64Assembler::Asub_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x1, wt, ws, wd, 0x11);
}

void Riscv64Assembler::Asub_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x2, wt, ws, wd, 0x11);
}

void Riscv64Assembler::Asub_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x3, wt, ws, wd, 0x11);
}

void Riscv64Assembler::Asub_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x0, wt, ws, wd, 0x11);
}

void Riscv64Assembler::Asub_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x1, wt, ws, wd, 0x11);
}

void Riscv64Assembler::Asub_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x2, wt, ws, wd, 0x11);
}

void Riscv64Assembler::Asub_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x3, wt, ws, wd, 0x11);
}

void Riscv64Assembler::MulvB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x0, wt, ws, wd, 0x12);
}

void Riscv64Assembler::MulvH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x1, wt, ws, wd, 0x12);
}

void Riscv64Assembler::MulvW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x2, wt, ws, wd, 0x12);
}

void Riscv64Assembler::MulvD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x3, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Div_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x0, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Div_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x1, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Div_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x2, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Div_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x3, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Div_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x0, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Div_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x1, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Div_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x2, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Div_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x3, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Mod_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x0, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Mod_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x1, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Mod_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x2, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Mod_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x3, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Mod_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x0, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Mod_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x1, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Mod_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x2, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Mod_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x3, wt, ws, wd, 0x12);
}

void Riscv64Assembler::Add_aB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x0, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Add_aH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x1, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Add_aW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x2, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Add_aD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x3, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Ave_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x0, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Ave_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x1, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Ave_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x2, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Ave_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x3, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Ave_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x0, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Ave_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x1, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Ave_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x2, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Ave_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x3, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Aver_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x0, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Aver_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x1, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Aver_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x2, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Aver_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x3, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Aver_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x0, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Aver_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x1, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Aver_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x2, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Aver_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x3, wt, ws, wd, 0x10);
}

void Riscv64Assembler::Max_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x0, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Max_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x1, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Max_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x2, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Max_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x3, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Max_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x3, 0x0, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Max_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x3, 0x1, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Max_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x3, 0x2, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Max_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x3, 0x3, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Min_sB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x0, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Min_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x1, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Min_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x2, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Min_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x3, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Min_uB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x0, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Min_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x1, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Min_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x2, wt, ws, wd, 0xe);
}

void Riscv64Assembler::Min_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x3, wt, ws, wd, 0xe);
}

void Riscv64Assembler::FaddW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x0, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FaddD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x1, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FsubW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x2, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FsubD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x3, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FmulW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x0, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FmulD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x1, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FdivW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x2, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FdivD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x3, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FmaxW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x0, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FmaxD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x1, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FminW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x0, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FminD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x1, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::Ffint_sW(VectorRegister wd, VectorRegister ws) {
  CHECK(HasMsa());
  EmitMsa2RF(0x19e, 0x0, ws, wd, 0x1e);
}

void Riscv64Assembler::Ffint_sD(VectorRegister wd, VectorRegister ws) {
  CHECK(HasMsa());
  EmitMsa2RF(0x19e, 0x1, ws, wd, 0x1e);
}

void Riscv64Assembler::Ftint_sW(VectorRegister wd, VectorRegister ws) {
  CHECK(HasMsa());
  EmitMsa2RF(0x19c, 0x0, ws, wd, 0x1e);
}

void Riscv64Assembler::Ftint_sD(VectorRegister wd, VectorRegister ws) {
  CHECK(HasMsa());
  EmitMsa2RF(0x19c, 0x1, ws, wd, 0x1e);
}

void Riscv64Assembler::SllB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x0, wt, ws, wd, 0xd);
}

void Riscv64Assembler::SllH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x1, wt, ws, wd, 0xd);
}

void Riscv64Assembler::SllW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x2, wt, ws, wd, 0xd);
}

void Riscv64Assembler::SllD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x0, 0x3, wt, ws, wd, 0xd);
}

void Riscv64Assembler::SraB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x0, wt, ws, wd, 0xd);
}

void Riscv64Assembler::SraH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x1, wt, ws, wd, 0xd);
}

void Riscv64Assembler::SraW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x2, wt, ws, wd, 0xd);
}

void Riscv64Assembler::SraD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x3, wt, ws, wd, 0xd);
}

void Riscv64Assembler::SrlB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x0, wt, ws, wd, 0xd);
}

void Riscv64Assembler::SrlH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x1, wt, ws, wd, 0xd);
}

void Riscv64Assembler::SrlW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x2, wt, ws, wd, 0xd);
}

void Riscv64Assembler::SrlD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x3, wt, ws, wd, 0xd);
}

void Riscv64Assembler::SlliB(VectorRegister wd, VectorRegister ws, int shamt3) {
  CHECK(HasMsa());
  CHECK(IsUint<3>(shamt3)) << shamt3;
  EmitMsaBIT(0x0, shamt3 | kMsaDfMByteMask, ws, wd, 0x9);
}

void Riscv64Assembler::SlliH(VectorRegister wd, VectorRegister ws, int shamt4) {
  CHECK(HasMsa());
  CHECK(IsUint<4>(shamt4)) << shamt4;
  EmitMsaBIT(0x0, shamt4 | kMsaDfMHalfwordMask, ws, wd, 0x9);
}

void Riscv64Assembler::SlliW(VectorRegister wd, VectorRegister ws, int shamt5) {
  CHECK(HasMsa());
  CHECK(IsUint<5>(shamt5)) << shamt5;
  EmitMsaBIT(0x0, shamt5 | kMsaDfMWordMask, ws, wd, 0x9);
}

void Riscv64Assembler::SlliD(VectorRegister wd, VectorRegister ws, int shamt6) {
  CHECK(HasMsa());
  CHECK(IsUint<6>(shamt6)) << shamt6;
  EmitMsaBIT(0x0, shamt6 | kMsaDfMDoublewordMask, ws, wd, 0x9);
}

void Riscv64Assembler::SraiB(VectorRegister wd, VectorRegister ws, int shamt3) {
  CHECK(HasMsa());
  CHECK(IsUint<3>(shamt3)) << shamt3;
  EmitMsaBIT(0x1, shamt3 | kMsaDfMByteMask, ws, wd, 0x9);
}

void Riscv64Assembler::SraiH(VectorRegister wd, VectorRegister ws, int shamt4) {
  CHECK(HasMsa());
  CHECK(IsUint<4>(shamt4)) << shamt4;
  EmitMsaBIT(0x1, shamt4 | kMsaDfMHalfwordMask, ws, wd, 0x9);
}

void Riscv64Assembler::SraiW(VectorRegister wd, VectorRegister ws, int shamt5) {
  CHECK(HasMsa());
  CHECK(IsUint<5>(shamt5)) << shamt5;
  EmitMsaBIT(0x1, shamt5 | kMsaDfMWordMask, ws, wd, 0x9);
}

void Riscv64Assembler::SraiD(VectorRegister wd, VectorRegister ws, int shamt6) {
  CHECK(HasMsa());
  CHECK(IsUint<6>(shamt6)) << shamt6;
  EmitMsaBIT(0x1, shamt6 | kMsaDfMDoublewordMask, ws, wd, 0x9);
}

void Riscv64Assembler::SrliB(VectorRegister wd, VectorRegister ws, int shamt3) {
  CHECK(HasMsa());
  CHECK(IsUint<3>(shamt3)) << shamt3;
  EmitMsaBIT(0x2, shamt3 | kMsaDfMByteMask, ws, wd, 0x9);
}

void Riscv64Assembler::SrliH(VectorRegister wd, VectorRegister ws, int shamt4) {
  CHECK(HasMsa());
  CHECK(IsUint<4>(shamt4)) << shamt4;
  EmitMsaBIT(0x2, shamt4 | kMsaDfMHalfwordMask, ws, wd, 0x9);
}

void Riscv64Assembler::SrliW(VectorRegister wd, VectorRegister ws, int shamt5) {
  CHECK(HasMsa());
  CHECK(IsUint<5>(shamt5)) << shamt5;
  EmitMsaBIT(0x2, shamt5 | kMsaDfMWordMask, ws, wd, 0x9);
}

void Riscv64Assembler::SrliD(VectorRegister wd, VectorRegister ws, int shamt6) {
  CHECK(HasMsa());
  CHECK(IsUint<6>(shamt6)) << shamt6;
  EmitMsaBIT(0x2, shamt6 | kMsaDfMDoublewordMask, ws, wd, 0x9);
}

void Riscv64Assembler::MoveV(VectorRegister wd, VectorRegister ws) {
  CHECK(HasMsa());
  EmitMsaBIT(0x1, 0x3e, ws, wd, 0x19);
}

void Riscv64Assembler::SplatiB(VectorRegister wd, VectorRegister ws, int n4) {
  CHECK(HasMsa());
  CHECK(IsUint<4>(n4)) << n4;
  EmitMsaELM(0x1, n4 | kMsaDfNByteMask, ws, wd, 0x19);
}

void Riscv64Assembler::SplatiH(VectorRegister wd, VectorRegister ws, int n3) {
  CHECK(HasMsa());
  CHECK(IsUint<3>(n3)) << n3;
  EmitMsaELM(0x1, n3 | kMsaDfNHalfwordMask, ws, wd, 0x19);
}

void Riscv64Assembler::SplatiW(VectorRegister wd, VectorRegister ws, int n2) {
  CHECK(HasMsa());
  CHECK(IsUint<2>(n2)) << n2;
  EmitMsaELM(0x1, n2 | kMsaDfNWordMask, ws, wd, 0x19);
}

void Riscv64Assembler::SplatiD(VectorRegister wd, VectorRegister ws, int n1) {
  CHECK(HasMsa());
  CHECK(IsUint<1>(n1)) << n1;
  EmitMsaELM(0x1, n1 | kMsaDfNDoublewordMask, ws, wd, 0x19);
}

void Riscv64Assembler::Copy_sB(GpuRegister rd, VectorRegister ws, int n4) {
  CHECK(HasMsa());
  CHECK(IsUint<4>(n4)) << n4;
  EmitMsaELM(0x2, n4 | kMsaDfNByteMask, ws, static_cast<VectorRegister>(rd), 0x19);
}

void Riscv64Assembler::Copy_sH(GpuRegister rd, VectorRegister ws, int n3) {
  CHECK(HasMsa());
  CHECK(IsUint<3>(n3)) << n3;
  EmitMsaELM(0x2, n3 | kMsaDfNHalfwordMask, ws, static_cast<VectorRegister>(rd), 0x19);
}

void Riscv64Assembler::Copy_sW(GpuRegister rd, VectorRegister ws, int n2) {
  CHECK(HasMsa());
  CHECK(IsUint<2>(n2)) << n2;
  EmitMsaELM(0x2, n2 | kMsaDfNWordMask, ws, static_cast<VectorRegister>(rd), 0x19);
}

void Riscv64Assembler::Copy_sD(GpuRegister rd, VectorRegister ws, int n1) {
  CHECK(HasMsa());
  CHECK(IsUint<1>(n1)) << n1;
  EmitMsaELM(0x2, n1 | kMsaDfNDoublewordMask, ws, static_cast<VectorRegister>(rd), 0x19);
}

void Riscv64Assembler::Copy_uB(GpuRegister rd, VectorRegister ws, int n4) {
  CHECK(HasMsa());
  CHECK(IsUint<4>(n4)) << n4;
  EmitMsaELM(0x3, n4 | kMsaDfNByteMask, ws, static_cast<VectorRegister>(rd), 0x19);
}

void Riscv64Assembler::Copy_uH(GpuRegister rd, VectorRegister ws, int n3) {
  CHECK(HasMsa());
  CHECK(IsUint<3>(n3)) << n3;
  EmitMsaELM(0x3, n3 | kMsaDfNHalfwordMask, ws, static_cast<VectorRegister>(rd), 0x19);
}

void Riscv64Assembler::Copy_uW(GpuRegister rd, VectorRegister ws, int n2) {
  CHECK(HasMsa());
  CHECK(IsUint<2>(n2)) << n2;
  EmitMsaELM(0x3, n2 | kMsaDfNWordMask, ws, static_cast<VectorRegister>(rd), 0x19);
}

void Riscv64Assembler::InsertB(VectorRegister wd, GpuRegister rs, int n4) {
  CHECK(HasMsa());
  CHECK(IsUint<4>(n4)) << n4;
  EmitMsaELM(0x4, n4 | kMsaDfNByteMask, static_cast<VectorRegister>(rs), wd, 0x19);
}

void Riscv64Assembler::InsertH(VectorRegister wd, GpuRegister rs, int n3) {
  CHECK(HasMsa());
  CHECK(IsUint<3>(n3)) << n3;
  EmitMsaELM(0x4, n3 | kMsaDfNHalfwordMask, static_cast<VectorRegister>(rs), wd, 0x19);
}

void Riscv64Assembler::InsertW(VectorRegister wd, GpuRegister rs, int n2) {
  CHECK(HasMsa());
  CHECK(IsUint<2>(n2)) << n2;
  EmitMsaELM(0x4, n2 | kMsaDfNWordMask, static_cast<VectorRegister>(rs), wd, 0x19);
}

void Riscv64Assembler::InsertD(VectorRegister wd, GpuRegister rs, int n1) {
  CHECK(HasMsa());
  CHECK(IsUint<1>(n1)) << n1;
  EmitMsaELM(0x4, n1 | kMsaDfNDoublewordMask, static_cast<VectorRegister>(rs), wd, 0x19);
}

void Riscv64Assembler::FillB(VectorRegister wd, GpuRegister rs) {
  CHECK(HasMsa());
  EmitMsa2R(0xc0, 0x0, static_cast<VectorRegister>(rs), wd, 0x1e);
}

void Riscv64Assembler::FillH(VectorRegister wd, GpuRegister rs) {
  CHECK(HasMsa());
  EmitMsa2R(0xc0, 0x1, static_cast<VectorRegister>(rs), wd, 0x1e);
}

void Riscv64Assembler::FillW(VectorRegister wd, GpuRegister rs) {
  CHECK(HasMsa());
  EmitMsa2R(0xc0, 0x2, static_cast<VectorRegister>(rs), wd, 0x1e);
}

void Riscv64Assembler::FillD(VectorRegister wd, GpuRegister rs) {
  CHECK(HasMsa());
  EmitMsa2R(0xc0, 0x3, static_cast<VectorRegister>(rs), wd, 0x1e);
}

void Riscv64Assembler::LdiB(VectorRegister wd, int imm8) {
  CHECK(HasMsa());
  CHECK(IsInt<8>(imm8)) << imm8;
  EmitMsaI10(0x6, 0x0, imm8 & kMsaS10Mask, wd, 0x7);
}

void Riscv64Assembler::LdiH(VectorRegister wd, int imm10) {
  CHECK(HasMsa());
  CHECK(IsInt<10>(imm10)) << imm10;
  EmitMsaI10(0x6, 0x1, imm10 & kMsaS10Mask, wd, 0x7);
}

void Riscv64Assembler::LdiW(VectorRegister wd, int imm10) {
  CHECK(HasMsa());
  CHECK(IsInt<10>(imm10)) << imm10;
  EmitMsaI10(0x6, 0x2, imm10 & kMsaS10Mask, wd, 0x7);
}

void Riscv64Assembler::LdiD(VectorRegister wd, int imm10) {
  CHECK(HasMsa());
  CHECK(IsInt<10>(imm10)) << imm10;
  EmitMsaI10(0x6, 0x3, imm10 & kMsaS10Mask, wd, 0x7);
}

void Riscv64Assembler::LdB(VectorRegister wd, GpuRegister rs, int offset) {
  CHECK(HasMsa());
  CHECK(IsInt<10>(offset)) << offset;
  EmitMsaMI10(offset & kMsaS10Mask, rs, wd, 0x8, 0x0);
}

void Riscv64Assembler::LdH(VectorRegister wd, GpuRegister rs, int offset) {
  CHECK(HasMsa());
  CHECK(IsInt<11>(offset)) << offset;
  CHECK_ALIGNED(offset, kRiscv64HalfwordSize);
  EmitMsaMI10((offset >> TIMES_2) & kMsaS10Mask, rs, wd, 0x8, 0x1);
}

void Riscv64Assembler::LdW(VectorRegister wd, GpuRegister rs, int offset) {
  CHECK(HasMsa());
  CHECK(IsInt<12>(offset)) << offset;
  CHECK_ALIGNED(offset, kRiscv64WordSize);
  EmitMsaMI10((offset >> TIMES_4) & kMsaS10Mask, rs, wd, 0x8, 0x2);
}

void Riscv64Assembler::LdD(VectorRegister wd, GpuRegister rs, int offset) {
  CHECK(HasMsa());
  CHECK(IsInt<13>(offset)) << offset;
  CHECK_ALIGNED(offset, kRiscv64DoublewordSize);
  EmitMsaMI10((offset >> TIMES_8) & kMsaS10Mask, rs, wd, 0x8, 0x3);
}

void Riscv64Assembler::StB(VectorRegister wd, GpuRegister rs, int offset) {
  CHECK(HasMsa());
  CHECK(IsInt<10>(offset)) << offset;
  EmitMsaMI10(offset & kMsaS10Mask, rs, wd, 0x9, 0x0);
}

void Riscv64Assembler::StH(VectorRegister wd, GpuRegister rs, int offset) {
  CHECK(HasMsa());
  CHECK(IsInt<11>(offset)) << offset;
  CHECK_ALIGNED(offset, kRiscv64HalfwordSize);
  EmitMsaMI10((offset >> TIMES_2) & kMsaS10Mask, rs, wd, 0x9, 0x1);
}

void Riscv64Assembler::StW(VectorRegister wd, GpuRegister rs, int offset) {
  CHECK(HasMsa());
  CHECK(IsInt<12>(offset)) << offset;
  CHECK_ALIGNED(offset, kRiscv64WordSize);
  EmitMsaMI10((offset >> TIMES_4) & kMsaS10Mask, rs, wd, 0x9, 0x2);
}

void Riscv64Assembler::StD(VectorRegister wd, GpuRegister rs, int offset) {
  CHECK(HasMsa());
  CHECK(IsInt<13>(offset)) << offset;
  CHECK_ALIGNED(offset, kRiscv64DoublewordSize);
  EmitMsaMI10((offset >> TIMES_8) & kMsaS10Mask, rs, wd, 0x9, 0x3);
}

void Riscv64Assembler::IlvlB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x0, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvlH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x1, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvlW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x2, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvlD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x3, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvrB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x0, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvrH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x1, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvrW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x2, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvrD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x3, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvevB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x0, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvevH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x1, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvevW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x2, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvevD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x6, 0x3, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvodB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x0, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvodH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x1, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvodW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x2, wt, ws, wd, 0x14);
}

void Riscv64Assembler::IlvodD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x7, 0x3, wt, ws, wd, 0x14);
}

void Riscv64Assembler::MaddvB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x0, wt, ws, wd, 0x12);
}

void Riscv64Assembler::MaddvH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x1, wt, ws, wd, 0x12);
}

void Riscv64Assembler::MaddvW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x2, wt, ws, wd, 0x12);
}

void Riscv64Assembler::MaddvD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x1, 0x3, wt, ws, wd, 0x12);
}

void Riscv64Assembler::MsubvB(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x0, wt, ws, wd, 0x12);
}

void Riscv64Assembler::MsubvH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x1, wt, ws, wd, 0x12);
}

void Riscv64Assembler::MsubvW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x2, wt, ws, wd, 0x12);
}

void Riscv64Assembler::MsubvD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x3, wt, ws, wd, 0x12);
}

void Riscv64Assembler::FmaddW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x0, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FmaddD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x1, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FmsubW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x2, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::FmsubD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x2, 0x3, wt, ws, wd, 0x1b);
}

void Riscv64Assembler::Hadd_sH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x1, wt, ws, wd, 0x15);
}

void Riscv64Assembler::Hadd_sW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x2, wt, ws, wd, 0x15);
}

void Riscv64Assembler::Hadd_sD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x4, 0x3, wt, ws, wd, 0x15);
}

void Riscv64Assembler::Hadd_uH(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x1, wt, ws, wd, 0x15);
}

void Riscv64Assembler::Hadd_uW(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x2, wt, ws, wd, 0x15);
}

void Riscv64Assembler::Hadd_uD(VectorRegister wd, VectorRegister ws, VectorRegister wt) {
  CHECK(HasMsa());
  EmitMsa3R(0x5, 0x3, wt, ws, wd, 0x15);
}

void Riscv64Assembler::PcntB(VectorRegister wd, VectorRegister ws) {
  CHECK(HasMsa());
  EmitMsa2R(0xc1, 0x0, ws, wd, 0x1e);
}

void Riscv64Assembler::PcntH(VectorRegister wd, VectorRegister ws) {
  CHECK(HasMsa());
  EmitMsa2R(0xc1, 0x1, ws, wd, 0x1e);
}

void Riscv64Assembler::PcntW(VectorRegister wd, VectorRegister ws) {
  CHECK(HasMsa());
  EmitMsa2R(0xc1, 0x2, ws, wd, 0x1e);
}

void Riscv64Assembler::PcntD(VectorRegister wd, VectorRegister ws) {
  CHECK(HasMsa());
  EmitMsa2R(0xc1, 0x3, ws, wd, 0x1e);
}

void Riscv64Assembler::ReplicateFPToVectorRegister(VectorRegister dst,
                                                  FpuRegister src,
                                                  bool is_double) {
  // Float or double in FPU register Fx can be considered as 0th element in vector register Wx.
  if (is_double) {
    SplatiD(dst, static_cast<VectorRegister>(src), 0);
  } else {
    SplatiW(dst, static_cast<VectorRegister>(src), 0);
  }
}

void Riscv64Assembler::LoadConst32(GpuRegister rd, int32_t value) {
  // Use 11-bit check here for avoiding sign-extension.
  if (IsInt<11>(value)) {
    Addiw(rd, ZERO, value);
  } else {
    int32_t l = value & 0xFFF;
    int32_t h = value >> 12;
    if ((l & 0x800) != 0) {
      h += 1;  // overflow ?
    }
    Lui(rd, h);
    Addiw(rd, rd, l);
  }
}

// This function is only used for testing purposes.
void Riscv64Assembler::RecordLoadConst64Path(int value ATTRIBUTE_UNUSED) {
}

void Riscv64Assembler::LoadConst64(GpuRegister rd, int64_t value) {
  // TemplateLoadConst64(this, rd, value);
  if (IsInt<32>(value)) {
    LoadConst32(rd, value);
  } else {
    // Need to optimize in the future.
    int32_t hi = value >> 32;
    int32_t lo = value;

    GpuRegister scratch = TMP2;

    LoadConst32(scratch, lo);
    LoadConst32(rd, hi);
    Slli(rd, rd, 32);
    Slli(scratch, scratch, 32);
    Srli(scratch, scratch, 32);
    Or(rd, rd, scratch);
  }
}

void Riscv64Assembler::Addiu32(GpuRegister rt, GpuRegister rs, int32_t value) {
  if (IsInt<12>(value)) {
    Addiw(rt, rs, value);
  } else {
    LoadConst32(TMP2, value);
    Addw(rt, rs, TMP2);
  }
}

// TODO: don't use rtmp, use daui, dahi, dati.
void Riscv64Assembler::Daddiu64(GpuRegister rt, GpuRegister rs, int64_t value, GpuRegister rtmp) {
  CHECK_NE(rs, rtmp);
  if (IsInt<12>(value)) {
    Addi(rt, rs, value);
  } else {
    LoadConst64(rtmp, value);
    Add(rt, rs, rtmp);
  }
}

void Riscv64Assembler::Branch::InitShortOrLong(Riscv64Assembler::Branch::OffsetBits offset_size,
                                              Riscv64Assembler::Branch::Type short_type,
                                              Riscv64Assembler::Branch::Type long_type) {
  type_ = (offset_size <= branch_info_[short_type].offset_size) ? short_type : long_type;
}

void Riscv64Assembler::Branch::InitializeType(Type initial_type) {
  OffsetBits offset_size_needed = GetOffsetSizeNeeded(location_, target_);

    switch (initial_type) {
      case kLabel:
      case kLiteral:
      case kLiteralUnsigned:
      case kLiteralLong:
        CHECK(!IsResolved());
        type_ = initial_type;
        break;
      case kCall:
        InitShortOrLong(offset_size_needed, kCall, kLongCall);
        break;
      case kCondBranch:
        switch (condition_) {
          case kUncond:
            InitShortOrLong(offset_size_needed, kUncondBranch, kLongUncondBranch);
            break;
          default:
            InitShortOrLong(offset_size_needed, kCondBranch, kLongCondBranch);
            break;
        }
        break;
      case kBareCall:
        type_ = kBareCall;
        CHECK_LE(offset_size_needed, GetOffsetSize());
        break;
      case kBareCondBranch:
        type_ = (condition_ == kUncond) ? kBareUncondBranch : kBareCondBranch;
        CHECK_LE(offset_size_needed, GetOffsetSize());
        break;
      default:
        LOG(FATAL) << "Unexpected branch type " << initial_type;
        UNREACHABLE();
    }

  old_type_ = type_;
}

bool Riscv64Assembler::Branch::IsNop(BranchCondition condition, GpuRegister lhs, GpuRegister rhs) {
  switch (condition) {
    case kCondLT:
    case kCondGT:
    case kCondNE:
    case kCondLTU:
      return lhs == rhs;
    default:
      return false;
  }
}

bool Riscv64Assembler::Branch::IsUncond(BranchCondition condition,
                                       GpuRegister lhs,
                                       GpuRegister rhs) {
  switch (condition) {
    case kUncond:
      return true;
    case kCondGE:
    case kCondLE:
    case kCondEQ:
    case kCondGEU:
      return lhs == rhs;
    default:
      return false;
  }
}

Riscv64Assembler::Branch::Branch(uint32_t location, uint32_t target, bool is_call, bool is_bare)
    : old_location_(location),
      location_(location),
      target_(target),
      lhs_reg_(ZERO),
      rhs_reg_(ZERO),
      condition_(kUncond) {
  InitializeType(
      (is_call ? (is_bare ? kBareCall : kCall) : (is_bare ? kBareCondBranch : kCondBranch)));
}

Riscv64Assembler::Branch::Branch(uint32_t location,
                                uint32_t target,
                                Riscv64Assembler::BranchCondition condition,
                                GpuRegister lhs_reg,
                                GpuRegister rhs_reg,
                                bool is_bare)
    : old_location_(location),
      location_(location),
      target_(target),
      lhs_reg_(lhs_reg),
      rhs_reg_(rhs_reg),
      condition_(condition) {
  // FIXME: T-HEAD
  // CHECK_NE(condition, kUncond);
  switch (condition) {
    case kCondEQ:
    case kCondNE:
    case kCondLT:
    case kCondGE:
    case kCondLE:
    case kCondGT:
    case kCondLTU:
    case kCondGEU:
      CHECK_NE(lhs_reg, ZERO);
      CHECK_NE(rhs_reg, ZERO);
      break;
    case kCondLTZ:
    case kCondGEZ:
    case kCondLEZ:
    case kCondGTZ:
    case kCondEQZ:
    case kCondNEZ:
      CHECK_NE(lhs_reg, ZERO);
      CHECK_EQ(rhs_reg, ZERO);
      break;
    case kUncond:
      UNREACHABLE();
  }
  CHECK(!IsNop(condition, lhs_reg, rhs_reg));
  if (IsUncond(condition, lhs_reg, rhs_reg)) {
    // Branch condition is always true, make the branch unconditional.
    condition_ = kUncond;
  }
  InitializeType((is_bare ? kBareCondBranch : kCondBranch));
}

Riscv64Assembler::Branch::Branch(uint32_t location, GpuRegister dest_reg, Type label_or_literal_type)
    : old_location_(location),
      location_(location),
      target_(kUnresolved),
      lhs_reg_(dest_reg),
      rhs_reg_(ZERO),
      condition_(kUncond) {
  CHECK_NE(dest_reg, ZERO);
  InitializeType(label_or_literal_type);
}

Riscv64Assembler::BranchCondition Riscv64Assembler::Branch::OppositeCondition(
    Riscv64Assembler::BranchCondition cond) {
  switch (cond) {
    case kCondLT:
      return kCondGE;
    case kCondGE:
      return kCondLT;
    case kCondLE:
      return kCondGT;
    case kCondGT:
      return kCondLE;
    case kCondLTZ:
      return kCondGEZ;
    case kCondGEZ:
      return kCondLTZ;
    case kCondLEZ:
      return kCondGTZ;
    case kCondGTZ:
      return kCondLEZ;
    case kCondEQ:
      return kCondNE;
    case kCondNE:
      return kCondEQ;
    case kCondEQZ:
      return kCondNEZ;
    case kCondNEZ:
      return kCondEQZ;
    case kCondLTU:
      return kCondGEU;
    case kCondGEU:
      return kCondLTU;
    case kUncond:
      // LOG(FATAL) << "Unexpected branch condition " << cond;
      LOG(FATAL) << "Unexpected branch condition ";
  }
  UNREACHABLE();
}

Riscv64Assembler::Branch::Type Riscv64Assembler::Branch::GetType() const {
  return type_;
}

Riscv64Assembler::BranchCondition Riscv64Assembler::Branch::GetCondition() const {
  return condition_;
}

GpuRegister Riscv64Assembler::Branch::GetLeftRegister() const {
  return lhs_reg_;
}

GpuRegister Riscv64Assembler::Branch::GetRightRegister() const {
  return rhs_reg_;
}

uint32_t Riscv64Assembler::Branch::GetTarget() const {
  return target_;
}

uint32_t Riscv64Assembler::Branch::GetLocation() const {
  return location_;
}

uint32_t Riscv64Assembler::Branch::GetOldLocation() const {
  return old_location_;
}

uint32_t Riscv64Assembler::Branch::GetLength() const {
  return branch_info_[type_].length;
}

uint32_t Riscv64Assembler::Branch::GetOldLength() const {
  return branch_info_[old_type_].length;
}

uint32_t Riscv64Assembler::Branch::GetSize() const {
  return GetLength() * sizeof(uint32_t);
}

uint32_t Riscv64Assembler::Branch::GetOldSize() const {
  return GetOldLength() * sizeof(uint32_t);
}

uint32_t Riscv64Assembler::Branch::GetEndLocation() const {
  return GetLocation() + GetSize();
}

uint32_t Riscv64Assembler::Branch::GetOldEndLocation() const {
  return GetOldLocation() + GetOldSize();
}

bool Riscv64Assembler::Branch::IsBare() const {
  switch (type_) {
    // R6 short branches (can't be promoted to long), forbidden/delay slots filled manually.
    case kBareUncondBranch:
    case kBareCondBranch:
    case kBareCall:
      return true;
    default:
      return false;
  }
}

bool Riscv64Assembler::Branch::IsLong() const {
  switch (type_) {
    // R6 short branches (can be promoted to long).
    case kUncondBranch:
    case kCondBranch:
    case kCall:
    // R6 short branches (can't be promoted to long), forbidden/delay slots filled manually.
    case kBareUncondBranch:
    case kBareCondBranch:
    case kBareCall:
      return false;
    // Long branches.
    case kLongUncondBranch:
    case kLongCondBranch:
    case kLongCall:
    // label.
    case kLabel:
    // literals.
    case kLiteral:
    case kLiteralUnsigned:
    case kLiteralLong:
      return true;
  }
  UNREACHABLE();
}

bool Riscv64Assembler::Branch::IsResolved() const {
  return target_ != kUnresolved;
}

Riscv64Assembler::Branch::OffsetBits Riscv64Assembler::Branch::GetOffsetSize() const {
  return branch_info_[type_].offset_size;
}

Riscv64Assembler::Branch::OffsetBits Riscv64Assembler::Branch::GetOffsetSizeNeeded(uint32_t location,
                                                                                 uint32_t target) {
  // For unresolved targets assume the shortest encoding
  // (later it will be made longer if needed).
  if (target == kUnresolved)
    return kOffset13;
  int64_t distance = static_cast<int64_t>(target) - location;
  // To simplify calculations in composite branches consisting of multiple instructions
  // bump up the distance by a value larger than the max byte size of a composite branch.
  distance += (distance >= 0) ? kMaxBranchSize : -kMaxBranchSize;
  if (IsInt<kOffset13>(distance))
    return kOffset13;
  else if (IsInt<kOffset21>(distance))
    return kOffset21;
  return kOffset32;
}

void Riscv64Assembler::Branch::Resolve(uint32_t target) {
  target_ = target;
}

void Riscv64Assembler::Branch::Relocate(uint32_t expand_location, uint32_t delta) {
  if (location_ > expand_location) {
    location_ += delta;
  }
  if (!IsResolved()) {
    return;  // Don't know the target yet.
  }
  if (target_ > expand_location) {
    target_ += delta;
  }
}

void Riscv64Assembler::Branch::PromoteToLong() {
  CHECK(!IsBare());  // Bare branches do not promote.
  switch (type_) {
    // R6 short branches (can be promoted to long).
    case kUncondBranch:
      type_ = kLongUncondBranch;
      break;
    case kCondBranch:
      type_ = kLongCondBranch;
      break;
    case kCall:
      type_ = kLongCall;
      break;
    default:
      // Note: 'type_' is already long.
      break;
  }
  CHECK(IsLong());
}

uint32_t Riscv64Assembler::Branch::PromoteIfNeeded(uint32_t max_short_distance) {
  // If the branch is still unresolved or already long, nothing to do.
  if (IsLong() || !IsResolved()) {
    return 0;
  }
  // Promote the short branch to long if the offset size is too small
  // to hold the distance between location_ and target_.
  if (GetOffsetSizeNeeded(location_, target_) > GetOffsetSize()) {
    PromoteToLong();
    uint32_t old_size = GetOldSize();
    uint32_t new_size = GetSize();
    CHECK_GT(new_size, old_size);
    return new_size - old_size;
  }
  // The following logic is for debugging/testing purposes.
  // Promote some short branches to long when it's not really required.
  if (UNLIKELY(max_short_distance != std::numeric_limits<uint32_t>::max() && !IsBare())) {
    int64_t distance = static_cast<int64_t>(target_) - location_;
    distance = (distance >= 0) ? distance : -distance;
    if (distance >= max_short_distance) {
      PromoteToLong();
      uint32_t old_size = GetOldSize();
      uint32_t new_size = GetSize();
      CHECK_GT(new_size, old_size);
      return new_size - old_size;
    }
  }
  return 0;
}

uint32_t Riscv64Assembler::Branch::GetOffsetLocation() const {
  return location_ + branch_info_[type_].instr_offset * sizeof(uint32_t);
}

uint32_t Riscv64Assembler::Branch::GetOffset() const {
  CHECK(IsResolved());
  uint32_t ofs_mask = 0xFFFFFFFF >> (32 - GetOffsetSize());
  // Calculate the byte distance between instructions and also account for
  // different PC-relative origins.
  uint32_t offset_location = GetOffsetLocation();
  uint32_t offset = target_ - offset_location - branch_info_[type_].pc_org * sizeof(uint32_t);
  // Prepare the offset for encoding into the instruction(s).
  offset = (offset & ofs_mask) >> branch_info_[type_].offset_shift;
  return offset;
}

Riscv64Assembler::Branch* Riscv64Assembler::GetBranch(uint32_t branch_id) {
  CHECK_LT(branch_id, branches_.size());
  return &branches_[branch_id];
}

const Riscv64Assembler::Branch* Riscv64Assembler::GetBranch(uint32_t branch_id) const {
  CHECK_LT(branch_id, branches_.size());
  return &branches_[branch_id];
}

void Riscv64Assembler::Bind(Riscv64Label* label) {
  CHECK(!label->IsBound());
  uint32_t bound_pc = buffer_.Size();

  // Walk the list of branches referring to and preceding this label.
  // Store the previously unknown target addresses in them.
  while (label->IsLinked()) {
    uint32_t branch_id = label->Position();
    Branch* branch = GetBranch(branch_id);
    branch->Resolve(bound_pc);

    uint32_t branch_location = branch->GetLocation();
    // Extract the location of the previous branch in the list (walking the list backwards;
    // the previous branch ID was stored in the space reserved for this branch).
    uint32_t prev = buffer_.Load<uint32_t>(branch_location);

    // On to the previous branch in the list...
    label->position_ = prev;
  }

  // Now make the label object contain its own location (relative to the end of the preceding
  // branch, if any; it will be used by the branches referring to and following this label).
  label->prev_branch_id_plus_one_ = branches_.size();
  if (label->prev_branch_id_plus_one_) {
    uint32_t branch_id = label->prev_branch_id_plus_one_ - 1;
    const Branch* branch = GetBranch(branch_id);
    bound_pc -= branch->GetEndLocation();
  }
  label->BindTo(bound_pc);
}

uint32_t Riscv64Assembler::GetLabelLocation(const Riscv64Label* label) const {
  CHECK(label->IsBound());
  uint32_t target = label->Position();
  if (label->prev_branch_id_plus_one_) {
    // Get label location based on the branch preceding it.
    uint32_t branch_id = label->prev_branch_id_plus_one_ - 1;
    const Branch* branch = GetBranch(branch_id);
    target += branch->GetEndLocation();
  }
  return target;
}

uint32_t Riscv64Assembler::GetAdjustedPosition(uint32_t old_position) {
  // We can reconstruct the adjustment by going through all the branches from the beginning
  // up to the old_position. Since we expect AdjustedPosition() to be called in a loop
  // with increasing old_position, we can use the data from last AdjustedPosition() to
  // continue where we left off and the whole loop should be O(m+n) where m is the number
  // of positions to adjust and n is the number of branches.
  if (old_position < last_old_position_) {
    last_position_adjustment_ = 0;
    last_old_position_ = 0;
    last_branch_id_ = 0;
  }
  while (last_branch_id_ != branches_.size()) {
    const Branch* branch = GetBranch(last_branch_id_);
    if (branch->GetLocation() >= old_position + last_position_adjustment_) {
      break;
    }
    last_position_adjustment_ += branch->GetSize() - branch->GetOldSize();
    ++last_branch_id_;
  }
  last_old_position_ = old_position;
  return old_position + last_position_adjustment_;
}

void Riscv64Assembler::FinalizeLabeledBranch(Riscv64Label* label) {
  uint32_t length = branches_.back().GetLength();
  if (!label->IsBound()) {
    // Branch forward (to a following label), distance is unknown.
    // The first branch forward will contain 0, serving as the terminator of
    // the list of forward-reaching branches.
    Emit(label->position_);
    length--;
    // Now make the label object point to this branch
    // (this forms a linked list of branches preceding this label).
    uint32_t branch_id = branches_.size() - 1;
    label->LinkTo(branch_id);
  }
  // Reserve space for the branch.
  for (; length != 0u; --length) {
    Nop();
  }
}

void Riscv64Assembler::Buncond(Riscv64Label* label, bool is_bare) {
  uint32_t target = label->IsBound() ? GetLabelLocation(label) : Branch::kUnresolved;
  branches_.emplace_back(buffer_.Size(), target, /* is_call= */ false, is_bare);
  FinalizeLabeledBranch(label);
}

void Riscv64Assembler::Bcond(Riscv64Label* label,
                            bool is_bare,
                            BranchCondition condition,
                            GpuRegister lhs,
                            GpuRegister rhs) {
  // If lhs = rhs, this can be a NOP.
  if (Branch::IsNop(condition, lhs, rhs)) {
    return;
  }
  uint32_t target = label->IsBound() ? GetLabelLocation(label) : Branch::kUnresolved;
  branches_.emplace_back(buffer_.Size(), target, condition, lhs, rhs, is_bare);
  FinalizeLabeledBranch(label);
}

void Riscv64Assembler::Call(Riscv64Label* label, bool is_bare) {
  uint32_t target = label->IsBound() ? GetLabelLocation(label) : Branch::kUnresolved;
  branches_.emplace_back(buffer_.Size(), target, /* is_call= */ true, is_bare);
  FinalizeLabeledBranch(label);
}

void Riscv64Assembler::LoadLabelAddress(GpuRegister dest_reg, Riscv64Label* label) {
  // Label address loads are treated as pseudo branches since they require very similar handling.
  DCHECK(!label->IsBound());
  branches_.emplace_back(buffer_.Size(), dest_reg, Branch::kLabel);
  FinalizeLabeledBranch(label);
}

Literal* Riscv64Assembler::NewLiteral(size_t size, const uint8_t* data) {
  // We don't support byte and half-word literals.
  if (size == 4u) {
    literals_.emplace_back(size, data);
    return &literals_.back();
  } else {
    DCHECK_EQ(size, 8u);
    long_literals_.emplace_back(size, data);
    return &long_literals_.back();
  }
}

void Riscv64Assembler::LoadLiteral(GpuRegister dest_reg,
                                  LoadOperandType load_type,
                                  Literal* literal) {
  // Literal loads are treated as pseudo branches since they require very similar handling.
  Branch::Type literal_type;
  switch (load_type) {
    case kLoadWord:
      DCHECK_EQ(literal->GetSize(), 4u);
      literal_type = Branch::kLiteral;
      break;
    case kLoadUnsignedWord:
      DCHECK_EQ(literal->GetSize(), 4u);
      literal_type = Branch::kLiteralUnsigned;
      break;
    case kLoadDoubleword:
      DCHECK_EQ(literal->GetSize(), 8u);
      literal_type = Branch::kLiteralLong;
      break;
    default:
      LOG(FATAL) << "Unexpected literal load type " << load_type;
      UNREACHABLE();
  }
  Riscv64Label* label = literal->GetLabel();
  DCHECK(!label->IsBound());
  branches_.emplace_back(buffer_.Size(), dest_reg, literal_type);
  FinalizeLabeledBranch(label);
}

JumpTable* Riscv64Assembler::CreateJumpTable(std::vector<Riscv64Label*>&& labels) {
  jump_tables_.emplace_back(std::move(labels));
  JumpTable* table = &jump_tables_.back();
  DCHECK(!table->GetLabel()->IsBound());
  return table;
}

void Riscv64Assembler::ReserveJumpTableSpace() {
  if (!jump_tables_.empty()) {
    for (JumpTable& table : jump_tables_) {
      Riscv64Label* label = table.GetLabel();
      Bind(label);

      // Bulk ensure capacity, as this may be large.
      size_t orig_size = buffer_.Size();
      size_t required_capacity = orig_size + table.GetSize();
      if (required_capacity > buffer_.Capacity()) {
        buffer_.ExtendCapacity(required_capacity);
      }
#ifndef NDEBUG
      buffer_.has_ensured_capacity_ = true;
#endif

      // Fill the space with dummy data as the data is not final
      // until the branches have been promoted. And we shouldn't
      // be moving uninitialized data during branch promotion.
      for (size_t cnt = table.GetData().size(), i = 0; i < cnt; i++) {
        buffer_.Emit<uint32_t>(0x1abe1234u);
      }

#ifndef NDEBUG
      buffer_.has_ensured_capacity_ = false;
#endif
    }
  }
}

void Riscv64Assembler::EmitJumpTables() {
  if (!jump_tables_.empty()) {
    CHECK(!overwriting_);
    // Switch from appending instructions at the end of the buffer to overwriting
    // existing instructions (here, jump tables) in the buffer.
    overwriting_ = true;

    for (JumpTable& table : jump_tables_) {
      Riscv64Label* table_label = table.GetLabel();
      uint32_t start = GetLabelLocation(table_label);
      overwrite_location_ = start;

      for (Riscv64Label* target : table.GetData()) {
        CHECK_EQ(buffer_.Load<uint32_t>(overwrite_location_), 0x1abe1234u);
        // The table will contain target addresses relative to the table start.
        uint32_t offset = GetLabelLocation(target) - start;
        Emit(offset);
      }
    }

    overwriting_ = false;
  }
}

void Riscv64Assembler::EmitLiterals() {
  if (!literals_.empty()) {
    for (Literal& literal : literals_) {
      Riscv64Label* label = literal.GetLabel();
      Bind(label);
      AssemblerBuffer::EnsureCapacity ensured(&buffer_);
      DCHECK_EQ(literal.GetSize(), 4u);
      for (size_t i = 0, size = literal.GetSize(); i != size; ++i) {
        buffer_.Emit<uint8_t>(literal.GetData()[i]);
      }
    }
  }
  if (!long_literals_.empty()) {
    // Reserve 4 bytes for potential alignment. If after the branch promotion the 64-bit
    // literals don't end up 8-byte-aligned, they will be moved down 4 bytes.
    Emit(0);  // NOP.
    for (Literal& literal : long_literals_) {
      Riscv64Label* label = literal.GetLabel();
      Bind(label);
      AssemblerBuffer::EnsureCapacity ensured(&buffer_);
      DCHECK_EQ(literal.GetSize(), 8u);
      for (size_t i = 0, size = literal.GetSize(); i != size; ++i) {
        buffer_.Emit<uint8_t>(literal.GetData()[i]);
      }
    }
  }
}

void Riscv64Assembler::PromoteBranches() {
  // Promote short branches to long as necessary.
  bool changed;
  do {
    changed = false;
    for (auto& branch : branches_) {
      CHECK(branch.IsResolved());
      uint32_t delta = branch.PromoteIfNeeded();
      // If this branch has been promoted and needs to expand in size,
      // relocate all branches by the expansion size.
      if (delta) {
        changed = true;
        uint32_t expand_location = branch.GetLocation();
        for (auto& branch2 : branches_) {
          branch2.Relocate(expand_location, delta);
        }
      }
    }
  } while (changed);

  // Account for branch expansion by resizing the code buffer
  // and moving the code in it to its final location.
  size_t branch_count = branches_.size();
  if (branch_count > 0) {
    // Resize.
    Branch& last_branch = branches_[branch_count - 1];
    uint32_t size_delta = last_branch.GetEndLocation() - last_branch.GetOldEndLocation();
    uint32_t old_size = buffer_.Size();
    buffer_.Resize(old_size + size_delta);
    // Move the code residing between branch placeholders.
    uint32_t end = old_size;
    for (size_t i = branch_count; i > 0; ) {
      Branch& branch = branches_[--i];
      uint32_t size = end - branch.GetOldEndLocation();
      buffer_.Move(branch.GetEndLocation(), branch.GetOldEndLocation(), size);
      end = branch.GetOldLocation();
    }
  }

  // Align 64-bit literals by moving them down by 4 bytes if needed.
  // This will reduce the PC-relative distance, which should be safe for both near and far literals.
  if (!long_literals_.empty()) {
    uint32_t first_literal_location = GetLabelLocation(long_literals_.front().GetLabel());
    size_t lit_size = long_literals_.size() * sizeof(uint64_t);
    size_t buf_size = buffer_.Size();
    // 64-bit literals must be at the very end of the buffer.
    CHECK_EQ(first_literal_location + lit_size, buf_size);
    if (!IsAligned<sizeof(uint64_t)>(first_literal_location)) {
      buffer_.Move(first_literal_location - sizeof(uint32_t), first_literal_location, lit_size);
      // The 4 reserved bytes proved useless, reduce the buffer size.
      buffer_.Resize(buf_size - sizeof(uint32_t));
      // Reduce target addresses in literal and address loads by 4 bytes in order for correct
      // offsets from PC to be generated.
      for (auto& branch : branches_) {
        uint32_t target = branch.GetTarget();
        if (target >= first_literal_location) {
          branch.Resolve(target - sizeof(uint32_t));
        }
      }
      // If after this we ever call GetLabelLocation() to get the location of a 64-bit literal,
      // we need to adjust the location of the literal's label as well.
      for (Literal& literal : long_literals_) {
        // Bound label's position is negative, hence incrementing it instead of decrementing.
        literal.GetLabel()->position_ += sizeof(uint32_t);
      }
    }
  }
}

// Note: make sure branch_info_[] and EmitBranch() are kept synchronized.
const Riscv64Assembler::Branch::BranchInfo Riscv64Assembler::Branch::branch_info_[] = {
  // short branches (can be promoted to long).
  {  1, 0, 0, Riscv64Assembler::Branch::kOffset21, 0 },  // kUncondBranch
  {  1, 0, 0, Riscv64Assembler::Branch::kOffset13, 0 },  // kCondBranch
  {  1, 0, 0, Riscv64Assembler::Branch::kOffset21, 0 },  // kCall
  // short branches (can't be promoted to long), forbidden/delay slots filled manually.
  {  1, 0, 0, Riscv64Assembler::Branch::kOffset21, 0 },  // kBareUncondBranch
  {  1, 0, 0, Riscv64Assembler::Branch::kOffset13, 0 },  // kBareCondBranch
  {  1, 0, 0, Riscv64Assembler::Branch::kOffset21, 0 },  // kBareCall

  // label.
  {  2, 0, 0, Riscv64Assembler::Branch::kOffset32, 0 },  // kLabel
  // literals.
  {  2, 0, 0, Riscv64Assembler::Branch::kOffset32, 0 },  // kLiteral
  {  2, 0, 0, Riscv64Assembler::Branch::kOffset32, 0 },  // kLiteralUnsigned
  {  2, 0, 0, Riscv64Assembler::Branch::kOffset32, 0 },  // kLiteralLong

  // Long branches.
  {  2, 0, 0, Riscv64Assembler::Branch::kOffset32, 0 },  // kLongUncondBranch
  {  3, 1, 0, Riscv64Assembler::Branch::kOffset32, 0 },  // kLongCondBranch
  {  2, 0, 0, Riscv64Assembler::Branch::kOffset32, 0 },  // kLongCall
};

// Note: make sure branch_info_[] and EmitBranch() are kept synchronized.
void Riscv64Assembler::EmitBranch(Riscv64Assembler::Branch* branch) {
  CHECK(overwriting_);
  overwrite_location_ = branch->GetLocation();
  uint32_t offset = branch->GetOffset();
  BranchCondition condition = branch->GetCondition();
  GpuRegister lhs = branch->GetLeftRegister();
  GpuRegister rhs = branch->GetRightRegister();
  switch (branch->GetType()) {
    // Short branches.
    case Branch::kUncondBranch:
      CHECK_EQ(overwrite_location_, branch->GetOffsetLocation());
      Bc(offset);
      break;
    case Branch::kCondBranch:
      CHECK_EQ(overwrite_location_, branch->GetOffsetLocation());
      EmitBcond(condition, lhs, rhs, offset);
      break;
    case Branch::kCall:
      CHECK_EQ(overwrite_location_, branch->GetOffsetLocation());
      Balc(offset);
      break;
    case Branch::kBareUncondBranch:
      CHECK_EQ(overwrite_location_, branch->GetOffsetLocation());
      Bc(offset);
      break;
    case Branch::kBareCondBranch:
      CHECK_EQ(overwrite_location_, branch->GetOffsetLocation());
      EmitBcond(condition, lhs, rhs, offset);
      break;
    case Branch::kBareCall:
      CHECK_EQ(overwrite_location_, branch->GetOffsetLocation());
      Balc(offset);
      break;

    // label.
    case Branch::kLabel:
      offset += (offset & 0x800) << 1;  // Account for sign extension in daddiu.
      CHECK_EQ(overwrite_location_, branch->GetOffsetLocation());
      Auipc(AT, High20Bits(offset));
      Addi(lhs, AT, Low12Bits(offset));
      break;
    // literals.
    case Branch::kLiteral:
      offset += (offset & 0x800) << 1;  // Account for sign extension in lw.
      CHECK_EQ(overwrite_location_, branch->GetOffsetLocation());
      Auipc(AT, High20Bits(offset));
      Lw(lhs, AT, Low12Bits(offset));
      break;
    case Branch::kLiteralUnsigned:
      offset += (offset & 0x800) << 1;  // Account for sign extension in lwu.
      CHECK_EQ(overwrite_location_, branch->GetOffsetLocation());
      Auipc(AT, High20Bits(offset));
      Lwu(lhs, AT, Low12Bits(offset));
      break;
    case Branch::kLiteralLong:
      offset += (offset & 0x800) << 1;  // Account for sign extension in ld.
      CHECK_EQ(overwrite_location_, branch->GetOffsetLocation());
      Auipc(AT, High20Bits(offset));
      Ld(lhs, AT, Low12Bits(offset));
      break;

    // Long branches.
    case Branch::kLongUncondBranch:
      offset += (offset & 0x800) << 1;  // Account for sign extension in jic.
      CHECK_EQ(overwrite_location_, branch->GetOffsetLocation());
      Auipc(AT, High20Bits(offset));
      Jic(AT, Low12Bits(offset));
      break;
    case Branch::kLongCondBranch:
      // Skip (2 + itself) instructions and continue if the Cond isn't taken.
      EmitBcond(Branch::OppositeCondition(condition), lhs, rhs, 12);
      offset += (offset & 0x800) << 1;  // Account for sign extension in jic.
      CHECK_EQ(overwrite_location_, branch->GetOffsetLocation());
      Auipc(AT, High20Bits(offset));
      Jic(AT, Low12Bits(offset));
      break;
    case Branch::kLongCall:
      offset += (offset & 0x800) << 1;  // Account for sign extension in jialc.
      CHECK_EQ(overwrite_location_, branch->GetOffsetLocation());
      Auipc(AT, High20Bits(offset));
      Jialc(AT, Low12Bits(offset));
      break;
  }
  CHECK_EQ(overwrite_location_, branch->GetEndLocation());
  CHECK_LT(branch->GetSize(), static_cast<uint32_t>(Branch::kMaxBranchSize));
}

void Riscv64Assembler::Bc(Riscv64Label* label, bool is_bare) {
  Buncond(label, is_bare);
}

void Riscv64Assembler::Balc(Riscv64Label* label, bool is_bare) {
  Call(label, is_bare);
}

void Riscv64Assembler::Jal(Riscv64Label* label, bool is_bare) {
  Call(label, is_bare);
}

void Riscv64Assembler::Bltc(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare) {
  Bcond(label, is_bare, kCondLT, rs, rt);
}

void Riscv64Assembler::Bltzc(GpuRegister rt, Riscv64Label* label, bool is_bare) {
  Bcond(label, is_bare, kCondLTZ, rt);
}

void Riscv64Assembler::Bgtzc(GpuRegister rt, Riscv64Label* label, bool is_bare) {
  Bcond(label, is_bare, kCondGTZ, rt);
}

void Riscv64Assembler::Bgec(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare) {
  Bcond(label, is_bare, kCondGE, rs, rt);
}

void Riscv64Assembler::Bgezc(GpuRegister rt, Riscv64Label* label, bool is_bare) {
  Bcond(label, is_bare, kCondGEZ, rt);
}

void Riscv64Assembler::Blezc(GpuRegister rt, Riscv64Label* label, bool is_bare) {
  Bcond(label, is_bare, kCondLEZ, rt);
}

void Riscv64Assembler::Bltuc(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare) {
  Bcond(label, is_bare, kCondLTU, rs, rt);
}

void Riscv64Assembler::Bgeuc(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare) {
  Bcond(label, is_bare, kCondGEU, rs, rt);
}

void Riscv64Assembler::Beqc(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare) {
  Bcond(label, is_bare, kCondEQ, rs, rt);
}

void Riscv64Assembler::Bnec(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare) {
  Bcond(label, is_bare, kCondNE, rs, rt);
}

void Riscv64Assembler::Beqzc(GpuRegister rs, Riscv64Label* label, bool is_bare) {
  Bcond(label, is_bare, kCondEQZ, rs);
}

void Riscv64Assembler::Bnezc(GpuRegister rs, Riscv64Label* label, bool is_bare) {
  Bcond(label, is_bare, kCondNEZ, rs);
}

void Riscv64Assembler::Bltz(GpuRegister rt, Riscv64Label* label, bool is_bare) {
  CHECK(is_bare);
  Bcond(label, is_bare, kCondLTZ, rt);
}

void Riscv64Assembler::Bgtz(GpuRegister rt, Riscv64Label* label, bool is_bare) {
  CHECK(is_bare);
  Bcond(label, is_bare, kCondGTZ, rt);
}

void Riscv64Assembler::Bgez(GpuRegister rt, Riscv64Label* label, bool is_bare) {
  CHECK(is_bare);
  Bcond(label, is_bare, kCondGEZ, rt);
}

void Riscv64Assembler::Blez(GpuRegister rt, Riscv64Label* label, bool is_bare) {
  CHECK(is_bare);
  Bcond(label, is_bare, kCondLEZ, rt);
}

void Riscv64Assembler::Beq(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare) {
  CHECK(is_bare);
  Bcond(label, is_bare, kCondEQ, rs, rt);
}

void Riscv64Assembler::Bne(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare) {
  CHECK(is_bare);
  Bcond(label, is_bare, kCondNE, rs, rt);
}

void Riscv64Assembler::Blt(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare) {
  CHECK(is_bare);
  Bcond(label, is_bare, kCondLT, rs, rt);
}

void Riscv64Assembler::Bge(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare) {
  CHECK(is_bare);
  Bcond(label, is_bare, kCondGE, rs, rt);
}

void Riscv64Assembler::Bltu(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare) {
  CHECK(is_bare);
  Bcond(label, is_bare, kCondLTU, rs, rt);
}

void Riscv64Assembler::Bgeu(GpuRegister rs, GpuRegister rt, Riscv64Label* label, bool is_bare) {
  CHECK(is_bare);
  Bcond(label, is_bare, kCondGEU, rs, rt);
}

void Riscv64Assembler::Beqz(GpuRegister rs, Riscv64Label* label, bool is_bare) {
//  CHECK(is_bare);
  Bcond(label, is_bare, kCondEQZ, rs);
}

void Riscv64Assembler::Bnez(GpuRegister rs, Riscv64Label* label, bool is_bare) {
//  CHECK(is_bare);
  Bcond(label, is_bare, kCondNEZ, rs);
}

void Riscv64Assembler::AdjustBaseAndOffset(GpuRegister& base,
                                          int32_t& offset,
                                          bool is_doubleword) {
  // This method is used to adjust the base register and offset pair
  // for a load/store when the offset doesn't fit into int16_t.
  // It is assumed that `base + offset` is sufficiently aligned for memory
  // operands that are machine word in size or smaller. For doubleword-sized
  // operands it's assumed that `base` is a multiple of 8, while `offset`
  // may be a multiple of 4 (e.g. 4-byte-aligned long and double arguments
  // and spilled variables on the stack accessed relative to the stack
  // pointer register).
  // We preserve the "alignment" of `offset` by adjusting it by a multiple of 8.
  CHECK_NE(base, AT);  // Must not overwrite the register `base` while loading `offset`.

  bool doubleword_aligned = IsAligned<kRiscv64DoublewordSize>(offset);
  bool two_accesses = is_doubleword && !doubleword_aligned;

  // IsInt<12> must be passed a signed value, hence the static cast below.
  if (IsInt<12>(offset) &&
      (!two_accesses || IsInt<12>(static_cast<int32_t>(offset + kRiscv64WordSize)))) {
    // Nothing to do: `offset` (and, if needed, `offset + 4`) fits into int12_t.
    return;
  }

  // Remember the "(mis)alignment" of `offset`, it will be checked at the end.
  uint32_t misalignment = offset & (kRiscv64DoublewordSize - 1);

  // First, see if `offset` can be represented as a sum of two 16-bit signed
  // offsets. This can save an instruction.
  // To simplify matters, only do this for a symmetric range of offsets from
  // about -64KB to about +64KB, allowing further addition of 4 when accessing
  // 64-bit variables with two 32-bit accesses.
  constexpr int32_t kMinOffsetForSimpleAdjustment = 0x7f8;  // Max int12_t that's a multiple of 8.
  constexpr int32_t kMaxOffsetForSimpleAdjustment = 2 * kMinOffsetForSimpleAdjustment;

  if (0 <= offset && offset <= kMaxOffsetForSimpleAdjustment) {
    Addi(AT, base, kMinOffsetForSimpleAdjustment);
    offset -= kMinOffsetForSimpleAdjustment;
  } else if (-kMaxOffsetForSimpleAdjustment <= offset && offset < 0) {
    Addi(AT, base, -kMinOffsetForSimpleAdjustment);
    offset += kMinOffsetForSimpleAdjustment;
  } else {
    // In more complex cases take advantage of the daui instruction, e.g.:
    //    daui   AT, base, offset_high
    //   [dahi   AT, 1]                       // When `offset` is close to +2GB.
    //    lw     reg_lo, offset_low(AT)
    //   [lw     reg_hi, (offset_low+4)(AT)]  // If misaligned 64-bit load.
    // or when offset_low+4 overflows int16_t:
    //    daui   AT, base, offset_high
    //    daddiu AT, AT, 8
    //    lw     reg_lo, (offset_low-8)(AT)
    //    lw     reg_hi, (offset_low-4)(AT)
    int32_t offset_low12 = 0xFFF & offset;
    int32_t offset_high20 = offset >> 12;

    if (offset_low12 & 0x800) {  // check int12_t sign bit
      offset_high20 += 1;
      offset_low12 |= 0xFFFFF000;  // sign extend offset_low12
    }

    Lui(AT, offset_high20);
    Add(AT, base, AT);

    if (two_accesses && !IsInt<12>(static_cast<int32_t>(offset_low12 + kRiscv64WordSize))) {
      // Avoid overflow in the 12-bit offset of the load/store instruction when adding 4.
      Addi(AT, AT, kRiscv64DoublewordSize);
      offset_low12 -= kRiscv64DoublewordSize;
    }

    offset = offset_low12;
  }
  base = AT;

  CHECK(IsInt<12>(offset));
  if (two_accesses) {
    CHECK(IsInt<12>(static_cast<int32_t>(offset + kRiscv64WordSize)));
  }
  CHECK_EQ(misalignment, offset & (kRiscv64DoublewordSize - 1));
}

void Riscv64Assembler::AdjustBaseOffsetAndElementSizeShift(GpuRegister& base,
                                                          int32_t& offset,
                                                          int& element_size_shift) {
  // This method is used to adjust the base register, offset and element_size_shift
  // for a vector load/store when the offset doesn't fit into allowed number of bits.
  // MSA ld.df and st.df instructions take signed offsets as arguments, but maximum
  // offset is dependant on the size of the data format df (10-bit offsets for ld.b,
  // 11-bit for ld.h, 12-bit for ld.w and 13-bit for ld.d).
  // If element_size_shift is non-negative at entry, it won't be changed, but offset
  // will be checked for appropriate alignment. If negative at entry, it will be
  // adjusted based on offset for maximum fit.
  // It's assumed that `base` is a multiple of 8.

  CHECK_NE(base, AT);  // Must not overwrite the register `base` while loading `offset`.

  if (element_size_shift >= 0) {
    CHECK_LE(element_size_shift, TIMES_8);
    CHECK_GE(JAVASTYLE_CTZ(offset), element_size_shift);
  } else if (IsAligned<kRiscv64DoublewordSize>(offset)) {
    element_size_shift = TIMES_8;
  } else if (IsAligned<kRiscv64WordSize>(offset)) {
    element_size_shift = TIMES_4;
  } else if (IsAligned<kRiscv64HalfwordSize>(offset)) {
    element_size_shift = TIMES_2;
  } else {
    element_size_shift = TIMES_1;
  }

  const int low_len = 10 + element_size_shift;  // How many low bits of `offset` ld.df/st.df
                                                // will take.
  int16_t low = offset & ((1 << low_len) - 1);  // Isolate these bits.
  low -= (low & (1 << (low_len - 1))) << 1;     // Sign-extend these bits.
  if (low == offset) {
    return;  // `offset` fits into ld.df/st.df.
  }

  // First, see if `offset` can be represented as a sum of two signed offsets.
  // This can save an instruction.

  // Max int16_t that's a multiple of element size.
  const int32_t kMaxDeltaForSimpleAdjustment = 0x8000 - (1 << element_size_shift);
  // Max ld.df/st.df offset that's a multiple of element size.
  const int32_t kMaxLoadStoreOffset = 0x1ff << element_size_shift;
  const int32_t kMaxOffsetForSimpleAdjustment = kMaxDeltaForSimpleAdjustment + kMaxLoadStoreOffset;

  if (IsInt<16>(offset)) {
    Daddiu(AT, base, offset);
    offset = 0;
  } else if (0 <= offset && offset <= kMaxOffsetForSimpleAdjustment) {
    Daddiu(AT, base, kMaxDeltaForSimpleAdjustment);
    offset -= kMaxDeltaForSimpleAdjustment;
  } else if (-kMaxOffsetForSimpleAdjustment <= offset && offset < 0) {
    Daddiu(AT, base, -kMaxDeltaForSimpleAdjustment);
    offset += kMaxDeltaForSimpleAdjustment;
  } else {
    // Let's treat `offset` as 64-bit to simplify handling of sign
    // extensions in the instructions that supply its smaller signed parts.
    //
    // 16-bit or smaller parts of `offset`:
    // |63  top  48|47  hi  32|31  upper  16|15  mid  13-10|12-9  low  0|
    //
    // Instructions that supply each part as a signed integer addend:
    // |dati       |dahi      |daui         |daddiu        |ld.df/st.df |
    //
    // `top` is always 0, so dati isn't used.
    // `hi` is 1 when `offset` is close to +2GB and 0 otherwise.
    uint64_t tmp = static_cast<uint64_t>(offset) - low;  // Exclude `low` from the rest of `offset`
                                                         // (accounts for sign of `low`).
    tmp += (tmp & (UINT64_C(1) << 15)) << 1;  // Account for sign extension in daddiu.
    tmp += (tmp & (UINT64_C(1) << 31)) << 1;  // Account for sign extension in daui.
    int16_t mid = Low16Bits(tmp);
    int16_t upper = High16Bits(tmp);
    int16_t hi = Low16Bits(High32Bits(tmp));
    Daui(AT, base, upper);
    if (hi != 0) {
      CHECK_EQ(hi, 1);
      Dahi(AT, hi);
    }
    if (mid != 0) {
      Daddiu(AT, AT, mid);
    }
    offset = low;
  }
  base = AT;
  CHECK_GE(JAVASTYLE_CTZ(offset), element_size_shift);
  CHECK(IsInt<10>(offset >> element_size_shift));
}

void Riscv64Assembler::LoadFromOffset(LoadOperandType type,
                                     GpuRegister reg,
                                     GpuRegister base,
                                     int32_t offset) {
  LoadFromOffset<>(type, reg, base, offset);
}

void Riscv64Assembler::LoadFpuFromOffset(LoadOperandType type,
                                        FpuRegister reg,
                                        GpuRegister base,
                                        int32_t offset) {
  LoadFpuFromOffset<>(type, reg, base, offset);
}

void Riscv64Assembler::EmitLoad(ManagedRegister m_dst, GpuRegister src_register, int32_t src_offset,
                               size_t size) {
  Riscv64ManagedRegister dst = m_dst.AsRiscv64();
  if (dst.IsNoRegister()) {
    CHECK_EQ(0u, size) << dst;
  } else if (dst.IsGpuRegister()) {
    if (size == 4) {
      LoadFromOffset(kLoadWord, dst.AsGpuRegister(), src_register, src_offset);
    } else if (size == 8) {
      CHECK_EQ(8u, size) << dst;
      LoadFromOffset(kLoadDoubleword, dst.AsGpuRegister(), src_register, src_offset);
    } else {
      UNIMPLEMENTED(FATAL) << "We only support Load() of size 4 and 8";
    }
  } else if (dst.IsFpuRegister()) {
    if (size == 4) {
      CHECK_EQ(4u, size) << dst;
      LoadFpuFromOffset(kLoadWord, dst.AsFpuRegister(), src_register, src_offset);
    } else if (size == 8) {
      CHECK_EQ(8u, size) << dst;
      LoadFpuFromOffset(kLoadDoubleword, dst.AsFpuRegister(), src_register, src_offset);
    } else {
      UNIMPLEMENTED(FATAL) << "We only support Load() of size 4 and 8";
    }
  }
}

void Riscv64Assembler::StoreToOffset(StoreOperandType type,
                                    GpuRegister reg,
                                    GpuRegister base,
                                    int32_t offset) {
  StoreToOffset<>(type, reg, base, offset);
}

void Riscv64Assembler::StoreFpuToOffset(StoreOperandType type,
                                       FpuRegister reg,
                                       GpuRegister base,
                                       int32_t offset) {
  StoreFpuToOffset<>(type, reg, base, offset);
}

static dwarf::Reg DWARFReg(GpuRegister reg) {
  return dwarf::Reg::Riscv64Core(static_cast<int>(reg));
}

constexpr size_t kFramePointerSize = 8;

void Riscv64Assembler::BuildFrame(size_t frame_size,
                                 ManagedRegister method_reg,
                                 ArrayRef<const ManagedRegister> callee_save_regs,
                                 const ManagedRegisterEntrySpills& entry_spills) {
  CHECK_ALIGNED(frame_size, kStackAlignment);
  DCHECK(!overwriting_);

  // Increase frame to required size.
  IncreaseFrameSize(frame_size);

  // Push callee saves and return address
  int stack_offset = frame_size - kFramePointerSize;
  StoreToOffset(kStoreDoubleword, RA, SP, stack_offset);
  cfi_.RelOffset(DWARFReg(RA), stack_offset);
  for (int i = callee_save_regs.size() - 1; i >= 0; --i) {
    stack_offset -= kFramePointerSize;
    GpuRegister reg = callee_save_regs[i].AsRiscv64().AsGpuRegister();
    StoreToOffset(kStoreDoubleword, reg, SP, stack_offset);
    cfi_.RelOffset(DWARFReg(reg), stack_offset);
  }

  // Write out Method*.
  StoreToOffset(kStoreDoubleword, method_reg.AsRiscv64().AsGpuRegister(), SP, 0);

  // Write out entry spills.
  int32_t offset = frame_size + kFramePointerSize;
  for (const ManagedRegisterSpill& spill : entry_spills) {
    Riscv64ManagedRegister reg = spill.AsRiscv64();
    int32_t size = spill.getSize();
    if (reg.IsNoRegister()) {
      // only increment stack offset.
      offset += size;
    } else if (reg.IsFpuRegister()) {
      StoreFpuToOffset((size == 4) ? kStoreWord : kStoreDoubleword,
          reg.AsFpuRegister(), SP, offset);
      offset += size;
    } else if (reg.IsGpuRegister()) {
      StoreToOffset((size == 4) ? kStoreWord : kStoreDoubleword,
          reg.AsGpuRegister(), SP, offset);
      offset += size;
    }
  }
}

void Riscv64Assembler::RemoveFrame(size_t frame_size,
                                  ArrayRef<const ManagedRegister> callee_save_regs,
                                  bool may_suspend ATTRIBUTE_UNUSED) {
  CHECK_ALIGNED(frame_size, kStackAlignment);
  DCHECK(!overwriting_);
  cfi_.RememberState();

  // Pop callee saves and return address
  int stack_offset = frame_size - (callee_save_regs.size() * kFramePointerSize) - kFramePointerSize;
  for (size_t i = 0; i < callee_save_regs.size(); ++i) {
    GpuRegister reg = callee_save_regs[i].AsRiscv64().AsGpuRegister();
    LoadFromOffset(kLoadDoubleword, reg, SP, stack_offset);
    cfi_.Restore(DWARFReg(reg));
    stack_offset += kFramePointerSize;
  }
  LoadFromOffset(kLoadDoubleword, RA, SP, stack_offset);
  cfi_.Restore(DWARFReg(RA));

  // Decrease frame to required size.
  DecreaseFrameSize(frame_size);

  // Then jump to the return address.
  Jr(RA);
  Nop();

  // The CFI should be restored for any code that follows the exit block.
  cfi_.RestoreState();
  cfi_.DefCFAOffset(frame_size);
}

void Riscv64Assembler::IncreaseFrameSize(size_t adjust) {
  CHECK_ALIGNED(adjust, kFramePointerSize);
  DCHECK(!overwriting_);
  Daddiu64(SP, SP, static_cast<int32_t>(-adjust));
  cfi_.AdjustCFAOffset(adjust);
}

void Riscv64Assembler::DecreaseFrameSize(size_t adjust) {
  CHECK_ALIGNED(adjust, kFramePointerSize);
  DCHECK(!overwriting_);
  Daddiu64(SP, SP, static_cast<int32_t>(adjust));
  cfi_.AdjustCFAOffset(-adjust);
}

void Riscv64Assembler::Store(FrameOffset dest, ManagedRegister msrc, size_t size) {
  Riscv64ManagedRegister src = msrc.AsRiscv64();
  if (src.IsNoRegister()) {
    CHECK_EQ(0u, size);
  } else if (src.IsGpuRegister()) {
    CHECK(size == 4 || size == 8) << size;
    if (size == 8) {
      StoreToOffset(kStoreDoubleword, src.AsGpuRegister(), SP, dest.Int32Value());
    } else if (size == 4) {
      StoreToOffset(kStoreWord, src.AsGpuRegister(), SP, dest.Int32Value());
    } else {
      UNIMPLEMENTED(FATAL) << "We only support Store() of size 4 and 8";
    }
  } else if (src.IsFpuRegister()) {
    CHECK(size == 4 || size == 8) << size;
    if (size == 8) {
      StoreFpuToOffset(kStoreDoubleword, src.AsFpuRegister(), SP, dest.Int32Value());
    } else if (size == 4) {
      StoreFpuToOffset(kStoreWord, src.AsFpuRegister(), SP, dest.Int32Value());
    } else {
      UNIMPLEMENTED(FATAL) << "We only support Store() of size 4 and 8";
    }
  }
}

void Riscv64Assembler::StoreRef(FrameOffset dest, ManagedRegister msrc) {
  Riscv64ManagedRegister src = msrc.AsRiscv64();
  CHECK(src.IsGpuRegister());
  StoreToOffset(kStoreWord, src.AsGpuRegister(), SP, dest.Int32Value());
}

void Riscv64Assembler::StoreRawPtr(FrameOffset dest, ManagedRegister msrc) {
  Riscv64ManagedRegister src = msrc.AsRiscv64();
  CHECK(src.IsGpuRegister());
  StoreToOffset(kStoreDoubleword, src.AsGpuRegister(), SP, dest.Int32Value());
}

void Riscv64Assembler::StoreImmediateToFrame(FrameOffset dest, uint32_t imm,
                                            ManagedRegister mscratch) {
  Riscv64ManagedRegister scratch = mscratch.AsRiscv64();
  CHECK(scratch.IsGpuRegister()) << scratch;
  LoadConst32(scratch.AsGpuRegister(), imm);
  StoreToOffset(kStoreWord, scratch.AsGpuRegister(), SP, dest.Int32Value());
}

void Riscv64Assembler::StoreStackOffsetToThread(ThreadOffset64 thr_offs,
                                               FrameOffset fr_offs,
                                               ManagedRegister mscratch) {
  Riscv64ManagedRegister scratch = mscratch.AsRiscv64();
  CHECK(scratch.IsGpuRegister()) << scratch;
  Daddiu64(scratch.AsGpuRegister(), SP, fr_offs.Int32Value());
  StoreToOffset(kStoreDoubleword, scratch.AsGpuRegister(), S1, thr_offs.Int32Value());
}

void Riscv64Assembler::StoreStackPointerToThread(ThreadOffset64 thr_offs) {
  StoreToOffset(kStoreDoubleword, SP, S1, thr_offs.Int32Value());
}

void Riscv64Assembler::StoreSpanning(FrameOffset dest, ManagedRegister msrc,
                                    FrameOffset in_off, ManagedRegister mscratch) {
  Riscv64ManagedRegister src = msrc.AsRiscv64();
  Riscv64ManagedRegister scratch = mscratch.AsRiscv64();
  StoreToOffset(kStoreDoubleword, src.AsGpuRegister(), SP, dest.Int32Value());
  LoadFromOffset(kLoadDoubleword, scratch.AsGpuRegister(), SP, in_off.Int32Value());
  StoreToOffset(kStoreDoubleword, scratch.AsGpuRegister(), SP, dest.Int32Value() + 8);
}

void Riscv64Assembler::Load(ManagedRegister mdest, FrameOffset src, size_t size) {
  return EmitLoad(mdest, SP, src.Int32Value(), size);
}

void Riscv64Assembler::LoadFromThread(ManagedRegister mdest, ThreadOffset64 src, size_t size) {
  return EmitLoad(mdest, S1, src.Int32Value(), size);
}

void Riscv64Assembler::LoadRef(ManagedRegister mdest, FrameOffset src) {
  Riscv64ManagedRegister dest = mdest.AsRiscv64();
  CHECK(dest.IsGpuRegister());
  LoadFromOffset(kLoadUnsignedWord, dest.AsGpuRegister(), SP, src.Int32Value());
}

void Riscv64Assembler::LoadRef(ManagedRegister mdest, ManagedRegister base, MemberOffset offs,
                              bool unpoison_reference) {
  Riscv64ManagedRegister dest = mdest.AsRiscv64();
  CHECK(dest.IsGpuRegister() && base.AsRiscv64().IsGpuRegister());
  LoadFromOffset(kLoadUnsignedWord, dest.AsGpuRegister(),
                 base.AsRiscv64().AsGpuRegister(), offs.Int32Value());
  if (unpoison_reference) {
    MaybeUnpoisonHeapReference(dest.AsGpuRegister());
  }
}

void Riscv64Assembler::LoadRawPtr(ManagedRegister mdest, ManagedRegister base,
                                 Offset offs) {
  Riscv64ManagedRegister dest = mdest.AsRiscv64();
  CHECK(dest.IsGpuRegister() && base.AsRiscv64().IsGpuRegister());
  LoadFromOffset(kLoadDoubleword, dest.AsGpuRegister(),
                 base.AsRiscv64().AsGpuRegister(), offs.Int32Value());
}

void Riscv64Assembler::LoadRawPtrFromThread(ManagedRegister mdest, ThreadOffset64 offs) {
  Riscv64ManagedRegister dest = mdest.AsRiscv64();
  CHECK(dest.IsGpuRegister());
  LoadFromOffset(kLoadDoubleword, dest.AsGpuRegister(), S1, offs.Int32Value());
}

void Riscv64Assembler::SignExtend(ManagedRegister mreg ATTRIBUTE_UNUSED,
                                 size_t size ATTRIBUTE_UNUSED) {
  UNIMPLEMENTED(FATAL) << "No sign extension necessary for RISCV64";
}

void Riscv64Assembler::ZeroExtend(ManagedRegister mreg ATTRIBUTE_UNUSED,
                                 size_t size ATTRIBUTE_UNUSED) {
  UNIMPLEMENTED(FATAL) << "No zero extension necessary for RISCV64";
}

void Riscv64Assembler::Move(ManagedRegister mdest, ManagedRegister msrc, size_t size) {
  Riscv64ManagedRegister dest = mdest.AsRiscv64();
  Riscv64ManagedRegister src = msrc.AsRiscv64();
  if (!dest.Equals(src)) {
    if (dest.IsGpuRegister()) {
      CHECK(src.IsGpuRegister()) << src;
      Move(dest.AsGpuRegister(), src.AsGpuRegister());
    } else if (dest.IsFpuRegister()) {
      CHECK(src.IsFpuRegister()) << src;
      if (size == 4) {
        MovS(dest.AsFpuRegister(), src.AsFpuRegister());
      } else if (size == 8) {
        MovD(dest.AsFpuRegister(), src.AsFpuRegister());
      } else {
        UNIMPLEMENTED(FATAL) << "We only support Copy() of size 4 and 8";
      }
    }
  }
}

void Riscv64Assembler::CopyRef(FrameOffset dest, FrameOffset src,
                              ManagedRegister mscratch) {
  Riscv64ManagedRegister scratch = mscratch.AsRiscv64();
  CHECK(scratch.IsGpuRegister()) << scratch;
  LoadFromOffset(kLoadWord, scratch.AsGpuRegister(), SP, src.Int32Value());
  StoreToOffset(kStoreWord, scratch.AsGpuRegister(), SP, dest.Int32Value());
}

void Riscv64Assembler::CopyRawPtrFromThread(FrameOffset fr_offs,
                                           ThreadOffset64 thr_offs,
                                           ManagedRegister mscratch) {
  Riscv64ManagedRegister scratch = mscratch.AsRiscv64();
  CHECK(scratch.IsGpuRegister()) << scratch;
  LoadFromOffset(kLoadDoubleword, scratch.AsGpuRegister(), S1, thr_offs.Int32Value());
  StoreToOffset(kStoreDoubleword, scratch.AsGpuRegister(), SP, fr_offs.Int32Value());
}

void Riscv64Assembler::CopyRawPtrToThread(ThreadOffset64 thr_offs,
                                         FrameOffset fr_offs,
                                         ManagedRegister mscratch) {
  Riscv64ManagedRegister scratch = mscratch.AsRiscv64();
  CHECK(scratch.IsGpuRegister()) << scratch;
  LoadFromOffset(kLoadDoubleword, scratch.AsGpuRegister(),
                 SP, fr_offs.Int32Value());
  StoreToOffset(kStoreDoubleword, scratch.AsGpuRegister(),
                S1, thr_offs.Int32Value());
}

void Riscv64Assembler::Copy(FrameOffset dest, FrameOffset src,
                           ManagedRegister mscratch, size_t size) {
  Riscv64ManagedRegister scratch = mscratch.AsRiscv64();
  CHECK(scratch.IsGpuRegister()) << scratch;
  CHECK(size == 4 || size == 8) << size;
  if (size == 4) {
    LoadFromOffset(kLoadWord, scratch.AsGpuRegister(), SP, src.Int32Value());
    StoreToOffset(kStoreDoubleword, scratch.AsGpuRegister(), SP, dest.Int32Value());
  } else if (size == 8) {
    LoadFromOffset(kLoadDoubleword, scratch.AsGpuRegister(), SP, src.Int32Value());
    StoreToOffset(kStoreDoubleword, scratch.AsGpuRegister(), SP, dest.Int32Value());
  } else {
    UNIMPLEMENTED(FATAL) << "We only support Copy() of size 4 and 8";
  }
}

void Riscv64Assembler::Copy(FrameOffset dest, ManagedRegister src_base, Offset src_offset,
                           ManagedRegister mscratch, size_t size) {
  GpuRegister scratch = mscratch.AsRiscv64().AsGpuRegister();
  CHECK(size == 4 || size == 8) << size;
  if (size == 4) {
    LoadFromOffset(kLoadWord, scratch, src_base.AsRiscv64().AsGpuRegister(),
                   src_offset.Int32Value());
    StoreToOffset(kStoreDoubleword, scratch, SP, dest.Int32Value());
  } else if (size == 8) {
    LoadFromOffset(kLoadDoubleword, scratch, src_base.AsRiscv64().AsGpuRegister(),
                   src_offset.Int32Value());
    StoreToOffset(kStoreDoubleword, scratch, SP, dest.Int32Value());
  } else {
    UNIMPLEMENTED(FATAL) << "We only support Copy() of size 4 and 8";
  }
}

void Riscv64Assembler::Copy(ManagedRegister dest_base, Offset dest_offset, FrameOffset src,
                           ManagedRegister mscratch, size_t size) {
  GpuRegister scratch = mscratch.AsRiscv64().AsGpuRegister();
  CHECK(size == 4 || size == 8) << size;
  if (size == 4) {
    LoadFromOffset(kLoadWord, scratch, SP, src.Int32Value());
    StoreToOffset(kStoreDoubleword, scratch, dest_base.AsRiscv64().AsGpuRegister(),
                  dest_offset.Int32Value());
  } else if (size == 8) {
    LoadFromOffset(kLoadDoubleword, scratch, SP, src.Int32Value());
    StoreToOffset(kStoreDoubleword, scratch, dest_base.AsRiscv64().AsGpuRegister(),
                  dest_offset.Int32Value());
  } else {
    UNIMPLEMENTED(FATAL) << "We only support Copy() of size 4 and 8";
  }
}

void Riscv64Assembler::Copy(FrameOffset dest ATTRIBUTE_UNUSED,
                           FrameOffset src_base ATTRIBUTE_UNUSED,
                           Offset src_offset ATTRIBUTE_UNUSED,
                           ManagedRegister mscratch ATTRIBUTE_UNUSED,
                           size_t size ATTRIBUTE_UNUSED) {
  UNIMPLEMENTED(FATAL) << "No RISCV64 implementation";
}

void Riscv64Assembler::Copy(ManagedRegister dest, Offset dest_offset,
                           ManagedRegister src, Offset src_offset,
                           ManagedRegister mscratch, size_t size) {
  GpuRegister scratch = mscratch.AsRiscv64().AsGpuRegister();
  CHECK(size == 4 || size == 8) << size;
  if (size == 4) {
    LoadFromOffset(kLoadWord, scratch, src.AsRiscv64().AsGpuRegister(), src_offset.Int32Value());
    StoreToOffset(kStoreDoubleword, scratch, dest.AsRiscv64().AsGpuRegister(), dest_offset.Int32Value());
  } else if (size == 8) {
    LoadFromOffset(kLoadDoubleword, scratch, src.AsRiscv64().AsGpuRegister(),
                   src_offset.Int32Value());
    StoreToOffset(kStoreDoubleword, scratch, dest.AsRiscv64().AsGpuRegister(),
                  dest_offset.Int32Value());
  } else {
    UNIMPLEMENTED(FATAL) << "We only support Copy() of size 4 and 8";
  }
}

void Riscv64Assembler::Copy(FrameOffset dest ATTRIBUTE_UNUSED,
                           Offset dest_offset ATTRIBUTE_UNUSED,
                           FrameOffset src ATTRIBUTE_UNUSED,
                           Offset src_offset ATTRIBUTE_UNUSED,
                           ManagedRegister mscratch ATTRIBUTE_UNUSED,
                           size_t size ATTRIBUTE_UNUSED) {
  UNIMPLEMENTED(FATAL) << "No RISCV64 implementation";
}

void Riscv64Assembler::MemoryBarrier(ManagedRegister mreg ATTRIBUTE_UNUSED) {
  // TODO: sync?
  UNIMPLEMENTED(FATAL) << "No RISCV64 implementation";
}

void Riscv64Assembler::CreateHandleScopeEntry(ManagedRegister mout_reg,
                                             FrameOffset handle_scope_offset,
                                             ManagedRegister min_reg,
                                             bool null_allowed) {
  Riscv64ManagedRegister out_reg = mout_reg.AsRiscv64();
  Riscv64ManagedRegister in_reg = min_reg.AsRiscv64();
  CHECK(in_reg.IsNoRegister() || in_reg.IsGpuRegister()) << in_reg;
  CHECK(out_reg.IsGpuRegister()) << out_reg;
  if (null_allowed) {
    Riscv64Label null_arg;
    // Null values get a handle scope entry value of 0.  Otherwise, the handle scope entry is
    // the address in the handle scope holding the reference.
    // e.g. out_reg = (handle == 0) ? 0 : (SP+handle_offset)
    if (in_reg.IsNoRegister()) {
      LoadFromOffset(kLoadUnsignedWord, out_reg.AsGpuRegister(),
                     SP, handle_scope_offset.Int32Value());
      in_reg = out_reg;
    }
    if (!out_reg.Equals(in_reg)) {
      LoadConst32(out_reg.AsGpuRegister(), 0);
    }
    Beqzc(in_reg.AsGpuRegister(), &null_arg);
    Daddiu64(out_reg.AsGpuRegister(), SP, handle_scope_offset.Int32Value());
    Bind(&null_arg);
  } else {
    Daddiu64(out_reg.AsGpuRegister(), SP, handle_scope_offset.Int32Value());
  }
}

void Riscv64Assembler::CreateHandleScopeEntry(FrameOffset out_off,
                                             FrameOffset handle_scope_offset,
                                             ManagedRegister mscratch,
                                             bool null_allowed) {
  Riscv64ManagedRegister scratch = mscratch.AsRiscv64();
  CHECK(scratch.IsGpuRegister()) << scratch;
  if (null_allowed) {
    Riscv64Label null_arg;
    LoadFromOffset(kLoadUnsignedWord, scratch.AsGpuRegister(), SP,
                   handle_scope_offset.Int32Value());
    // Null values get a handle scope entry value of 0.  Otherwise, the handle scope entry is
    // the address in the handle scope holding the reference.
    // e.g. scratch = (scratch == 0) ? 0 : (SP+handle_scope_offset)
    Beqzc(scratch.AsGpuRegister(), &null_arg);
    Daddiu64(scratch.AsGpuRegister(), SP, handle_scope_offset.Int32Value());
    Bind(&null_arg);
  } else {
    Daddiu64(scratch.AsGpuRegister(), SP, handle_scope_offset.Int32Value());
  }
  StoreToOffset(kStoreDoubleword, scratch.AsGpuRegister(), SP, out_off.Int32Value());
}

// Given a handle scope entry, load the associated reference.
void Riscv64Assembler::LoadReferenceFromHandleScope(ManagedRegister mout_reg,
                                                   ManagedRegister min_reg) {
  Riscv64ManagedRegister out_reg = mout_reg.AsRiscv64();
  Riscv64ManagedRegister in_reg = min_reg.AsRiscv64();
  CHECK(out_reg.IsGpuRegister()) << out_reg;
  CHECK(in_reg.IsGpuRegister()) << in_reg;
  Riscv64Label null_arg;
  if (!out_reg.Equals(in_reg)) {
    LoadConst32(out_reg.AsGpuRegister(), 0);
  }
  Beqzc(in_reg.AsGpuRegister(), &null_arg);
  LoadFromOffset(kLoadDoubleword, out_reg.AsGpuRegister(),
                 in_reg.AsGpuRegister(), 0);
  Bind(&null_arg);
}

void Riscv64Assembler::VerifyObject(ManagedRegister src ATTRIBUTE_UNUSED,
                                   bool could_be_null ATTRIBUTE_UNUSED) {
  // TODO: not validating references
}

void Riscv64Assembler::VerifyObject(FrameOffset src ATTRIBUTE_UNUSED,
                                   bool could_be_null ATTRIBUTE_UNUSED) {
  // TODO: not validating references
}

void Riscv64Assembler::Call(ManagedRegister mbase, Offset offset, ManagedRegister mscratch) {
  Riscv64ManagedRegister base = mbase.AsRiscv64();
  Riscv64ManagedRegister scratch = mscratch.AsRiscv64();
  CHECK(base.IsGpuRegister()) << base;
  CHECK(scratch.IsGpuRegister()) << scratch;
  LoadFromOffset(kLoadDoubleword, scratch.AsGpuRegister(),
                 base.AsGpuRegister(), offset.Int32Value());
  Jalr(scratch.AsGpuRegister());
  Nop();
  // TODO: place reference map on call
}

void Riscv64Assembler::Call(FrameOffset base, Offset offset, ManagedRegister mscratch) {
  Riscv64ManagedRegister scratch = mscratch.AsRiscv64();
  CHECK(scratch.IsGpuRegister()) << scratch;
  // Call *(*(SP + base) + offset)
  LoadFromOffset(kLoadDoubleword, scratch.AsGpuRegister(),
                 SP, base.Int32Value());
  LoadFromOffset(kLoadDoubleword, scratch.AsGpuRegister(),
                 scratch.AsGpuRegister(), offset.Int32Value());
  Jalr(scratch.AsGpuRegister());
  Nop();
  // TODO: place reference map on call
}

void Riscv64Assembler::CallFromThread(ThreadOffset64 offset ATTRIBUTE_UNUSED,
                                     ManagedRegister mscratch ATTRIBUTE_UNUSED) {
  UNIMPLEMENTED(FATAL) << "No RISCV64 implementation";
}

void Riscv64Assembler::GetCurrentThread(ManagedRegister tr) {
  Move(tr.AsRiscv64().AsGpuRegister(), S1);
}

void Riscv64Assembler::GetCurrentThread(FrameOffset offset,
                                       ManagedRegister mscratch ATTRIBUTE_UNUSED) {
  StoreToOffset(kStoreDoubleword, S1, SP, offset.Int32Value());
}

void Riscv64Assembler::ExceptionPoll(ManagedRegister mscratch, size_t stack_adjust) {
  Riscv64ManagedRegister scratch = mscratch.AsRiscv64();
  exception_blocks_.emplace_back(scratch, stack_adjust);
  LoadFromOffset(kLoadDoubleword,
                 scratch.AsGpuRegister(),
                 S1,
                 Thread::ExceptionOffset<kRiscv64PointerSize>().Int32Value());
  Bnezc(scratch.AsGpuRegister(), exception_blocks_.back().Entry());
}

void Riscv64Assembler::EmitExceptionPoll(Riscv64ExceptionSlowPath* exception) {
  Bind(exception->Entry());
  if (exception->stack_adjust_ != 0) {  // Fix up the frame.
    DecreaseFrameSize(exception->stack_adjust_);
  }
  // Pass exception object as argument.
  // Don't care about preserving A0 as this call won't return.
  CheckEntrypointTypes<kQuickDeliverException, void, mirror::Object*>();
  Move(A0, exception->scratch_.AsGpuRegister());
  // Set up call to Thread::Current()->pDeliverException
  LoadFromOffset(kLoadDoubleword,
                 T9,
                 S1,
                 QUICK_ENTRYPOINT_OFFSET(kRiscv64PointerSize, pDeliverException).Int32Value());
  Jr(T9);
  Nop();

  // Call never returns
  Break();
}

// TODO dvt porting...
void Riscv64Assembler::EmitI5(uint16_t funct7, uint16_t imm5, GpuRegister rs1, int funct3, GpuRegister rd, int opcode) {
  uint32_t encoding = static_cast<uint32_t>(funct7) << 25 |
                      (static_cast<uint32_t>(imm5) & 0x1F) << 20 |
                      static_cast<uint32_t>(rs1) << 15 |
                      static_cast<uint32_t>(funct3) << 12 |
                      static_cast<uint32_t>(rd) << 7 |
                      opcode;
  Emit(encoding);
}

void Riscv64Assembler::EmitI6(uint16_t funct6, uint16_t imm6, GpuRegister rs1, int funct3, GpuRegister rd, int opcode) {
  uint32_t encoding = static_cast<uint32_t>(funct6) << 25 |
                      (static_cast<uint32_t>(imm6) & 0x3F) << 20 |
                      static_cast<uint32_t>(rs1) << 15 |
                      static_cast<uint32_t>(funct3) << 12 |
                      static_cast<uint32_t>(rd) << 7 |
                      opcode;
  Emit(encoding);
}

void Riscv64Assembler::EmitB(uint16_t imm, GpuRegister rs2, GpuRegister rs1, int funct3, int opcode) {
  CHECK(IsUint<13>(imm)) << imm;
  uint32_t encoding = (static_cast<uint32_t>(imm)&0x1000) >> 12 << 31 |
                      (static_cast<uint32_t>(imm)&0x07E0) >> 5 << 25 |
                      static_cast<uint32_t>(rs2) << 20 |
                      static_cast<uint32_t>(rs1) << 15 |
                      static_cast<uint32_t>(funct3) << 12 |
                      (static_cast<uint32_t>(imm)&0x1E) >> 1 << 8 |
                      (static_cast<uint32_t>(imm)&0x0800) >> 11 << 7|
                      opcode;
  Emit(encoding);
}

void Riscv64Assembler::EmitU(uint32_t imm, GpuRegister rd, int opcode) {
  uint32_t encoding = static_cast<uint32_t>(imm) << 12 |
                      static_cast<uint32_t>(rd) << 7 |
                      opcode;
  Emit(encoding);
}

void Riscv64Assembler::EmitJ(uint32_t imm20, GpuRegister rd, int opcode) {
  CHECK(IsUint<21>(imm20)) << imm20;
  // Riscv JAL: J-Imm = (offset x 2), encode (imm20>>1) into instruction.
  uint32_t encoding = (static_cast<uint32_t>(imm20)&0x100000) >>20<< 31 |
                      (static_cast<uint32_t>(imm20)&0x07FE) >> 1 << 21 |
                      (static_cast<uint32_t>(imm20)&0x800) >> 11 << 20 |
                      (static_cast<uint32_t>(imm20)&0xFF000) >> 12 << 12 |
                      static_cast<uint32_t>(rd) << 7 |
                      opcode;
  Emit(encoding);
}

void Riscv64Assembler::Add(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x0, rs2, rs1, 0x0, rd, 0x33);
}

void Riscv64Assembler::Sub(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x20, rs2, rs1, 0x0, rd, 0x33);
}

void Riscv64Assembler::Sll(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x00, rs2, rs1, 0x01, rd, 0x33);
}

void Riscv64Assembler::Slt(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x00, rs2, rs1, 0x02, rd, 0x33);
}

void Riscv64Assembler::Sltu(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x00, rs2, rs1, 0x03, rd, 0x33);
}

void Riscv64Assembler::Xor(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x00, rs2, rs1, 0x04, rd, 0x33);
}

void Riscv64Assembler::Srl(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x00, rs2, rs1, 0x05, rd, 0x33);
}

void Riscv64Assembler::Sra(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x20, rs2, rs1, 0x05, rd, 0x33);
}

void Riscv64Assembler::Or(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x00, rs2, rs1, 0x06, rd, 0x33);
}

void Riscv64Assembler::And(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x00, rs2, rs1, 0x07, rd, 0x33);
}

// RV32I-I
void Riscv64Assembler::Jalr(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x0, rd, 0x67);
}

void Riscv64Assembler::Lb(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x0, rd, 0x03);
}

void Riscv64Assembler::Lh(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x1, rd, 0x03);
}

void Riscv64Assembler::Lw(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x2, rd, 0x03);
}

void Riscv64Assembler::Lbu(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x4, rd, 0x03);
}

void Riscv64Assembler::Lhu(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x5, rd, 0x03);
}

void Riscv64Assembler::Addi(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x0, rd, 0x13);
}

void Riscv64Assembler::Slti(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x2, rd, 0x13);
}

void Riscv64Assembler::Sltiu(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x3, rd, 0x13);
}

void Riscv64Assembler::Xori(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x4, rd, 0x13);
}

void Riscv64Assembler::Ori(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x6, rd, 0x13);
}

void Riscv64Assembler::Andi(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x7, rd, 0x13);
}

void Riscv64Assembler::Slli(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI6(0x0, offset, rs1, 0x1, rd, 0x13);
}

void Riscv64Assembler::Srli(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI6(0x0, offset, rs1, 0x5, rd, 0x13);
}

void Riscv64Assembler::Srai(GpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI6(0x20, offset, rs1, 0x5, rd, 0x13);
}

void Riscv64Assembler::Fence(uint8_t pred, uint8_t succ) {
  EmitI(0x0 << 8 | pred << 4 | succ, 0x0, 0x0, 0x0, 0xf);
}

void Riscv64Assembler::FenceI() {
  EmitI(0x0 << 6| 0x0 << 4 | 0x0, 0x0, 0x1, 0x0, 0xf);
}

void Riscv64Assembler::Ecall() {
  EmitI(0x0, 0x0, 0x0, 0x0, 0x73);
}

void Riscv64Assembler::Ebreak() {
  EmitI(0x1, 0x0, 0x0, 0x0, 0x73);
}

void Riscv64Assembler::Csrrw(GpuRegister rd, GpuRegister rs1, uint16_t csr) {
  EmitI(csr, rs1, 0x1, rd, 0x73);
}

void Riscv64Assembler::Csrrs(GpuRegister rd, GpuRegister rs1, uint16_t csr) {
  EmitI(csr, rs1, 0x2, rd, 0x73);
}

void Riscv64Assembler::Csrrc(GpuRegister rd, GpuRegister rs1, uint16_t csr) {
  EmitI(csr, rs1, 0x3, rd, 0x73);
}

void Riscv64Assembler::Csrrwi(GpuRegister rd, uint16_t csr, uint8_t zimm) {
  EmitI(csr, zimm, 0x5, rd, 0x73);
}

void Riscv64Assembler::Csrrsi(GpuRegister rd, uint16_t csr, uint8_t zimm) {
  EmitI(csr, zimm, 0x6, rd, 0x73);
}

void Riscv64Assembler::Csrrci(GpuRegister rd, uint16_t csr, uint8_t zimm) {
  EmitI(csr, zimm, 0x7, rd, 0x73);
}

// RV32I-S
void Riscv64Assembler::Sb(GpuRegister rs2, GpuRegister rs1, uint16_t offset) {
  EmitS(offset, rs2, rs1, 0x0, 0x23);
}

void Riscv64Assembler::Sh(GpuRegister rs2, GpuRegister rs1, uint16_t offset) {
  EmitS(offset, rs2, rs1, 0x1, 0x23);
}

void Riscv64Assembler::Sw(GpuRegister rs2, GpuRegister rs1, uint16_t offset) {
  EmitS(offset, rs2, rs1, 0x2, 0x23);
}

// RV32I-B
void Riscv64Assembler::Beq(GpuRegister rs1, GpuRegister rs2, uint16_t offset) {
  EmitB(offset, rs2, rs1, 0x0, 0x63);
}

void Riscv64Assembler::Bne(GpuRegister rs1, GpuRegister rs2, uint16_t offset) {
  EmitB(offset, rs2, rs1, 0x1, 0x63);
}

void Riscv64Assembler::Blt(GpuRegister rs1, GpuRegister rs2, uint16_t offset) {
  EmitB(offset, rs2, rs1, 0x4, 0x63);
}

void Riscv64Assembler::Bge(GpuRegister rs1, GpuRegister rs2, uint16_t offset) {
  EmitB(offset, rs2, rs1, 0x5, 0x63);
}

void Riscv64Assembler::Bltu(GpuRegister rs1, GpuRegister rs2, uint16_t offset) {
  EmitB(offset, rs2, rs1, 0x6, 0x63);
}

void Riscv64Assembler::Bgeu(GpuRegister rs1, GpuRegister rs2, uint16_t offset) {
  EmitB(offset, rs2, rs1, 0x7, 0x63);
}

// RV32I-U
void Riscv64Assembler::Lui(GpuRegister rd, uint32_t imm20) {
  EmitU(imm20, rd, 0x37);
}

void Riscv64Assembler::Auipc(GpuRegister rd, uint32_t imm20) {
  EmitU(imm20, rd, 0x17);
}

// RV32I-J
void Riscv64Assembler::Jal(GpuRegister rd, uint32_t imm20) {
  EmitJ(imm20, rd, 0x6F);
}

// RV64I-R
void Riscv64Assembler::Addw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x0, rs2, rs1, 0x0, rd, 0x3b);
}

void Riscv64Assembler::Subw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x20, rs2, rs1, 0x0, rd, 0x3b);
}

void Riscv64Assembler::Sllw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x0, rs2, rs1, 0x1, rd, 0x3b);
}

void Riscv64Assembler::Srlw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x0, rs2, rs1, 0x5, rd, 0x3b);
}

void Riscv64Assembler::Sraw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x20, rs2, rs1, 0x5, rd, 0x3b);
}

// RV64I-I
void Riscv64Assembler::Lwu(GpuRegister rd, GpuRegister rs1, uint16_t imm12) {
  EmitI(imm12, rs1, 0x6, rd, 0x3);
}

void Riscv64Assembler::Ld(GpuRegister rd, GpuRegister rs1, uint16_t imm12) {
  EmitI(imm12, rs1, 0x3, rd, 0x3);
}

void Riscv64Assembler::Addiw(GpuRegister rd, GpuRegister rs1, int16_t imm12) {
  CHECK(imm12 >= -2048) << imm12;
  CHECK(imm12 < 4096) << imm12;
  EmitI(imm12, rs1, 0x0, rd, 0x1b);
}

void Riscv64Assembler::Slliw(GpuRegister rd, GpuRegister rs1, int16_t shamt) {
  CHECK(static_cast<uint16_t>(shamt) < 32) << shamt;
  EmitR(0x0, shamt, rs1, 0x1, rd, 0x1b);  // borrow EmitR to implement this function
}

void Riscv64Assembler::Srliw(GpuRegister rd, GpuRegister rs1, int16_t shamt) {
  CHECK(static_cast<uint16_t>(shamt) < 32) << shamt;
  EmitR(0x0, shamt, rs1, 0x5, rd, 0x1b);  // borrow EmitR to implement this function
}

void Riscv64Assembler::Sraiw(GpuRegister rd, GpuRegister rs1, int16_t shamt) {
  CHECK(static_cast<uint16_t>(shamt) < 32) << shamt;
  EmitR(0x20, shamt, rs1, 0x5, rd, 0x1b);  // borrow EmitR to implement this function
}

// RV64I-S
void Riscv64Assembler::Sd(GpuRegister rs2, GpuRegister rs1, uint16_t imm12) {
  EmitS(imm12, rs2, rs1, 0x3, 0x23);
}

// RV32M-R
void Riscv64Assembler::Mul(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x1, rs2, rs1, 0x0, rd, 0x33);
}

void Riscv64Assembler::Mulh(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x1, rs2, rs1, 0x1, rd, 0x33);
}

void Riscv64Assembler::Mulhsu(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x1, rs2, rs1, 0x2, rd, 0x33);
}

void Riscv64Assembler::Mulhu(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x1, rs2, rs1, 0x3, rd, 0x33);
}

void Riscv64Assembler::Div(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x1, rs2, rs1, 0x4, rd, 0x33);
}

void Riscv64Assembler::Divu(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x1, rs2, rs1, 0x5, rd, 0x33);
}

void Riscv64Assembler::Rem(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x1, rs2, rs1, 0x6, rd, 0x33);
}

void Riscv64Assembler::Remu(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x1, rs2, rs1, 0x7, rd, 0x33);
}

// RV64M-R
void Riscv64Assembler::Mulw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x1, rs2, rs1, 0x0, rd, 0x3b);
}

void Riscv64Assembler::Divw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x1, rs2, rs1, 0x4, rd, 0x3b);
}

void Riscv64Assembler::Divuw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x1, rs2, rs1, 0x5, rd, 0x3b);
}

void Riscv64Assembler::Remw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x1, rs2, rs1, 0x6, rd, 0x3b);
}

void Riscv64Assembler::Remuw(GpuRegister rd, GpuRegister rs1, GpuRegister rs2) {
  EmitR(0x1, rs2, rs1, 0x7, rd, 0x3b);
}

// RV32A
void Riscv64Assembler::LrW(GpuRegister rd, GpuRegister rs1) {
  EmitR4(0x2, 0x0, 0x0, rs1, 0x2, rd, 0x2f);
}

void Riscv64Assembler::ScW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x3, 0x0, rs2, rs1, 0x2, rd, 0x2f);
}

void Riscv64Assembler::AmoSwapW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x1, 0x0, rs2, rs1, 0x2, rd, 0x2f);
}

void Riscv64Assembler::AmoAddW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x0, 0x0, rs2, rs1, 0x2, rd, 0x2f);
}

void Riscv64Assembler::AmoXorW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x4, 0x0, rs2, rs1, 0x2, rd, 0x2f);
}

void Riscv64Assembler::AmoAndW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0xc, 0x0, rs2, rs1, 0x2, rd, 0x2f);
}

void Riscv64Assembler::AmoOrW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x8, 0x0, rs2, rs1, 0x2, rd, 0x2f);
}

void Riscv64Assembler::AmoMinW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x10, 0x0, rs2, rs1, 0x2, rd, 0x2f);
}

void Riscv64Assembler::AmoMaxW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x14, 0x0, rs2, rs1, 0x2, rd, 0x2f);
}

void Riscv64Assembler::AmoMinuW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x18, 0x0, rs2, rs1, 0x2, rd, 0x2f);
}

void Riscv64Assembler::AmoMaxuW(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x1c, 0x0, rs2, rs1, 0x2, rd, 0x2f);
}

// RV64A
void Riscv64Assembler::LrD(GpuRegister rd, GpuRegister rs1) {
  EmitR4(0x2, 0x0, 0x0, rs1, 0x3, rd, 0x2f);
}

void Riscv64Assembler::ScD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x3, 0x0, rs2, rs1, 0x3, rd, 0x2f);
}

void Riscv64Assembler::AmoSwapD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x1, 0x0, rs2, rs1, 0x3, rd, 0x2f);
}

void Riscv64Assembler::AmoAddD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x0, 0x0, rs2, rs1, 0x3, rd, 0x2f);
}

void Riscv64Assembler::AmoXorD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x4, 0x0, rs2, rs1, 0x3, rd, 0x2f);
}

void Riscv64Assembler::AmoAndD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0xc, 0x0, rs2, rs1, 0x3, rd, 0x2f);
}

void Riscv64Assembler::AmoOrD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x8, 0x0, rs2, rs1, 0x3, rd, 0x2f);
}

void Riscv64Assembler::AmoMinD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x10, 0x0, rs2, rs1, 0x3, rd, 0x2f);
}

void Riscv64Assembler::AmoMaxD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x14, 0x0, rs2, rs1, 0x3, rd, 0x2f);
}

void Riscv64Assembler::AmoMinuD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x18, 0x0, rs2, rs1, 0x3, rd, 0x2f);
}

void Riscv64Assembler::AmoMaxuD(GpuRegister rd, GpuRegister rs2, GpuRegister rs1) {
  EmitR4(0x1c, 0x0, rs2, rs1, 0x3, rd, 0x2f);
}

// RV32F-I
void Riscv64Assembler::FLw(FpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x2, rd, 0x7);
}

// RV32F-S
void Riscv64Assembler::FSw(FpuRegister rs2, GpuRegister rs1, uint16_t offset) {
  EmitS(offset, rs2, rs1, 0x2, 0x27);
}

// RV32F-R
void Riscv64Assembler::FMAddS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3) {
  EmitR4(rs3, 0x0, rs2, rs1, FRM, rd, 0x43);
}

void Riscv64Assembler::FMSubS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3) {
  EmitR4(rs3, 0x0, rs2, rs1, FRM, rd, 0x47);
}

void Riscv64Assembler::FNMSubS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3) {
  EmitR4(rs3, 0x0, rs2, rs1, FRM, rd, 0x4b);
}

void Riscv64Assembler::FNMAddS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3) {
  EmitR4(rs3, 0x0, rs2, rs1, FRM, rd, 0x4f);
}

void Riscv64Assembler::FAddS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x0, rs2, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FSubS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x4, rs2, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FMulS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x8, rs2, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FDivS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0xc, rs2, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FSqrtS(FpuRegister rd, FpuRegister rs1) {
  EmitR(0x2c, 0x0, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FSgnjS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x10, rs2, rs1, 0x0, rd, 0x53);
}

void Riscv64Assembler::FSgnjnS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x10, rs2, rs1, 0x1, rd, 0x53);
}

void Riscv64Assembler::FSgnjxS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x10, rs2, rs1, 0x2, rd, 0x53);
}

void Riscv64Assembler::FMinS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x14, rs2, rs1, 0x0, rd, 0x53);
}

void Riscv64Assembler::FMaxS(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x14, rs2, rs1, 0x1, rd, 0x53);
}

void Riscv64Assembler::FCvtWS(GpuRegister rd, FpuRegister rs1, FPRoundingMode frm) {
  EmitR(0x60, 0x0, rs1, frm, rd, 0x53);
}

void Riscv64Assembler::FCvtWuS(GpuRegister rd, FpuRegister rs1) {
  EmitR(0x60, 0x1, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FMvXW(GpuRegister rd, FpuRegister rs1) {
  EmitR(0x70, 0x0, rs1, 0x0, rd, 0x53);
}

void Riscv64Assembler::FEqS(GpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x50, rs2, rs1, 0x2, rd, 0x53);
}

void Riscv64Assembler::FLtS(GpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x50, rs2, rs1, 0x1, rd, 0x53);
}

void Riscv64Assembler::FLeS(GpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x50, rs2, rs1, 0x0, rd, 0x53);
}

void Riscv64Assembler::FClassS(GpuRegister rd, FpuRegister rs1) {
  EmitR(0x70, 0x0, rs1, 0x1, rd, 0x53);
}

void Riscv64Assembler::FCvtSW(FpuRegister rd, GpuRegister rs1) {
  EmitR(0x68, 0x0, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FCvtSWu(FpuRegister rd, GpuRegister rs1) {
  EmitR(0x68, 0x1, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FMvWX(FpuRegister rd, GpuRegister rs1) {
  EmitR(0x78, 0x0, rs1, 0x0, rd, 0x53);
}

// RV64F-R
void Riscv64Assembler::FCvtLS(GpuRegister rd, FpuRegister rs1, FPRoundingMode frm) {
  EmitR(0x60, 0x2, rs1, frm, rd, 0x53);
}

void Riscv64Assembler::FCvtLuS(GpuRegister rd, FpuRegister rs1) {
  EmitR(0x60, 0x3, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FCvtSL(FpuRegister rd, GpuRegister rs1) {
  EmitR(0x68, 0x2, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FCvtSLu(FpuRegister rd, GpuRegister rs1) {
  EmitR(0x68, 0x3, rs1, FRM, rd, 0x53);
}

// RV32D-I
void Riscv64Assembler::FLd(FpuRegister rd, GpuRegister rs1, uint16_t offset) {
  EmitI(offset, rs1, 0x3, rd, 0x7);
}

// RV32D-S
void Riscv64Assembler::FSd(FpuRegister rs2, GpuRegister rs1, uint16_t offset) {
  EmitS(offset, rs2, rs1, 0x3, 0x27);
}

// RV32D-R
void Riscv64Assembler::FMAddD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3) {
  EmitR4(rs3, 0x1, rs2, rs1, FRM, rd, 0x43);
}

void Riscv64Assembler::FMSubD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3) {
  EmitR4(rs3, 0x1, rs2, rs1, FRM, rd, 0x47);
}

void Riscv64Assembler::FNMSubD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3) {
  EmitR4(rs3, 0x1, rs2, rs1, FRM, rd, 0x4b);
}

void Riscv64Assembler::FNMAddD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2, FpuRegister rs3) {
  EmitR4(rs3, 0x1, rs2, rs1, FRM, rd, 0x4f);
}

void Riscv64Assembler::FAddD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x1, rs2, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FSubD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x5, rs2, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FMulD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x9, rs2, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FDivD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0xd, rs2, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FSqrtD(FpuRegister rd, FpuRegister rs1) {
  EmitR(0x2d, 0x0, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FSgnjD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x11, rs2, rs1, 0x0, rd, 0x53);
}

void Riscv64Assembler::FSgnjnD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x11, rs2, rs1, 0x1, rd, 0x53);
}

void Riscv64Assembler::FSgnjxD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x11, rs2, rs1, 0x2, rd, 0x53);
}

void Riscv64Assembler::FMinD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x15, rs2, rs1, 0x0, rd, 0x53);
}

void Riscv64Assembler::FMaxD(FpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x15, rs2, rs1, 0x1, rd, 0x53);
}

void Riscv64Assembler::FCvtSD(FpuRegister rd, FpuRegister rs1) {
  EmitR(0x20, 0x1, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FCvtDS(FpuRegister rd, FpuRegister rs1) {
  // EmitR(0x21, 0x0, rs1, FRM, rd, 0x53);
  EmitR(0x21, 0x0, rs1, 0x0, rd, 0x53);  // TODO need confirm:FRM=0x0 gived by gcc compiler, why?
}

void Riscv64Assembler::FEqD(GpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x51, rs2, rs1, 0x2, rd, 0x53);
}

void Riscv64Assembler::FLtD(GpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x51, rs2, rs1, 0x1, rd, 0x53);
}

void Riscv64Assembler::FLeD(GpuRegister rd, FpuRegister rs1, FpuRegister rs2) {
  EmitR(0x51, rs2, rs1, 0x0, rd, 0x53);
}

void Riscv64Assembler::FClassD(GpuRegister rd, FpuRegister rs1) {
  EmitR(0x71, 0x0, rs1, 0x1, rd, 0x53);
}

void Riscv64Assembler::FCvtWD(GpuRegister rd, FpuRegister rs1, FPRoundingMode frm) {
  EmitR(0x61, 0x0, rs1, frm, rd, 0x53);
}

void Riscv64Assembler::FCvtWuD(GpuRegister rd, FpuRegister rs1) {
  EmitR(0x61, 0x1, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FCvtDW(FpuRegister rd, GpuRegister rs1) {
  // EmitR(0x69, 0x0, rs1, FRM, rd, 0x53);// TODO confirm FRM='000' or '111'
  EmitR(0x69, 0x0, rs1, 0x0, rd, 0x53);
}

void Riscv64Assembler::FCvtDWu(FpuRegister rd, GpuRegister rs1) {
  // EmitR(0x69, 0x1, rs1, FRM, rd, 0x53);// TODO confirm FRM='000' or '111'
  EmitR(0x69, 0x1, rs1, 0x0, rd, 0x53);
}

// RV64D-R
void Riscv64Assembler::FCvtLD(GpuRegister rd, FpuRegister rs1, FPRoundingMode frm) {
  EmitR(0x61, 0x2, rs1, frm, rd, 0x53);
}

void Riscv64Assembler::FCvtLuD(GpuRegister rd, FpuRegister rs1) {
  EmitR(0x61, 0x3, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FMvXD(GpuRegister rd, FpuRegister rs1) {
  EmitR(0x71, 0x0, rs1, 0x0, rd, 0x53);
}

void Riscv64Assembler::FCvtDL(FpuRegister rd, GpuRegister rs1) {
  EmitR(0x69, 0x2, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FCvtDLu(FpuRegister rd, GpuRegister rs1) {
  EmitR(0x69, 0x3, rs1, FRM, rd, 0x53);
}

void Riscv64Assembler::FMvDX(FpuRegister rd, GpuRegister rs1) {
  EmitR(0x79, 0x0, rs1, 0x0, rd, 0x53);
}

}  // namespace riscv64
}  // namespace art
