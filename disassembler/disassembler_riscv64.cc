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

#include "disassembler_riscv64.h"

#include <ostream>
#include <sstream>
#include <memory>
#include <iostream>
#include <tuple>

#include "android-base/logging.h"
#include "android-base/stringprintf.h"

#include "base/bit_utils.h"

using android::base::StringPrintf;

namespace art {
namespace riscv64 {

enum GpuReg{};
enum FpuReg{};
enum CSReg{};
enum Ordering{};

template<typename Tuple, typename Func, size_t N>
struct __tuple_processor {
  inline static void __tuple_process(Tuple &t, size_t size, Func& f) {
    __tuple_processor<Tuple, Func, N - 1>::__tuple_process(t, size, f);
    f(size, N - 1, std::get<N - 1>(t));
  }
};

template<typename Tuple, typename Func>
struct __tuple_processor<Tuple, Func, 1> {
  inline static void __tuple_process(Tuple &t, size_t size, Func& f) {
    f(size, 0, std::get<0>(t));
  }
};

template<typename Tuple, typename Func>
struct __tuple_processor<Tuple, Func, 0> {
  inline static void __tuple_process(Tuple &, size_t, Func& ) {
  }
};

template<typename... Args, typename Func>
void tuple_for_each(std::tuple<Args...> &t, Func &&f) {
  __tuple_processor<decltype(t), Func, sizeof...(Args)>::__tuple_process(t, sizeof...(Args), f);
}

template<typename Type>
std::string ToString(Type v) {
  std::stringstream ss;
  ss << v;
  return ss.str();
}

std::string ToString(GpuReg id) {
  static const char* names[32] = {
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "fp", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"
  };
  return names[id];
}

std::string ToString(FpuReg id) {
  static const char* names[32] = {
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
    "fs0", "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5",
    "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7",
    "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11"
  };
  return names[id];
}

std::string ToString(CSReg id) {
  switch (static_cast<uint32_t>(id)) {
  case 0x000: return "ustatus";
  case 0x001: return "fflags";
  case 0x002: return "frm";
  case 0x003: return "fcsr";
  case 0x004: return "uie";
  case 0x005: return "utvec";
  case 0x040: return "uscratch";
  case 0x041: return "uepc";
  case 0x042: return "ucause";
  case 0x043: return "utval";
  case 0x044: return "uip";
  case 0xC00: return "cycle";
  case 0xC01: return "time";
  case 0xC02: return "instret";
  default:
    return StringPrintf("0x%x", id);
  }
}

std::string ToString(Ordering order) {
  std::stringstream ss;
  if ((0xF & order) == 0) {
    ss << "unknown";
  } else {
    if ((0x8 & order) != 0) ss << "i";
    if ((0x4 & order) != 0) ss << "o";
    if ((0x2 & order) != 0) ss << "r";
    if ((0x1 & order) != 0) ss << "w";
  }
  return ss.str();
}

template<typename Type>
bool IsReg(Type) { return false; }

template<>
bool IsReg(GpuReg) { return true; }

struct InstructionInfo {
  uint32_t mask;
  uint32_t value;
  const char* name;
  // const char* type;
  std::string type;
};

struct Instruction {
  explicit Instruction(const char* name) : name_(name) {}
  virtual ~Instruction() {}

  std::string Dump(uint32_t data) {
    DecodeFields(data);

    std::stringstream ss;
    ss << this->name_;

    std::string fields_str = DumpFields(data);
    if (!fields_str.empty()) {
      ss << " " << fields_str;
    }

    std::string comment_str = DumpComments(data);
    if (!comment_str.empty()) {
      ss << ";" << comment_str;
    }

    return ss.str();
  }

  virtual void DecodeFields(uint32_t data ATTRIBUTE_UNUSED) { }
  virtual std::string DumpFields(uint32_t data ATTRIBUTE_UNUSED) { return ""; }
  virtual std::string DumpComments(uint32_t data ATTRIBUTE_UNUSED) { return ""; }

  const char* name_;
};

template<typename _Type, uint32_t bits, uint32_t shift, uint32_t position = 0>
struct InstructionField {
  using Type = _Type;

  void Decode(uint32_t encoding) {
    uint32_t mask = 0xFFFFFFFF >> (32-bits) << shift;
    int32_t sign_extend = static_cast<int32_t>(encoding & mask) >> shift << position;
    value = static_cast<Type>(sign_extend);
  }

  Type value;
};

template<typename _Type, typename ..._SubFields>
struct InstructionFieldComp {
  using Type = _Type;
  using SubFields = std::tuple<_SubFields...>;

  void Decode(uint32_t encoding) {
    tuple_for_each(fields, [this, &encoding](size_t, size_t, auto& field) {
      field.Decode(encoding);
      value = value | field.value;
    });
  }

  Type value = 0;
  SubFields fields;
};

template<bool address_mode = false, typename ...Fields>
struct InstructionBase : public Instruction {
  using Instruction::Instruction;
  using FieldTuple = std::tuple<Fields...>;

  void DecodeFields(uint32_t data) override {
    tuple_for_each(fields, [&](size_t, size_t, auto& field) { field.Decode(data); });
  }

  std::string DumpFields(uint32_t data ATTRIBUTE_UNUSED) override {
    std::stringstream ss;
    tuple_for_each(fields, [&](size_t size, size_t i, auto& field) {
      ss << ToString(field.value);

      if (i == size - 2 && address_mode) {
        if (IsReg(field.value)) {
          ss << ",(";
        } else {
          ss << "(";
        }
      } else if (i == size - 1 && address_mode) {
        ss << ")";
      } else if (i < size - 1) {
        ss << ",";
      }
    });
    return ss.str();
  }

  FieldTuple fields;
};

typedef InstructionField<GpuReg, 5, 7> RD;
typedef InstructionField<GpuReg, 5, 15> RS1;
typedef InstructionField<GpuReg, 5, 20> RS2;
typedef InstructionField<FpuReg, 5, 7> FRD;
typedef InstructionField<FpuReg, 5, 15> FRS1;
typedef InstructionField<FpuReg, 5, 20> FRS2;
typedef InstructionField<FpuReg, 5, 27> FRS3;
typedef InstructionField<CSReg, 12, 20> CSR;
typedef InstructionField<int32_t, 12, 20> II;  // I type instruction, signed imm12
typedef InstructionField<uint32_t, 12, 20> IU;  // I type instruction, unsigned imm12
typedef InstructionField<uint32_t, 5, 20> Shamt;
typedef InstructionField<int32_t, 20, 12> UI;

// Naming rule: Instruction[print-mode][instruction-type][operand-type]
// print-mode         G:general, A:address likely
// instruction-type   R/I/S/B/U/J
// operand-type       r:general register, f:float point register, i: immediate
typedef InstructionBase<false, RD, RS1, RS2> InstructionRrrr;
typedef InstructionBase<true, RD, RS1> InstructionARrr;
typedef InstructionBase<true, RD, RS2, RS1> InstructionARrrr;
typedef InstructionBase<false, FRD, FRS1> InstructionRff;
typedef InstructionBase<false, RD, FRS1> InstructionRrf;
typedef InstructionBase<false, FRD, RS1> InstructionRfr;
typedef InstructionBase<false, FRD, FRS1, FRS2> InstructionRfff;
typedef InstructionBase<false, RD, FRS1, FRS2> InstructionRrff;
typedef InstructionBase<false, FRD, FRS1, FRS2, FRS3> InstructionRffff;

typedef InstructionBase<false, RD, RS1, II> InstructionIrri;
typedef InstructionBase<false, RD, II, RS1> InstructionIrir;
typedef InstructionBase<true, RD, II, RS1> InstructionAIrir;
typedef InstructionBase<true, FRD, II, RS1> InstructionAIfir;
typedef InstructionBase<false, RD, RS1, IU> InstructionIrru;
typedef InstructionBase<false, RD, InstructionField<CSReg, 12, 20>,
    InstructionField<uint16_t, 5, 15>> InstructionIrii;
typedef InstructionBase<false, InstructionField<Ordering, 4, 24>,
    InstructionField<Ordering, 4, 20>> InstructionIii;
typedef InstructionBase<false> InstructionI0;

typedef InstructionBase<true, RS2, InstructionFieldComp<int16_t,
    InstructionField<uint32_t, 5, 7, 0>,
    InstructionField<int32_t, 7, 25, 5>>, RS1> InstructionS;

typedef InstructionBase<true, FRS2, InstructionFieldComp<int16_t,
    InstructionField<uint32_t, 5, 7, 0>,
    InstructionField<int32_t, 7, 25, 5>>, RS1> InstructionSfir;

typedef InstructionBase<false, RS1, RS2, InstructionFieldComp<int16_t,
    InstructionField<uint32_t, 1, 7, 11>,
    InstructionField<uint32_t, 4, 8, 1>,
    InstructionField<uint32_t, 6, 25, 5>,
    InstructionField<int32_t, 1, 31, 12>>> InstructionB;
typedef InstructionBase<false, RD, RS1, Shamt> InstructionRrri;
typedef InstructionBase<false, RD, UI> InstructionU;
typedef InstructionBase<false, RD, InstructionFieldComp<int32_t,
    InstructionField<uint32_t, 8, 12, 12>,
    InstructionField<uint32_t, 1, 20, 11>,
    InstructionField<uint32_t, 10, 21, 1>,
    InstructionField<int32_t, 1, 31, 20>>> InstructionJ;

std::unique_ptr<Instruction> CreateInstruction(InstructionInfo* info) {
  if (info->type == "Rrrr") {
    return std::make_unique<InstructionRrrr>(info->name);
  } else if (info->type == "ARrr") {
    return std::make_unique<InstructionARrr>(info->name);
  } else if (info->type == "ARrrr") {
    return std::make_unique<InstructionARrrr>(info->name);
  } else if (info->type == "Rff") {
    return std::make_unique<InstructionRff>(info->name);
  } else if (info->type == "Rrf") {
    return std::make_unique<InstructionRrf>(info->name);
  } else if (info->type == "Rfr") {
    return std::make_unique<InstructionRfr>(info->name);
  } else if (info->type == "Rfff") {
    return std::make_unique<InstructionRfff>(info->name);
  } else if (info->type == "Rrff") {
    return std::make_unique<InstructionRrff>(info->name);
  } else if (info->type == "Rffff") {
    return std::make_unique<InstructionRffff>(info->name);
  } else if (info->type == "Rrri") {
    return std::make_unique<InstructionRrri>(info->name);
  } else if (info->type == "AIrir") {
    return std::make_unique<InstructionAIrir>(info->name);
  } else if (info->type == "AIfir") {
    return std::make_unique<InstructionAIfir>(info->name);
  } else if (info->type == "Irri") {
    return std::make_unique<InstructionIrri>(info->name);
  } else if (info->type == "Irii") {
    return std::make_unique<InstructionIrii>(info->name);
  } else if (info->type == "Irir") {
    return std::make_unique<InstructionIrir>(info->name);
  } else if (info->type == "Iii") {
    return std::make_unique<InstructionIii>(info->name);
  } else if (info->type == "I0") {
    return std::make_unique<InstructionI0>(info->name);
  } else if (info->type == "S") {
    return std::make_unique<InstructionS>(info->name);
  } else if (info->type == "Sfir") {
    return std::make_unique<InstructionSfir>(info->name);
  } else if (info->type == "B") {
    return std::make_unique<InstructionB>(info->name);
  } else if (info->type == "U") {
    return std::make_unique<InstructionU>(info->name);
  } else if (info->type == "J") {
    return std::make_unique<InstructionJ>(info->name);
  }
  return nullptr;
}

#define COMP_OPCODE_R(opcode, funct3, funct7)   \
    (static_cast<uint32_t>(funct7) << 25 | static_cast<uint32_t>(funct3) << 12 | opcode)
#define COMP_OPCODE_R5(opcode, funct3, funct5)   \
    (static_cast<uint32_t>(funct5) << 27 | static_cast<uint32_t>(funct3) << 12 | opcode)
#define COMP_OPCODE_R4(opcode, funct3, funct2)   \
    (static_cast<uint32_t>(funct2) << 25 | static_cast<uint32_t>(funct3) << 12 | opcode)
#define COMP_OPCODE_R2(opcode, funct3, funct5, funct7)   \
    (static_cast<uint32_t>(funct7) << 25 | static_cast<uint32_t>(funct5) << 20 \
    |static_cast<uint32_t>(funct3) << 12 | opcode)
#define COMP_OPCODE_S(opcode, funct3)   \
    (static_cast<uint32_t>(funct3) << 12 | opcode)
#define COMP_OPCODE_B(opcode, funct3)   COMP_OPCODE_S(opcode, funct3)
#define COMP_OPCODE_I(opcode, funct3)   COMP_OPCODE_S(opcode, funct3)
#define COMP_OPCODE_IS(opcode, funct3, funct12)   \
    (static_cast<uint32_t>(funct12) << 20 | static_cast<uint32_t>(funct3) << 12 | opcode)
#define COMP_OPCODE_U(opcode)   (opcode)
#define COMP_OPCODE_J(opcode)   COMP_OPCODE_U(opcode)

// static const uint32_t kOpcodeShift = 26;
// static const uint32_t kMsaSpecialMask = (0x3f << kOpcodeShift);

static const uint32_t kRTypeMask = (uint32_t)0x7f << 25 | (uint32_t)0x7 << 12 | 0x7f;
static const uint32_t kR5TypeMask = (uint32_t)0x1f << 27 | (uint32_t)0x7 << 12 | 0x7f;
static const uint32_t kR4TypeMask = (uint32_t)0x03 << 25 | (uint32_t)0x7 << 12 | 0x7f;
static const uint32_t kR2TypeMask = (uint32_t)0x7f << 25 | (uint32_t)0x1f << 20 | (uint32_t)0x7 << 12 | 0x7f;
static const uint32_t kSTypeMask = (uint32_t)0x7 << 12 | 0x7f;
static const uint32_t kBTypeMask = kSTypeMask;
static const uint32_t kITypeMask = kSTypeMask;
static const uint32_t kISTypeMask = (uint32_t)0xFFF << 20 | (uint32_t)0x7 << 12 | 0x7f;
static const uint32_t kUTypeMask = 0x7f;
static const uint32_t kJTypeMask = kUTypeMask;

static const uint32_t kFRM = 0x7;

InstructionInfo gInstructions[] = {
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x0,  0x0), "add", "Rrrr"},  // RV32I-R
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x0, 0x20), "sub", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x1,  0x0), "sll", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x2,  0x0), "slt", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x3,  0x0), "sltu", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x4,  0x0), "xor", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x5,  0x0), "srl", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x5, 0x20), "sra", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x6,  0x0), "or", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x7,  0x0), "and", "Rrrr"},
  {kSTypeMask, COMP_OPCODE_S(0x23,  0x0), "sb", "S"},  // RV32I-S
  {kSTypeMask, COMP_OPCODE_S(0x23,  0x1), "sh", "S"},
  {kSTypeMask, COMP_OPCODE_S(0x23,  0x2), "sw", "S"},
  {kBTypeMask, COMP_OPCODE_B(0x63,  0x0), "beq", "B"},  // RV32I-B
  {kBTypeMask, COMP_OPCODE_B(0x63,  0x1), "bne", "B"},
  {kBTypeMask, COMP_OPCODE_B(0x63,  0x4), "blt", "B"},
  {kBTypeMask, COMP_OPCODE_B(0x63,  0x5), "bge", "B"},
  {kBTypeMask, COMP_OPCODE_B(0x63,  0x6), "bltu", "B"},
  {kBTypeMask, COMP_OPCODE_B(0x63,  0x7), "bgeu", "B"},
  {kITypeMask, COMP_OPCODE_I(0x67,  0x0), "jalr", "AIrir"},  // RV32I-I
  {kITypeMask, COMP_OPCODE_I(0x03,  0x0), "lb", "AIrir"},
  {kITypeMask, COMP_OPCODE_I(0x03,  0x1), "lh", "AIrir"},
  {kITypeMask, COMP_OPCODE_I(0x03,  0x2), "lw", "AIrir"},
  {kITypeMask, COMP_OPCODE_I(0x03,  0x4), "lbu", "AIrir"},
  {kITypeMask, COMP_OPCODE_I(0x03,  0x5), "lhu", "AIrir"},
  {kITypeMask, COMP_OPCODE_I(0x13,  0x0), "addi", "Irri"},
  {kITypeMask, COMP_OPCODE_I(0x13,  0x2), "slti", "Irri"},
  {kITypeMask, COMP_OPCODE_I(0x13,  0x3), "sltiu", "Irri"},
  {kITypeMask, COMP_OPCODE_I(0x13,  0x4), "xori", "Irri"},
  {kITypeMask, COMP_OPCODE_I(0x13,  0x6), "ori", "Irri"},
  {kITypeMask, COMP_OPCODE_I(0x13,  0x7), "andi", "Irri"},
  {kITypeMask, COMP_OPCODE_I(0x0f,  0x0), "fence", "Iii"},
  {kITypeMask, COMP_OPCODE_I(0x0f,  0x1), "fence.i", "I0"},
  {kISTypeMask, COMP_OPCODE_IS(0x73, 0x0, 0x0), "ecall", "I0"},
  {kISTypeMask, COMP_OPCODE_IS(0x73, 0x0, 0x1), "ebreak", "I0"},
  {kITypeMask, COMP_OPCODE_I(0x73,  0x1), "csrrw", "Irir"},  // @todo need confirm
  {kITypeMask, COMP_OPCODE_I(0x73,  0x2), "csrrs", "Irir"},  // @todo need confirm
  {kITypeMask, COMP_OPCODE_I(0x73,  0x3), "csrrc", "Irir"},  // @todo need confirm
  {kITypeMask, COMP_OPCODE_I(0x73,  0x5), "csrrwi", "Irii"},
  {kITypeMask, COMP_OPCODE_I(0x73,  0x6), "csrrsi", "Irii"},
  {kITypeMask, COMP_OPCODE_I(0x73,  0x7), "csrrci", "Irii"},
  {kRTypeMask, COMP_OPCODE_R(0x13,  0x1,  0x0), "slli", "Rrri"},
  {kRTypeMask, COMP_OPCODE_R(0x13,  0x5,  0x0), "srli", "Rrri"},
  {kRTypeMask, COMP_OPCODE_R(0x13,  0x5, 0x20), "srai", "Rrri"},
  {kUTypeMask, COMP_OPCODE_U(0x37), "lui", "U"},  // RV32I-U
  {kUTypeMask, COMP_OPCODE_U(0x17), "auipc", "U"},
  {kJTypeMask, COMP_OPCODE_J(0x6f), "jal", "J"},
  {kITypeMask, COMP_OPCODE_I(0x03,  0x6), "lwu", "AIrir"},  // RV64I-I
  {kITypeMask, COMP_OPCODE_I(0x03,  0x3), "ld", "AIrir"},
  {kITypeMask, COMP_OPCODE_I(0x1b,  0x0), "addiw", "Irri"},
  {kSTypeMask, COMP_OPCODE_S(0x23,  0x3), "sd", "S"},  // RV64I-S
  {kRTypeMask, COMP_OPCODE_R(0x1b,  0x1,  0x0), "slliw", "Rrri"},  // RV64I-R
  {kRTypeMask, COMP_OPCODE_R(0x1b,  0x5,  0x0), "srliw", "Rrri"},
  {kRTypeMask, COMP_OPCODE_R(0x1b,  0x5, 0x20), "sraiw", "Rrri"},
  {kRTypeMask, COMP_OPCODE_R(0x3b,  0x0,  0x0), "addw", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x3b,  0x0, 0x20), "subw", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x3b,  0x1,  0x0), "sllw", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x3b,  0x5,  0x0), "srlw", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x3b,  0x5, 0x20), "sraw", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x0,  0x1), "mul", "Rrrr"},  // RV32M-R
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x1,  0x1), "mulh", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x2,  0x1), "mulhsu", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x3,  0x1), "mulhu", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x4,  0x1), "div", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x5,  0x1), "divu", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x6,  0x1), "rem", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x33,  0x7,  0x1), "remu", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x3b,  0x0,  0x1), "mulw", "Rrrr"},  // RV64M-R
  {kRTypeMask, COMP_OPCODE_R(0x3b,  0x4,  0x1), "divw", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x3b,  0x5,  0x1), "divuw", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x3b,  0x6,  0x1), "remw", "Rrrr"},
  {kRTypeMask, COMP_OPCODE_R(0x3b,  0x7,  0x1), "remuw", "Rrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x2,  0x2), "lr.w", "ARrr"},  // RV32A-R
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x2,  0x3), "sc.w", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x2,  0x1), "amoswap.w", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x2,  0x0), "amoadd.w", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x2,  0x4), "amoxor.w", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x2,  0xc), "amoand.w", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x2,  0x8), "amoor.w", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x2, 0x10), "amomin.w", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x2, 0x14), "amomax.w", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x2, 0x18), "amominu.w", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x2, 0x1c), "amomaxu.w", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x3,  0x2), "lr.d", "ARrr"},  // RV64A-R
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x3,  0x3), "sc.d", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x3,  0x1), "amoswap.d", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x3,  0x0), "amoadd.d", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x3,  0x4), "amoxor.d", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x3,  0xc), "amoand.d", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x3,  0x8), "amoor.d", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x3, 0x10), "amomin.d", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x3, 0x14), "amomax.d", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x3, 0x18), "amominu.d", "ARrrr"},
  {kR5TypeMask, COMP_OPCODE_R5(0x2f,  0x3, 0x1c), "amomaxu.d", "ARrrr"},
  {kITypeMask, COMP_OPCODE_I(0x07,  0x2), "flw", "AIfir"},  // RV32F-I
  {kSTypeMask, COMP_OPCODE_S(0x27,  0x2), "fsw", "Sfir"},  // RV32F-S
  {kR4TypeMask, COMP_OPCODE_R4(0x43, kFRM,  0x0), "fmadd.s", "Rffff"},  // RV32F-R
  {kR4TypeMask, COMP_OPCODE_R4(0x47, kFRM,  0x0), "fmsub.s", "Rffff"},
  {kR4TypeMask, COMP_OPCODE_R4(0x4b, kFRM,  0x0), "fnmsub.s", "Rffff"},
  {kR4TypeMask, COMP_OPCODE_R4(0x4f, kFRM,  0x0), "fnmadd.s", "Rffff"},
  {kRTypeMask, COMP_OPCODE_R(0x53, kFRM,  0x0), "fadd.s", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53, kFRM,  0x4), "fsub.s", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53, kFRM,  0x8), "fmul.s", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53, kFRM,  0xc), "fdiv.s", "Rfff"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM,  0x0, 0x2c), "fsqrt.s", "Rff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x0, 0x10), "fsgnj.s", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x1, 0x10), "fsgnjn.s", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x2, 0x10), "fsgnjx.s", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x0, 0x14), "fmin.s", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x1, 0x14), "fmax.s", "Rfff"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM,  0x0, 0x60), "fcvt.w.s", "Rrf"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM,  0x1, 0x60), "fcvt.wu.s", "Rrf"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, 0x00,  0x0, 0x70), "fmv.x.w", "Rrf"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x2, 0x50), "feq.s", "Rrff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x1, 0x50), "flt.s", "Rrff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x0, 0x50), "fle.s", "Rrff"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, 0x01, 0x0, 0x70), "fclass.s", "Rrf"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM, 0x0, 0x68), "fcvt.s.w", "Rfr"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM, 0x1, 0x68), "fcvt.s.wu", "Rfr"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, 0x00, 0x0, 0x78), "fmv.w.x", "Rfr"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM,  0x2, 0x60), "fcvt.l.s", "Rrf"},  // RV64F-R
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM,  0x3, 0x60), "fcvt.lu.s", "Rrf"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM, 0x2, 0x68), "fcvt.s.l", "Rfr"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM, 0x3, 0x68), "fcvt.s.lu", "Rfr"},
  {kITypeMask, COMP_OPCODE_I(0x07,  0x3), "fld", "AIfir"},  // RV32D-I
  {kSTypeMask, COMP_OPCODE_S(0x27,  0x3), "fsd", "Sfir"},  // RV32D-S
  {kR4TypeMask, COMP_OPCODE_R4(0x43, kFRM,  0x1), "fmadd.d", "Rffff"},  // RV32D-R
  {kR4TypeMask, COMP_OPCODE_R4(0x47, kFRM,  0x1), "fmsub.d", "Rffff"},
  {kR4TypeMask, COMP_OPCODE_R4(0x4b, kFRM,  0x1), "fnmsub.d", "Rffff"},
  {kR4TypeMask, COMP_OPCODE_R4(0x4f, kFRM,  0x1), "fnmadd.d", "Rffff"},
  {kRTypeMask, COMP_OPCODE_R(0x53, kFRM,  0x1), "fadd.d", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53, kFRM,  0x5), "fsub.d", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53, kFRM,  0x9), "fmul.d", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53, kFRM,  0xd), "fdiv.d", "Rfff"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM,  0x0, 0x2d), "fsqrt.d", "Rff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x0, 0x11), "fsgnj.d", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x1, 0x11), "fsgnjn.d", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x2, 0x11), "fsgnjx.d", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x0, 0x15), "fmin.d", "Rfff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x1, 0x15), "fmax.d", "Rfff"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM,  0x1, 0x20), "fcvt.s.d", "Rff"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, 0x00,  0x0, 0x21), "fcvt.d.s", "Rff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x2, 0x51), "feq.d", "Rrff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x1, 0x51), "flt.d", "Rrff"},
  {kRTypeMask, COMP_OPCODE_R(0x53,  0x0, 0x51), "fle.d", "Rrff"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, 0x01, 0x0, 0x71), "fclass.d", "Rrf"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM,  0x0, 0x61), "fcvt.w.d", "Rrf"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM,  0x1, 0x61), "fcvt.wu.d", "Rrf"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, 0x00, 0x0, 0x69), "fcvt.d.w", "Rfr"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, 0x00, 0x1, 0x69), "fcvt.d.wu", "Rfr"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM,  0x2, 0x61), "fcvt.l.d", "Rrf"},  // RV64D-R
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM,  0x3, 0x61), "fcvt.lu.d", "Rrf"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, 0x00, 0x0, 0x71), "fmv.x.d", "Rrf"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM, 0x2, 0x69), "fcvt.d.l", "Rfr"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, kFRM, 0x3, 0x69), "fcvt.d.lu", "Rfr"},
  {kR2TypeMask, COMP_OPCODE_R2(0x53, 0x00, 0x0, 0x79), "fmv.d.x", "Rfr"},
};

static uint32_t ReadU32(const uint8_t* ptr) {
  // We only support little-endian RISCV64.
  return ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
}

std::string DisassemblerRiscv64::DumpInstruction(uint32_t encoding) {
  uint32_t op = encoding & 0x7f;
  std::string ass_instructon = StringPrintf("op=%x", op);

  for (size_t iinst = 0; iinst < arraysize(gInstructions); iinst++) {
    InstructionInfo& inst_info = gInstructions[iinst];
    uint32_t opcode = inst_info.mask & encoding;
    if (opcode == inst_info.value) {
      std::unique_ptr<Instruction> inst = CreateInstruction(&inst_info);
      if (!inst) {
        // std::cout << "not handle Instruction type :"<< inst_info.name << std::endl;
        break;
      }

      ass_instructon = inst->Dump(encoding);
      int32_t base = 0;
      int32_t offset = 0;

      if (inst_info.type == "AIrir") {
        InstructionAIrir* ptr = reinterpret_cast<InstructionAIrir*>(inst.get());
        offset = std::get<1>(ptr->fields).value;
        base = std::get<2>(ptr->fields).value;
      } else if (inst_info.type == "S") {
        InstructionS* ptr = reinterpret_cast<InstructionS*>(inst.get());
        offset = std::get<1>(ptr->fields).value;
        base = std::get<2>(ptr->fields).value;
      }

      if ((inst_info.type == "AIrir" || inst_info.type == "S") && base == 9) {
        std::stringstream ss;
        GetDisassemblerOptions()->thread_offset_name_function_(ss, offset);
        ass_instructon += "  ; ";
        ass_instructon += ss.str();
      }
      break;
    }
  }
  return ass_instructon;
}

size_t DisassemblerRiscv64::Dump(std::ostream& os, const uint8_t* instr_ptr) {
  uint32_t instruction = ReadU32(instr_ptr);
  std::string ass_instruction = DumpInstruction(instruction);

  os << FormatInstructionPointer(instr_ptr)
      << StringPrintf(": %08x\t%-7s ", instruction, ass_instruction.c_str())
      << '\n';

  last_ptr_ = instr_ptr;
  last_instr_ = instruction;
  return 4;
}

void DisassemblerRiscv64::Dump(std::ostream& os, const uint8_t* begin, const uint8_t* end) {
  for (const uint8_t* cur = begin; cur < end; cur += 4) {
    Dump(os, cur);
  }
}

}  // namespace riscv64
}  // namespace art
