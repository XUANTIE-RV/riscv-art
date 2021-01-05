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

#ifndef ART_RUNTIME_ARCH_RISCV64_REGISTERS_RISCV64_H_
#define ART_RUNTIME_ARCH_RISCV64_REGISTERS_RISCV64_H_

#include <iosfwd>

#include "base/macros.h"

namespace art {
namespace riscv64 {

enum GpuRegister {
  ZERO =  0,
  RA   =  1,  // Return Address
  SP   =  2,  // Stack pointer.
  GP   =  3,  // Global pointer
  TP   =  4,  // Thread pointer.
  T0   =  5,  // Temporary register.
  T1   =  6,  // Temporary register.
  T2   =  7,  // Temporary register.
  S0   =  8,  // Saved register/Frame pointer.
  S1   =  9,  // Saved register
  A0   = 10,  // Function arguments/return values.
  A1   = 11,  // Function arguments/return values.
  A2   = 12,  // Function arguments.
  A3   = 13,  // ...
  A4   = 14,
  A5   = 15,
  A6   = 16,
  A7   = 17,
  S2   = 18,  // Saved register
  S3   = 19,  // ...
  S4   = 20,
  S5   = 21,
  S6   = 22,
  S7   = 23,
  S8   = 24,
  S9   = 25,
  S10  = 26,
  S11  = 27,
  T3   = 28,  // Temporary register.
  T4   = 29,  // ...
  T5   = 30,
  T6   = 31,
  PC   = 32,

  FP   = S0,  // Frame pointer.
  TMP  = T5,
  TMP2 = T4,
  AT   = T3,
  V0   = A0,
  V1   = A1,

  TR   = S1,  // ART Thread Register (Same as the definition in asm_support_riscv64.S)
  T9   = T6,
  kNumberOfGpuRegisters = 32,
  kNoGpuRegister = -1  // Signals an illegal register.
};
std::ostream& operator<<(std::ostream& os, const GpuRegister& rhs);

// Values for floating point registers.
enum FpuRegister {
  FT0  =  0,  // Temporary register.
  FT1  =  1,  // ...
  FT2  =  2,
  FT3  =  3,
  FT4  =  4,
  FT5  =  5,
  FT6  =  6,
  FT7  =  7,
  FS0  =  8,  // Saved register
  FS1  =  9,  // ...
  FA0  = 10,  // Function arguments/return values.
  FA1  = 11,  // Function arguments/return values.
  FA2  = 12,  // Function arguments
  FA3  = 13,
  FA4  = 14,
  FA5  = 15,
  FA6  = 16,
  FA7  = 17,
  FS2  = 18,   // Saved register
  FS3  = 19,
  FS4  = 20,
  FS5  = 21,
  FS6  = 22,
  FS7  = 23,
  FS8  = 24,
  FS9  = 25,
  FS10 = 26,
  FS11 = 27,
  FT8  = 28,  // Temporary register.
  FT9  = 29,
  FT10 = 30,
  FT11 = 31,

  F0   = FA0,
  FTMP = FT11,   // scratch register
  FTMP2 = FT10,  // scratch register (in addition to FTMP, reserved for MSA instructions)
  kNumberOfFpuRegisters = 32,
  kNoFpuRegister = -1,
};
std::ostream& operator<<(std::ostream& os, const FpuRegister& rhs);

// Values for vector registers.
enum VectorRegister {
  W0  =  0,
  W1  =  1,
  W2  =  2,
  W3  =  3,
  W4  =  4,
  W5  =  5,
  W6  =  6,
  W7  =  7,
  W8  =  8,
  W9  =  9,
  W10 = 10,
  W11 = 11,
  W12 = 12,
  W13 = 13,
  W14 = 14,
  W15 = 15,
  W16 = 16,
  W17 = 17,
  W18 = 18,
  W19 = 19,
  W20 = 20,
  W21 = 21,
  W22 = 22,
  W23 = 23,
  W24 = 24,
  W25 = 25,
  W26 = 26,
  W27 = 27,
  W28 = 28,
  W29 = 29,
  W30 = 30,
  W31 = 31,
  kNumberOfVectorRegisters = 32,
  kNoVectorRegister = -1,
};
std::ostream& operator<<(std::ostream& os, const VectorRegister& rhs);

}  // namespace riscv64
}  // namespace art

#endif  // ART_RUNTIME_ARCH_RISCV64_REGISTERS_RISCV64_H_
