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

#include "context_riscv64.h"

#include "base/bit_utils.h"
#include "base/bit_utils_iterator.h"
#include "quick/quick_method_frame_info.h"

namespace art {
namespace riscv64 {

static constexpr uintptr_t gZero = 0;

void Riscv64Context::Reset() {
  std::fill_n(gprs_, arraysize(gprs_), nullptr);
  std::fill_n(fprs_, arraysize(fprs_), nullptr);
  gprs_[SP] = &sp_;
  gprs_[T6] = &t6_;
  gprs_[A0] = &arg0_;
  // Initialize registers with easy to spot debug values.
  sp_ = Riscv64Context::kBadGprBase + SP;
  t6_ = Riscv64Context::kBadGprBase + T6;
  arg0_ = 0;
}

void Riscv64Context::FillCalleeSaves(uint8_t* frame, const QuickMethodFrameInfo& frame_info) {
  int spill_pos = 0;
  int gpr_spill_pos = spill_pos + 2;

  // Core registers come first, from the highest down to the lowest.
  for (uint32_t core_reg : HighToLowBits(frame_info.CoreSpillMask())) {
    if (core_reg == RA) {
      // RA is at top of the frame
      gprs_[RA] = CalleeSaveAddress(frame, 0, frame_info.FrameSizeInBytes());
    } else if (core_reg == S0) {
      // FP is at the next to the top of the frame
      gprs_[S0] = CalleeSaveAddress(frame, 1, frame_info.FrameSizeInBytes());
    } else {
      // the other saved register is from (top - 2) offset
      gprs_[core_reg] = CalleeSaveAddress(frame, gpr_spill_pos, frame_info.FrameSizeInBytes());
      ++gpr_spill_pos;
    }
    ++spill_pos;
  }
  DCHECK_EQ(spill_pos, POPCOUNT(frame_info.CoreSpillMask()));

  // FP registers come second, from the highest down to the lowest.
  for (uint32_t fp_reg : HighToLowBits(frame_info.FpSpillMask())) {
    fprs_[fp_reg] = CalleeSaveAddress(frame, spill_pos, frame_info.FrameSizeInBytes());
    ++spill_pos;
  }
  DCHECK_EQ(spill_pos, POPCOUNT(frame_info.CoreSpillMask()) + POPCOUNT(frame_info.FpSpillMask()));
}

void Riscv64Context::SetGPR(uint32_t reg, uintptr_t value) {
  CHECK_LT(reg, static_cast<uint32_t>(kNumberOfGpuRegisters));
  DCHECK(IsAccessibleGPR(reg));
  CHECK_NE(gprs_[reg], &gZero);  // Can't overwrite this static value since they are never reset.
  *gprs_[reg] = value;
}

void Riscv64Context::SetFPR(uint32_t reg, uintptr_t value) {
  CHECK_LT(reg, static_cast<uint32_t>(kNumberOfFpuRegisters));
  DCHECK(IsAccessibleFPR(reg));
  CHECK_NE(fprs_[reg], &gZero);  // Can't overwrite this static value since they are never reset.
  *fprs_[reg] = value;
}

void Riscv64Context::SmashCallerSaves() {
  // This needs to be 0 because we want a null/zero return value.
  // gprs_[A0] = const_cast<uintptr_t*>(&gZero);
  // gprs_[A1] = const_cast<uintptr_t*>(&gZero);
  gprs_[A0] = const_cast<uintptr_t*>(&gZero);
  gprs_[A1] = const_cast<uintptr_t*>(&gZero);
  gprs_[A2] = nullptr;
  gprs_[A3] = nullptr;
  gprs_[A4] = nullptr;
  gprs_[A5] = nullptr;
  gprs_[A6] = nullptr;
  gprs_[A7] = nullptr;

  // gprs_[T0] = nullptr;
  // gprs_[T1] = nullptr;
  // gprs_[T2] = nullptr;
  // gprs_[T3] = nullptr;
  // gprs_[T4] = nullptr;
  // gprs_[T5] = nullptr;
  // gprs_[T6] = nullptr;

  // f0-f7 / f10-f17 / f28-f31 are caller-saved;
  fprs_[FA0] = nullptr;
  fprs_[FA1] = nullptr;
  fprs_[FA2] = nullptr;
  fprs_[FA3] = nullptr;
  fprs_[FA4] = nullptr;
  fprs_[FA5] = nullptr;
  fprs_[FA6] = nullptr;
  fprs_[FA7] = nullptr;

  fprs_[FT0] = nullptr;
  fprs_[FT1] = nullptr;
  fprs_[FT2] = nullptr;
  fprs_[FT3] = nullptr;
  fprs_[FT4] = nullptr;
  fprs_[FT5] = nullptr;
  fprs_[FT6] = nullptr;
  fprs_[FT7] = nullptr;
  fprs_[FT8] = nullptr;
  fprs_[FT9] = nullptr;
  fprs_[FT10] = nullptr;
  fprs_[FT11] = nullptr;
}

extern "C" NO_RETURN void art_quick_do_long_jump(uint64_t*, uint64_t*);

void Riscv64Context::DoLongJump() {
  uintptr_t gprs[kNumberOfGpuRegisters];
  uintptr_t fprs[kNumberOfFpuRegisters];
  for (size_t i = 0; i < kNumberOfGpuRegisters; ++i) {
    gprs[i] = gprs_[i] != nullptr ? *gprs_[i] : Riscv64Context::kBadGprBase + i;
  }
  for (size_t i = 0; i < kNumberOfFpuRegisters; ++i) {
    fprs[i] = fprs_[i] != nullptr ? *fprs_[i] : Riscv64Context::kBadFprBase + i;
  }
  art_quick_do_long_jump(gprs, fprs);
}

}  // namespace riscv64
}  // namespace art
