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

#include "calling_convention_riscv64.h"

#include <android-base/logging.h>

#include "arch/instruction_set.h"
#include "handle_scope-inl.h"
#include "utils/riscv64/managed_register_riscv64.h"

namespace art {
namespace riscv64 {

// Up to kow many args can be enregistered. The rest of the args must go on the stack.
constexpr size_t kMaxRegisterArguments = 8u;
// Up to how many float-like (float, double) args can be enregistered.
// The rest of the args must go on the stack.
constexpr size_t kMaxFloatOrDoubleRegisterArguments = 8u;
// Up to how many integer-like (pointers, objects, longs, int, short, bool, etc) args can be
// enregistered. The rest of the args must go on the stack.
constexpr size_t kMaxIntLikeRegisterArguments = 8u;


static const GpuRegister kGpuArgumentRegisters[] = {
  A0, A1, A2, A3, A4, A5, A6, A7
};

static const FpuRegister kFpuArgumentRegisters[] = {
  FA0, FA1, FA2, FA3, FA4, FA5, FA6, FA7
};

static constexpr ManagedRegister kCalleeSaveRegisters[] = {
    // Core registers.
    Riscv64ManagedRegister::FromGpuRegister(S2),
    Riscv64ManagedRegister::FromGpuRegister(S3),
    Riscv64ManagedRegister::FromGpuRegister(S4),
    Riscv64ManagedRegister::FromGpuRegister(S5),
    Riscv64ManagedRegister::FromGpuRegister(S6),
    Riscv64ManagedRegister::FromGpuRegister(S7),
    Riscv64ManagedRegister::FromGpuRegister(S8),
    Riscv64ManagedRegister::FromGpuRegister(S9),
    Riscv64ManagedRegister::FromGpuRegister(S10),
    Riscv64ManagedRegister::FromGpuRegister(S0),
    // No hard float callee saves.
};

static constexpr uint32_t CalculateCoreCalleeSpillMask() {
  // RA is a special callee save which is not reported by CalleeSaveRegisters().
  uint32_t result = 1 << RA;
  for (auto&& r : kCalleeSaveRegisters) {
    if (r.AsRiscv64().IsGpuRegister()) {
      result |= (1 << r.AsRiscv64().AsGpuRegister());
    }
  }
  return result;
}

static constexpr uint32_t kCoreCalleeSpillMask = CalculateCoreCalleeSpillMask();
static constexpr uint32_t kFpCalleeSpillMask = 0u;

// Calling convention
ManagedRegister Riscv64ManagedRuntimeCallingConvention::InterproceduralScratchRegister() {
  return Riscv64ManagedRegister::FromGpuRegister(T6);
}

ManagedRegister Riscv64JniCallingConvention::InterproceduralScratchRegister() {
  return Riscv64ManagedRegister::FromGpuRegister(T6);
}

static ManagedRegister ReturnRegisterForShorty(const char* shorty) {
  if (shorty[0] == 'F' || shorty[0] == 'D') {
    return Riscv64ManagedRegister::FromFpuRegister(FA0);
  } else if (shorty[0] == 'V') {
    return Riscv64ManagedRegister::NoRegister();
  } else {
    return Riscv64ManagedRegister::FromGpuRegister(A0);
  }
}

ManagedRegister Riscv64ManagedRuntimeCallingConvention::ReturnRegister() {
  return ReturnRegisterForShorty(GetShorty());
}

ManagedRegister Riscv64JniCallingConvention::ReturnRegister() {
  return ReturnRegisterForShorty(GetShorty());
}

ManagedRegister Riscv64JniCallingConvention::IntReturnRegister() {
  return Riscv64ManagedRegister::FromGpuRegister(A0);
}

// Managed runtime calling convention

ManagedRegister Riscv64ManagedRuntimeCallingConvention::MethodRegister() {
  return Riscv64ManagedRegister::FromGpuRegister(A0);
}

bool Riscv64ManagedRuntimeCallingConvention::IsCurrentParamInRegister() {
  return false;  // Everything moved to stack on entry.
}

bool Riscv64ManagedRuntimeCallingConvention::IsCurrentParamOnStack() {
  return true;
}

ManagedRegister Riscv64ManagedRuntimeCallingConvention::CurrentParamRegister() {
  LOG(FATAL) << "Should not reach here";
  UNREACHABLE();
}

FrameOffset Riscv64ManagedRuntimeCallingConvention::CurrentParamStackOffset() {
  CHECK(IsCurrentParamOnStack());
  FrameOffset result =
      FrameOffset(displacement_.Int32Value() +  // displacement
                  kFramePointerSize +  // Method ref
                  (itr_slots_ * sizeof(uint32_t)));  // offset into in args
  return result;
}

const ManagedRegisterEntrySpills& Riscv64ManagedRuntimeCallingConvention::EntrySpills() {
  if ((entry_spills_.size() == 0) && (NumArgs() > 0)) {
    int gp_reg_index = 1;   // we start from X1/W1, X0 holds ArtMethod*.
    int fp_reg_index = 0;   // D0/S0.

    // We need to choose the correct register (D/S or X/W) since the managed
    // stack uses 32bit stack slots.
    ResetIterator(FrameOffset(0));
    while (HasNext()) {
      if (IsCurrentParamAFloatOrDouble()) {  // FP regs.
          if (fp_reg_index < 8) {
            FpuRegister arg = kFpuArgumentRegisters[fp_reg_index];
            Riscv64ManagedRegister reg = Riscv64ManagedRegister::FromFpuRegister(arg);
            entry_spills_.push_back(reg, IsCurrentParamADouble() ? 8 : 4);
            fp_reg_index++;
          } else {
            if (!IsCurrentParamADouble()) {
              entry_spills_.push_back(ManagedRegister::NoRegister(), 4);
            } else {
              entry_spills_.push_back(ManagedRegister::NoRegister(), 8);
            }
          }
      } else {  // GP regs.
          if (gp_reg_index < 8) {
            GpuRegister arg = kGpuArgumentRegisters[gp_reg_index];
            Riscv64ManagedRegister reg = Riscv64ManagedRegister::FromGpuRegister(arg);
            entry_spills_.push_back(reg,
                                  (IsCurrentParamALong() && (!IsCurrentParamAReference())) ? 8 : 4);
            gp_reg_index++;
          } else {
            if (IsCurrentParamALong() && (!IsCurrentParamAReference())) {
              entry_spills_.push_back(ManagedRegister::NoRegister(), 8);
            } else {
              entry_spills_.push_back(ManagedRegister::NoRegister(), 4);
            }
          }
      }
      Next();
    }
  }
  return entry_spills_;
}

// JNI calling convention

Riscv64JniCallingConvention::Riscv64JniCallingConvention(bool is_static,
                                                       bool is_synchronized,
                                                       bool is_critical_native,
                                                       const char* shorty)
    : JniCallingConvention(is_static,
                           is_synchronized,
                           is_critical_native,
                           shorty,
                           kRiscv64PointerSize) {
}

uint32_t Riscv64JniCallingConvention::CoreSpillMask() const {
  return kCoreCalleeSpillMask;
}

uint32_t Riscv64JniCallingConvention::FpSpillMask() const {
  return kFpCalleeSpillMask;
}

ManagedRegister Riscv64JniCallingConvention::ReturnScratchRegister() const {
  return Riscv64ManagedRegister::FromGpuRegister(AT);
}

size_t Riscv64JniCallingConvention::FrameSize() {
  // ArtMethod*, RA and callee save area size, local reference segment state.
  size_t method_ptr_size = static_cast<size_t>(kFramePointerSize);
  size_t ra_and_callee_save_area_size = (CalleeSaveRegisters().size() + 1) * kFramePointerSize;

  size_t frame_data_size = method_ptr_size + ra_and_callee_save_area_size;
  if (LIKELY(HasLocalReferenceSegmentState())) {                     // Local ref. segment state.
    // Local reference segment state is sometimes excluded.
    frame_data_size += sizeof(uint32_t);
  }
  // References plus 2 words for HandleScope header.
  size_t handle_scope_size = HandleScope::SizeOf(kRiscv64PointerSize, ReferenceCount());

  size_t total_size = frame_data_size;
  if (LIKELY(HasHandleScope())) {
    // HandleScope is sometimes excluded.
    total_size += handle_scope_size;                                 // Handle scope size.
  }

  // Plus return value spill area size.
  total_size += SizeOfReturnValue();

  return RoundUp(total_size, kStackAlignment);
}

size_t Riscv64JniCallingConvention::OutArgSize() {
  return RoundUp(NumberOfOutgoingStackArgs() * kFramePointerSize, kStackAlignment);
}

ArrayRef<const ManagedRegister> Riscv64JniCallingConvention::CalleeSaveRegisters() const {
  return ArrayRef<const ManagedRegister>(kCalleeSaveRegisters);
}

bool Riscv64JniCallingConvention::IsCurrentParamInRegister() {
  if (IsCurrentParamAFloatOrDouble()) {
    return (itr_float_and_doubles_ < kMaxFloatOrDoubleRegisterArguments);
  } else {
    return ((itr_args_ - itr_float_and_doubles_) < kMaxIntLikeRegisterArguments);
  }
  // TODO: Can we just call CurrentParamRegister to figure this out?
}

bool Riscv64JniCallingConvention::IsCurrentParamOnStack() {
  return !IsCurrentParamInRegister();
}

ManagedRegister Riscv64JniCallingConvention::CurrentParamRegister() {
  CHECK(IsCurrentParamInRegister());
  if (IsCurrentParamAFloatOrDouble()) {
    // CHECK_LT(itr_float_and_doubles_, kMaxFloatOrDoubleRegisterArguments);
    return Riscv64ManagedRegister::FromFpuRegister(kFpuArgumentRegisters[itr_float_and_doubles_]);
  } else {
    int gp_reg = itr_args_ - itr_float_and_doubles_;
    // CHECK_LT(static_cast<unsigned int>(gp_reg), kMaxIntLikeRegisterArguments);
    return Riscv64ManagedRegister::FromGpuRegister(kGpuArgumentRegisters[gp_reg]);
  }
}

FrameOffset Riscv64JniCallingConvention::CurrentParamStackOffset() {
  CHECK(IsCurrentParamOnStack());
  size_t args_on_stack = itr_args_
    - std::min(kMaxFloatOrDoubleRegisterArguments,
    static_cast<size_t>(itr_float_and_doubles_))
    - std::min(kMaxIntLikeRegisterArguments,
    static_cast<size_t>(itr_args_ - itr_float_and_doubles_));

  size_t offset = displacement_.Int32Value() - OutArgSize() + (args_on_stack * kFramePointerSize);
  CHECK_LT(offset, OutArgSize());
  return FrameOffset(offset);
  // TODO: Seems identical to X86_64 code.
}

size_t Riscv64JniCallingConvention::NumberOfOutgoingStackArgs() {
  // all arguments including JNI args
  size_t all_args = NumArgs() + NumberOfExtraArgumentsForJni();
  DCHECK_GE(all_args, NumFloatOrDoubleArgs());

  size_t all_stack_args = all_args
    - std::min(kMaxFloatOrDoubleRegisterArguments,
    static_cast<size_t>(NumFloatOrDoubleArgs()))
    - std::min(kMaxIntLikeRegisterArguments,
    static_cast<size_t>((all_args - NumFloatOrDoubleArgs())));

  // TODO: Seems similar to X86_64 code except it doesn't count return pc.
  return all_stack_args;
}

}  // namespace riscv64
}  // namespace art
