/*
 * Copyright (C) 2016 The Android Open Source Project
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

#include "jni_macro_assembler_riscv64.h"

#include "entrypoints/quick/quick_entrypoints.h"
#include "managed_register_riscv64.h"
#include "offsets.h"
#include "thread.h"

namespace art {
namespace riscv64 {

#define __ asm_.

Riscv64JNIMacroAssembler::~Riscv64JNIMacroAssembler() {
}

void Riscv64JNIMacroAssembler::FinalizeCode() {
  __ FinalizeCode();
}

void Riscv64JNIMacroAssembler::GetCurrentThread(ManagedRegister tr) {
  __ GetCurrentThread(tr);
}

void Riscv64JNIMacroAssembler::GetCurrentThread(FrameOffset offset, ManagedRegister scratch) {
  __ GetCurrentThread(offset, scratch);
}

// See Riscv64 PCS Section 5.2.2.1.
void Riscv64JNIMacroAssembler::IncreaseFrameSize(size_t adjust) {
  __ IncreaseFrameSize(adjust);
}

// See Riscv64 PCS Section 5.2.2.1.
void Riscv64JNIMacroAssembler::DecreaseFrameSize(size_t adjust) {
  __ DecreaseFrameSize(adjust);
}

void Riscv64JNIMacroAssembler::Store(FrameOffset offs, ManagedRegister m_src, size_t size) {
  __ Store(offs, m_src, size);
}

void Riscv64JNIMacroAssembler::StoreRef(FrameOffset offs, ManagedRegister m_src) {
  __ StoreRef(offs, m_src);
}

void Riscv64JNIMacroAssembler::StoreRawPtr(FrameOffset offs, ManagedRegister m_src) {
  __ StoreRawPtr(offs, m_src);
}

void Riscv64JNIMacroAssembler::StoreImmediateToFrame(FrameOffset offs,
                                                   uint32_t imm,
                                                   ManagedRegister m_scratch) {
  __ StoreImmediateToFrame(offs, imm, m_scratch);
}

void Riscv64JNIMacroAssembler::StoreStackOffsetToThread(ThreadOffset64 tr_offs,
                                                      FrameOffset fr_offs,
                                                      ManagedRegister m_scratch) {
  __ StoreStackOffsetToThread(tr_offs, fr_offs, m_scratch);
}

void Riscv64JNIMacroAssembler::StoreStackPointerToThread(ThreadOffset64 tr_offs) {
  __ StoreStackPointerToThread(tr_offs);
}

void Riscv64JNIMacroAssembler::StoreSpanning(FrameOffset dest_off,
                                           ManagedRegister m_source,
                                           FrameOffset in_off,
                                           ManagedRegister m_scratch) {
  __ StoreSpanning(dest_off, m_source, in_off, m_scratch);
}

void Riscv64JNIMacroAssembler::Load(ManagedRegister m_dst, FrameOffset src, size_t size) {
  __ Load(m_dst, src, size);
}

void Riscv64JNIMacroAssembler::LoadFromThread(ManagedRegister m_dst,
                                            ThreadOffset64 src,
                                            size_t size) {
  __ LoadFromThread(m_dst, src, size);
}

void Riscv64JNIMacroAssembler::LoadRef(ManagedRegister m_dst, FrameOffset offs) {
  __ LoadRef(m_dst, offs);
}

void Riscv64JNIMacroAssembler::LoadRef(ManagedRegister m_dst,
                                     ManagedRegister m_base,
                                     MemberOffset offs,
                                     bool unpoison_reference) {
  __ LoadRef(m_dst, m_base, offs, unpoison_reference);
}

void Riscv64JNIMacroAssembler::LoadRawPtr(ManagedRegister m_dst,
                                        ManagedRegister m_base,
                                        Offset offs) {
  __ LoadRawPtr(m_dst, m_base, offs);
}

void Riscv64JNIMacroAssembler::LoadRawPtrFromThread(ManagedRegister m_dst, ThreadOffset64 offs) {
  __ LoadRawPtrFromThread(m_dst, offs);
}

// Copying routines.
void Riscv64JNIMacroAssembler::Move(ManagedRegister m_dst, ManagedRegister m_src, size_t size) {
  __ Move(m_dst, m_src, size);
}

void Riscv64JNIMacroAssembler::CopyRawPtrFromThread(FrameOffset fr_offs,
                                                  ThreadOffset64 tr_offs,
                                                  ManagedRegister m_scratch) {
  __ CopyRawPtrFromThread(fr_offs, tr_offs, m_scratch);
}

void Riscv64JNIMacroAssembler::CopyRawPtrToThread(ThreadOffset64 tr_offs,
                                                FrameOffset fr_offs,
                                                ManagedRegister m_scratch) {
  __ CopyRawPtrToThread(tr_offs, fr_offs, m_scratch);
}

void Riscv64JNIMacroAssembler::CopyRef(FrameOffset dest, FrameOffset src, ManagedRegister m_scratch) {
  __ CopyRef(dest, src, m_scratch);
}

void Riscv64JNIMacroAssembler::Copy(FrameOffset dest,
                                  FrameOffset src,
                                  ManagedRegister m_scratch,
                                  size_t size) {
  __ Copy(dest, src, m_scratch, size);
}

void Riscv64JNIMacroAssembler::Copy(FrameOffset dest,
                                  ManagedRegister src_base,
                                  Offset src_offset,
                                  ManagedRegister m_scratch,
                                  size_t size) {
  __ Copy(dest, src_base, src_offset, m_scratch, size);
}

void Riscv64JNIMacroAssembler::Copy(ManagedRegister m_dest_base,
                                  Offset dest_offs,
                                  FrameOffset src,
                                  ManagedRegister m_scratch,
                                  size_t size) {
  __ Copy(m_dest_base, dest_offs, src, m_scratch, size);
}

void Riscv64JNIMacroAssembler::Copy(FrameOffset dst,
                                  FrameOffset src_base,
                                  Offset src_offset,
                                  ManagedRegister mscratch,
                                  size_t size) {
  __ Copy(dst, src_base, src_offset, mscratch, size);
}

void Riscv64JNIMacroAssembler::Copy(ManagedRegister m_dest,
                                  Offset dest_offset,
                                  ManagedRegister m_src,
                                  Offset src_offset,
                                  ManagedRegister m_scratch,
                                  size_t size) {
  __ Copy(m_dest, dest_offset, m_src, src_offset, m_scratch, size);
}

void Riscv64JNIMacroAssembler::Copy(FrameOffset dst,
                                  Offset dest_offset,
                                  FrameOffset src,
                                  Offset src_offset,
                                  ManagedRegister scratch,
                                  size_t size) {
  __ Copy(dst, dest_offset, src, src_offset, scratch, size);
}

void Riscv64JNIMacroAssembler::MemoryBarrier(ManagedRegister m_scratch) {
  __ MemoryBarrier(m_scratch);
}

void Riscv64JNIMacroAssembler::SignExtend(ManagedRegister mreg, size_t size) {
  __ SignExtend(mreg, size);
}

void Riscv64JNIMacroAssembler::ZeroExtend(ManagedRegister mreg, size_t size) {
  __ ZeroExtend(mreg, size);
}

void Riscv64JNIMacroAssembler::VerifyObject(ManagedRegister m_src, bool could_be_null) {
  // TODO: not validating references.
  __ VerifyObject(m_src, could_be_null);
}

void Riscv64JNIMacroAssembler::VerifyObject(FrameOffset src, bool could_be_null) {
  // TODO: not validating references.
  __ VerifyObject(src, could_be_null);
}

void Riscv64JNIMacroAssembler::Call(ManagedRegister m_base, Offset offs, ManagedRegister m_scratch) {
  __ Call(m_base, offs, m_scratch);
}

void Riscv64JNIMacroAssembler::Call(FrameOffset base, Offset offs, ManagedRegister m_scratch) {
  __ Call(base, offs, m_scratch);
}

void Riscv64JNIMacroAssembler::CallFromThread(ThreadOffset64 offset,
                                            ManagedRegister scratch) {
  __ CallFromThread(offset, scratch);
}

void Riscv64JNIMacroAssembler::CreateHandleScopeEntry(ManagedRegister m_out_reg,
                                                    FrameOffset handle_scope_offs,
                                                    ManagedRegister m_in_reg,
                                                    bool null_allowed) {
  __ CreateHandleScopeEntry(m_out_reg, handle_scope_offs, m_in_reg, null_allowed);
}

void Riscv64JNIMacroAssembler::CreateHandleScopeEntry(FrameOffset out_off,
                                                    FrameOffset handle_scope_offset,
                                                    ManagedRegister m_scratch,
                                                    bool null_allowed) {
  __ CreateHandleScopeEntry(out_off, handle_scope_offset, m_scratch, null_allowed);
}

void Riscv64JNIMacroAssembler::LoadReferenceFromHandleScope(ManagedRegister m_out_reg,
                                                          ManagedRegister m_in_reg) {
  __ LoadReferenceFromHandleScope(m_out_reg, m_in_reg);
}

void Riscv64JNIMacroAssembler::ExceptionPoll(ManagedRegister m_scratch, size_t stack_adjust) {
  __ ExceptionPoll(m_scratch, stack_adjust);
}

std::unique_ptr<JNIMacroLabel> Riscv64JNIMacroAssembler::CreateLabel() {
  return std::unique_ptr<JNIMacroLabel>(new Riscv64JNIMacroLabel());
}

void Riscv64JNIMacroAssembler::Jump(JNIMacroLabel* label) {
  CHECK(label != nullptr);
  __ Bc(down_cast<Riscv64Label*>(Riscv64JNIMacroLabel::Cast(label)->AsRiscv64()));
}

void Riscv64JNIMacroAssembler::Jump(JNIMacroLabel* label,
                                  JNIMacroUnaryCondition condition,
                                  ManagedRegister test) {
  CHECK(label != nullptr);

  switch (condition) {
    case JNIMacroUnaryCondition::kZero:
      __ Beqzc(test.AsRiscv64().AsGpuRegister(), down_cast<Riscv64Label*>(Riscv64JNIMacroLabel::Cast(label)->AsRiscv64()));
      break;
    case JNIMacroUnaryCondition::kNotZero:
      __ Bnezc(test.AsRiscv64().AsGpuRegister(), down_cast<Riscv64Label*>(Riscv64JNIMacroLabel::Cast(label)->AsRiscv64()));
      break;
    default:
      LOG(FATAL) << "Not implemented unary condition: " << static_cast<int>(condition);
      UNREACHABLE();
  }
}

void Riscv64JNIMacroAssembler::Bind(JNIMacroLabel* label) {
  CHECK(label != nullptr);
  __ Bind(Riscv64JNIMacroLabel::Cast(label)->AsRiscv64());
}

void Riscv64JNIMacroAssembler::BuildFrame(size_t frame_size,
                                        ManagedRegister method_reg,
                                        ArrayRef<const ManagedRegister> callee_save_regs,
                                        const ManagedRegisterEntrySpills& entry_spills) {
  __ BuildFrame(frame_size, method_reg, callee_save_regs, entry_spills);
}

void Riscv64JNIMacroAssembler::RemoveFrame(size_t frame_size,
                                         ArrayRef<const ManagedRegister> callee_save_regs,
                                         bool may_suspend) {
  __ RemoveFrame(frame_size, callee_save_regs, may_suspend);
}

#undef ___

}  // namespace riscv64
}  // namespace art
