%def const(helper="UndefinedConstHandler"):
    /* const/class vAA, type@BBBB */
    /* const/method-handle vAA, method_handle@BBBB */
    /* const/method-type vAA, proto@BBBB */
    /* const/string vAA, string@@BBBB */
    .extern $helper
    EXPORT_PC
    lhu     a0, 2(rPC)                  # a0 <- BBBB
    srli    a1, rINST, 8                # a1 <- AA
    addi     a2, rFP, OFF_FP_SHADOWFRAME
    move    a3, rSELF
    PREFETCH_INST 2                     # load rINST
    jal     $helper                     # (index, tgt_reg, shadow_frame, self)

    bnez    a0, MterpPossibleException  # let reference interpreter deal with it.
    ADVANCE 2                           # advance rPC
    GET_INST_OPCODE a0                  # extract opcode from rINST
    GOTO_OPCODE a0                      # jump to next instruction

%def unused():
/*
 * Bail to reference interpreter to throw.
 */
    j       MterpFallback

%def op_const():
    /* const vAA, #+BBBBbbbb */
    srli    a2, rINST, 8                # a2 <- AA
    lhu     a0, 2(rPC)                  # a0 <- bbbb (low)
    lhu     a1, 4(rPC)                  # a1 <- BBBB (high)
    FETCH_ADVANCE_INST 3                # advance rPC, load rINST
    slli    a1, a1, 16                  # a0 = BBBBbbbb
    or     a0, a0, a1
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG a0, a2                     # vAA <- +BBBBbbbb
    GOTO_OPCODE t4                      # jump to next instruction

%def op_const_16():
    /* const/16 vAA, #+BBBB */
    srli    a2, rINST, 8                # a2 <- AA
    lh      a0, 2(rPC)                  # a0 <- sign-extended BBBB
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG a0, a2                     # vAA <- +BBBB
    GOTO_OPCODE t4                      # jump to next instruction

%def op_const_4():
    /* const/4 vA, #+B */
    # EXT  rINST[11:8]      --> a2      # a2 <- A
    li     t6, 0x0F00
    and    a2, rINST, t6
    srli   a2, a2, 8

    # Sign EXT rINST[15:12] --> a0      # a0 <- B
    lh       a0, 0(rPC)                 # sign extend load A and B in a0
    srai     a0, a0, 12                 # shift B into its final position

    FETCH_ADVANCE_INST 1                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG a0, a2                     # vA <- +B
    GOTO_OPCODE t4                      # jump to next instruction

%def op_const_class():
%  const(helper="MterpConstClass")

%def op_const_high16():
    /* const/high16 vAA, #+BBBB0000 */
    srli    a2, rINST, 8                # a2 <- AA
    lh      a0, 2(rPC)                  # a0 <- BBBB
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    sll     a0, a0, 16                  # a0 <- BBBB0000
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG a0, a2                     # vAA <- +BBBB0000
    GOTO_OPCODE t4                     # jump to next instruction

%def op_const_method_handle():
%  const(helper="MterpConstMethodHandle")

%def op_const_method_type():
%  const(helper="MterpConstMethodType")

%def op_const_string():
%  const(helper="MterpConstString")

%def op_const_string_jumbo():
    /* const/string vAA, String//BBBBBBBB */
    .extern MterpConstString
    EXPORT_PC
    lh      a0, 2(rPC)                  # a0 <- bbbb (low)
    lh      a4, 4(rPC)                  # a4 <- BBBB (high)
    srli     a1, rINST, 8                # a1 <- AA
    INS     a0, a4, 16, 16              # a0 <- BBBBbbbb
    addi    a2, rFP, OFF_FP_SHADOWFRAME
    move    a3, rSELF
    PREFETCH_INST 3                     # load rINST
    jal     MterpConstString            # (index, tgt_reg, shadow_frame, self)

    bnez    a0, MterpPossibleException  # let reference interpreter deal with it.
    ADVANCE 3                           # advance rPC
    GET_INST_OPCODE t4                  # extract opcode from rINST
    GOTO_OPCODE t4                      # jump to next instruction

%def op_const_wide():
    /* const-wide vAA, #+HHHHhhhhBBBBbbbb */
    srli    a4, rINST, 8                # a4 <- AA
    lh      a0, 2(rPC)                  # a0 <- bbbb (low)
    lh      a1, 4(rPC)                  # a1 <- BBBB (low middle)
    lh      a2, 6(rPC)                  # a2 <- hhhh (high middle)
    lh      a3, 8(rPC)                  # a3 <- HHHH (high)
    FETCH_ADVANCE_INST 5                # advance rPC, load rINST
    INS     a0, a1, 16, 16              # a0 = BBBBbbbb
    INS     a2, a3, 16, 16              # a2 = HHHHhhhh
    DINSU   a0, a2, 32, 32              # a0 = HHHHhhhhBBBBbbbb
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_WIDE a0, a4                # vAA <- +HHHHhhhhBBBBbbbb
    GOTO_OPCODE t4                      # jump to next instruction

%def op_const_wide_16():
    /* const-wide/16 vAA, #+BBBB */
    srli     a2, rINST, 8                # a2 <- AA
    lh      a0, 2(rPC)                  # a0 <- sign-extended BBBB
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_WIDE a0, a2                # vAA <- +BBBB
    GOTO_OPCODE t4                      # jump to next instruction

%def op_const_wide_32():
    /* const-wide/32 vAA, #+BBBBbbbb */
    srli     a2, rINST, 8                # a2 <- AA
    lh      a0, 2(rPC)                  # a0 <- bbbb (low)
    lh      a1, 4(rPC)                  # a1 <- BBBB (high)
    FETCH_ADVANCE_INST 3                # advance rPC, load rINST
    INS     a0, a1, 16, 16              # a0 = BBBBbbbb
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_WIDE a0, a2                # vAA <- +BBBBbbbb
    GOTO_OPCODE t4                      # jump to next instruction

%def op_const_wide_high16():
    /* const-wide/high16 vAA, #+BBBB000000000000 */
    srli     a2, rINST, 8                # a2 <- AA
    lh      a0, 2(rPC)                  # a0 <- BBBB
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    slli    a0, a0, 48                  # dsll32  a0, a0, 16                  # a0 <- BBBB000000000000
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_WIDE a0, a2                # vAA <- +BBBB000000000000
    GOTO_OPCODE t4                      # jump to next instruction

%def op_monitor_enter():
    /*
     * Synchronize on an object.
     */
    /* monitor-enter vAA */
    .extern artLockObjectFromCode
    EXPORT_PC
    srli     a2, rINST, 8                # a2 <- AA
    GET_VREG_U a0, a2                   # a0 <- vAA (object)
    move    a1, rSELF                   # a1 <- self
    jal     artLockObjectFromCode
    bnez    a0, MterpException
    FETCH_ADVANCE_INST 1
    ld      a0, THREAD_USE_MTERP_OFFSET(rSELF)
    beqz    a0, MterpFallback
    GET_INST_OPCODE t4                  # extract opcode from rINST
    GOTO_OPCODE t4                      # jump to next instruction

%def op_monitor_exit():
    /*
     * Unlock an object.
     *
     * Exceptions that occur when unlocking a monitor need to appear as
     * if they happened at the following instruction.  See the Dalvik
     * instruction spec.
     */
    /* monitor-exit vAA */
    .extern artUnlockObjectFromCode
    EXPORT_PC
    srli     a2, rINST, 8                # a2 <- AA
    GET_VREG_U a0, a2                   # a0 <- vAA (object)
    move    a1, rSELF                   # a1 <- self
    jal     artUnlockObjectFromCode     # v0 <- success for unlock(self, obj)
    bnez    a0, MterpException
    FETCH_ADVANCE_INST 1                # before throw: advance rPC, load rINST
    ld      a0, THREAD_USE_MTERP_OFFSET(rSELF)
    beqz    a0, MterpFallback
    GET_INST_OPCODE t4                  # extract opcode from rINST
    GOTO_OPCODE t4                      # jump to next instruction

%def op_move(is_object="0"):
    /* for move, move-object, long-to-int */
    /* op vA, vB */
    EXT     a2, rINST, 8, 4             # a2 <- A
    EXT     a3, rINST, 12, 4            # a3 <- B
    FETCH_ADVANCE_INST 1                # advance rPC, load rINST
    GET_VREG a0, a3                     # a0 <- vB
    GET_INST_OPCODE t4                  # extract opcode from rINST
    .if $is_object
    SET_VREG_OBJECT a0, a2              # vA <- vB
    .else
    SET_VREG a0, a2                     # vA <- vB
    .endif
    GOTO_OPCODE t4                      # jump to next instruction

%def op_move_16(is_object="0"):
    /* for: move/16, move-object/16 */
    /* op vAAAA, vBBBB */
    lhu     a3, 4(rPC)                  # a3 <- BBBB
    lhu     a2, 2(rPC)                  # a2 <- AAAA
    FETCH_ADVANCE_INST 3                # advance rPC, load rINST
    GET_VREG a0, a3                     # a0 <- vBBBB
    GET_INST_OPCODE t4                  # extract opcode from rINST
    .if $is_object
    SET_VREG_OBJECT a0, a2              # vAAAA <- vBBBB
    .else
    SET_VREG a0, a2                     # vAAAA <- vBBBB
    .endif
    GOTO_OPCODE t4                      # jump to next instruction

%def op_move_exception():
    /* move-exception vAA */
    srli    a2, rINST, 8                # a2 <- AA
    ld      a0, THREAD_EXCEPTION_OFFSET(rSELF)  # load exception obj
    FETCH_ADVANCE_INST 1                # advance rPC, load rINST
    SET_VREG_OBJECT a0, a2              # vAA <- exception obj
    GET_INST_OPCODE t4                  # extract opcode from rINST
    sd      zero, THREAD_EXCEPTION_OFFSET(rSELF)  # clear exception
    GOTO_OPCODE t4                      # jump to next instruction

%def op_move_from16(is_object="0"):
    /* for: move/from16, move-object/from16 */
    /* op vAA, vBBBB */
    lhu     a3, 2(rPC)                  # a3 <- BBBB
    srli     a2, rINST, 8                # a2 <- AA
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    GET_VREG a0, a3                     # a0 <- vBBBB
    GET_INST_OPCODE t4                  # extract opcode from rINST
    .if $is_object
    SET_VREG_OBJECT a0, a2              # vAA <- vBBBB
    .else
    SET_VREG a0, a2                     # vAA <- vBBBB
    .endif
    GOTO_OPCODE t4                      # jump to next instruction

%def op_move_object():
%  op_move(is_object="1")

%def op_move_object_16():
%  op_move_16(is_object="1")

%def op_move_object_from16():
%  op_move_from16(is_object="1")

%def op_move_result(is_object="0"):
    /* for: move-result, move-result-object */
    /* op vAA */
    srli     a2, rINST, 8                # a2 <- AA
    FETCH_ADVANCE_INST 1                # advance rPC, load rINST
    ld      a0, OFF_FP_RESULT_REGISTER(rFP)  # get pointer to result JType
    lw      a0, 0(a0)                   # a0 <- result.i
    GET_INST_OPCODE t4                  # extract opcode from rINST
    .if $is_object
    SET_VREG_OBJECT a0, a2              # vAA <- result
    .else
    SET_VREG a0, a2                     # vAA <- result
    .endif
    GOTO_OPCODE t4                      # jump to next instruction

%def op_move_result_object():
%  op_move_result(is_object="1")

%def op_move_result_wide():
    /* for: move-result-wide */
    /* op vAA */
    srli     a2, rINST, 8                # a2 <- AA
    FETCH_ADVANCE_INST 1                # advance rPC, load rINST
    ld      a0, OFF_FP_RESULT_REGISTER(rFP)  # get pointer to result JType
    ld      a0, 0(a0)                   # a0 <- result.j
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_WIDE a0, a2                # vAA <- result
    GOTO_OPCODE t4                      # jump to next instruction

%def op_move_wide():
    /* move-wide vA, vB */
    /* NOTE: regs can overlap, e.g. "move v6,v7" or "move v7,v6" */
    EXT     a3, rINST, 12, 4            # a3 <- B
    EXT     a2, rINST, 8, 4             # a2 <- A
    GET_VREG_WIDE a0, a3                # a0 <- vB
    FETCH_ADVANCE_INST 1                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_WIDE a0, a2                # vA <- vB
    GOTO_OPCODE t4                      # jump to next instruction

%def op_move_wide_16():
    /* move-wide/16 vAAAA, vBBBB */
    /* NOTE: regs can overlap, e.g. "move v6,v7" or "move v7,v6" */
    lhu     a3, 4(rPC)                  # a3 <- BBBB
    lhu     a2, 2(rPC)                  # a2 <- AAAA
    GET_VREG_WIDE a0, a3                # a0 <- vBBBB
    FETCH_ADVANCE_INST 3                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_WIDE a0, a2                # vAAAA <- vBBBB
    GOTO_OPCODE t4                      # jump to next instruction

%def op_move_wide_from16():
    /* move-wide/from16 vAA, vBBBB */
    /* NOTE: regs can overlap, e.g. "move v6,v7" or "move v7,v6" */
    lhu     a3, 2(rPC)                  # a3 <- BBBB
    srli     a2, rINST, 8                # a2 <- AA
    GET_VREG_WIDE a0, a3                # a0 <- vBBBB
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_WIDE a0, a2                # vAA <- vBBBB
    GOTO_OPCODE t4                      # jump to next instruction

%def op_nop():
    FETCH_ADVANCE_INST 1                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    GOTO_OPCODE t4                      # jump to next instruction

%def op_unused_3e():
%  unused()

%def op_unused_3f():
%  unused()

%def op_unused_40():
%  unused()

%def op_unused_41():
%  unused()

%def op_unused_42():
%  unused()

%def op_unused_43():
%  unused()

%def op_unused_79():
%  unused()

%def op_unused_7a():
%  unused()

%def op_unused_f3():
%  unused()

%def op_unused_f4():
%  unused()

%def op_unused_f5():
%  unused()

%def op_unused_f6():
%  unused()

%def op_unused_f7():
%  unused()

%def op_unused_f8():
%  unused()

%def op_unused_f9():
%  unused()

%def op_unused_fc():
%  unused()

%def op_unused_fd():
%  unused()
