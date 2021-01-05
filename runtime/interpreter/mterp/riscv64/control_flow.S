%def bincmp(condition=""):
    /*
     * Generic two-operand compare-and-branch operation.  Provide a "condition"
     * fragment that specifies the comparison to perform, e.g. for
     * "if-le" you would use "le".
     *
     * For: if-eq, if-ne, if-lt, if-ge, if-gt, if-le
     */
    /* if-cmp vA, vB, +CCCC */
    EXT     a2, rINST, 8, 4             # a2 <- A
    EXT     a3, rINST, 12, 4            # a3 <- B
    lh      rINST, 2(rPC)               # rINST <- offset (sign-extended CCCC)
    GET_VREG a0, a2                     # a0 <- vA
    GET_VREG a1, a3                     # a1 <- vB
    b${condition} a0, a1, MterpCommonTakenBranchNoFlags
    li      t4, JIT_CHECK_OSR           # possible OSR re-entry?
    beq    rPROFILE, t4, .L_check_not_taken_osr
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    GOTO_OPCODE t4                      # jump to next instruction

%def zcmp(condition=""):
    /*
     * Generic one-operand compare-and-branch operation.  Provide a "condition"
     * fragment that specifies the comparison to perform, e.g. for
     * "if-lez" you would use "le".
     *
     * For: if-eqz, if-nez, if-ltz, if-gez, if-gtz, if-lez
     */
    /* if-cmp vAA, +BBBB */
    srli     a2, rINST, 8                # a2 <- AA
    lh       rINST, 2(rPC)               # rINST <- offset (sign-extended BBBB)
    GET_VREG a0, a2                     # a0 <- vAA
    b${condition}z a0, MterpCommonTakenBranchNoFlags
    li       t4, JIT_CHECK_OSR           # possible OSR re-entry?
    beq      rPROFILE, t4, .L_check_not_taken_osr
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    GOTO_OPCODE t4                      # jump to next instruction

%def op_goto():
    /*
     * Unconditional branch, 8-bit offset.
     *
     * The branch distance is a signed code-unit offset, which we need to
     * double to get a byte offset.
     */
     /* goto +AA */
    lb      rINST, 1(rPC)                 # sign extended load 8-bit offset into rINST

    j       MterpCommonTakenBranchNoFlags

%def op_goto_16():
    /*
     * Unconditional branch, 16-bit offset.
     *
     * The branch distance is a signed code-unit offset, which we need to
     * double to get a byte offset.
     */
    /* goto/16 +AAAA */
    lh      rINST, 2(rPC)               # rINST <- offset (sign-extended AAAA)
    j       MterpCommonTakenBranchNoFlags

%def op_goto_32():
    /*
     * Unconditional branch, 32-bit offset.
     *
     * The branch distance is a signed code-unit offset, which we need to
     * double to get a byte offset.
     *
     * Unlike most opcodes, this one is allowed to branch to itself, so
     * our "backward branch" test must be "<=0" instead of "<0".
     */
    /* goto/32 +AAAAAAAA */
    lhu      rINST, 2(rPC)               # rINST <- aaaa (low)
    lhu      a1, 4(rPC)                  # a1 <- AAAA (high)
    slli     a1, a1, 16                  # rINST <- offset (sign-extended AAAAaaaa)
    or       rINST, rINST, a1
    j       MterpCommonTakenBranchNoFlags

%def op_if_eq():
%  bincmp(condition="eq")

%def op_if_eqz():
%  zcmp(condition="eq")

%def op_if_ge():
%  bincmp(condition="ge")

%def op_if_gez():
%  zcmp(condition="ge")

%def op_if_gt():
%  bincmp(condition="gt")

%def op_if_gtz():
%  zcmp(condition="gt")

%def op_if_le():
%  bincmp(condition="le")

%def op_if_lez():
%  zcmp(condition="le")

%def op_if_lt():
%  bincmp(condition="lt")

%def op_if_ltz():
%  zcmp(condition="lt")

%def op_if_ne():
%  bincmp(condition="ne")

%def op_if_nez():
%  zcmp(condition="ne")

%def op_packed_switch(func="MterpDoPackedSwitch"):
    /*
     * Handle a packed-switch or sparse-switch instruction.  In both cases
     * we decode it and hand it off to a helper function.
     *
     * We don't really expect backward branches in a switch statement, but
     * they're perfectly legal, so we check for them here.
     *
     * for: packed-switch, sparse-switch
     */
    /* op vAA, +BBBBBBBB */
    .extern $func
    lhu     a0, 2(rPC)                  # a0 <- 000000000000bbbb (lo)
    lh      a1, 4(rPC)                  # a1 <- ssssssssssssBBBB (hi)
    srli    a3, rINST, 8                # a3 <- AA
    slli    a1, a1, 16
    or      a0, a1, a0                  # a0<- ssssssssBBBBbbbb
    GET_VREG a1, a3                     # a1 <- vAA
    DLSA    a0, a0, rPC, 1              # a0 <- PC + ssssssssBBBBbbbb*2
    jal     $func                       # v0 <- code-unit branch offset
    move    rINST, a0
    j       MterpCommonTakenBranchNoFlags

%def op_return(instr="GET_VREG"):
    /*
     * Return a 32-bit value.
     *
     * for: return (sign-extend), return-object (zero-extend)
     */
    /* op vAA */
    .extern MterpThreadFenceForConstructor
    .extern MterpSuspendCheck
    jal     MterpThreadFenceForConstructor

    lw      ra, THREAD_FLAGS_OFFSET(rSELF)
    move    a0, rSELF
    andi    ra, ra, THREAD_SUSPEND_OR_CHECKPOINT_REQUEST
    beqz    ra, 1f
    jal     MterpSuspendCheck           # (self)
1:
    srli    a2, rINST, 8                # a2 <- AA
    $instr  a0, a2                      # a0 <- vAA
    j       MterpReturn

%def op_return_object():
%  op_return(instr="GET_VREG_U")

%def op_return_void():
    .extern MterpThreadFenceForConstructor
    .extern MterpSuspendCheck
    jal     MterpThreadFenceForConstructor

    lw      ra, THREAD_FLAGS_OFFSET(rSELF)
    move    a0, rSELF
    andi    ra, ra, THREAD_SUSPEND_OR_CHECKPOINT_REQUEST
    beqz    ra, 1f
    jal     MterpSuspendCheck           # (self)
1:
    li      a0, 0
    j       MterpReturn

%def op_return_void_no_barrier():
    .extern MterpSuspendCheck
    lw      ra, THREAD_FLAGS_OFFSET(rSELF)
    move    a0, rSELF
    andi    ra, ra, THREAD_SUSPEND_OR_CHECKPOINT_REQUEST
    beqz    ra, 1f
    jal     MterpSuspendCheck           # (self)
1:
    li      a0, 0
    j       MterpReturn

%def op_return_wide():
    /*
     * Return a 64-bit value.
     */
    /* return-wide vAA */
    /* op vAA */
    .extern MterpThreadFenceForConstructor
    .extern MterpSuspendCheck
    lw      ra, THREAD_FLAGS_OFFSET(rSELF)
    jal     MterpThreadFenceForConstructor

    move    a0, rSELF
    andi    ra, ra, THREAD_SUSPEND_OR_CHECKPOINT_REQUEST
    beqz    ra, 1f
    jal     MterpSuspendCheck           # (self)
1:
    srli    a2, rINST, 8                # a2 <- AA
    GET_VREG_WIDE a0, a2                # a0 <- vAA
    j       MterpReturn

%def op_sparse_switch():
%  op_packed_switch(func="MterpDoSparseSwitch")

%def op_throw():
    /*
     * Throw an exception object in the current thread.
     */
    /* throw vAA */
    EXPORT_PC
    srli     a2, rINST, 8                # a2 <- AA
    GET_VREG_U a0, a2                   # a0 <- vAA (exception object)
    sd      a0, THREAD_EXCEPTION_OFFSET(rSELF)  # thread->exception <- obj
    beqz    a0, common_errNullObject

    j       MterpException
