%def fbinop(instr=""):
    /*:
     * Generic 32-bit floating-point operation.
     *
     * For: add-float, sub-float, mul-float, div-float.
     * form: <op> f0, f0, f1
     */
    /* binop vAA, vBB, vCC */
    srli    a4, rINST, 8                # a4 <- AA
    lbu     a2, 2(rPC)                  # a2 <- BB
    lbu     a3, 3(rPC)                  # a3 <- CC
    GET_VREG_FLOAT f0, a2               # f0 <- vBB
    GET_VREG_FLOAT f1, a3               # f1 <- vCC
    $instr                              # f0 <- f0 op f1
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_FLOAT f0, a4               # vAA <- f0
    GOTO_OPCODE t4                      # jump to next instruction

%def fbinop2addr(instr=""):
    /*:
     * Generic 32-bit "/2addr" floating-point operation.
     *
     * For: add-float/2addr, sub-float/2addr, mul-float/2addr, div-float/2addr.
     * form: <op> f0, f0, f1
     */
    /* binop/2addr vA, vB */
    EXT     a2, rINST, 8, 4             # a2 <- A
    EXT     a3, rINST, 12, 4            # a3 <- B
    GET_VREG_FLOAT f0, a2               # f0 <- vA
    GET_VREG_FLOAT f1, a3               # f1 <- vB
    $instr                              # f0 <- f0 op f1
    FETCH_ADVANCE_INST 1                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_FLOAT f0, a2               # vA <- f0
    GOTO_OPCODE t4                      # jump to next instruction

%def fbinopWide(instr=""):
    /*:
     * Generic 64-bit floating-point operation.
     *
     * For: add-double, sub-double, mul-double, div-double.
     * form: <op> f0, f0, f1
     */
    /* binop vAA, vBB, vCC */
    srli    a4, rINST, 8                # a4 <- AA
    lbu     a2, 2(rPC)                  # a2 <- BB
    lbu     a3, 3(rPC)                  # a3 <- CC
    GET_VREG_DOUBLE f0, a2              # f0 <- vBB
    GET_VREG_DOUBLE f1, a3              # f1 <- vCC
    $instr                              # f0 <- f0 op f1
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_DOUBLE f0, a4              # vAA <- f0
    GOTO_OPCODE t4                      # jump to next instruction

%def fbinopWide2addr(instr=""):
    /*:
     * Generic 64-bit "/2addr" floating-point operation.
     *
     * For: add-double/2addr, sub-double/2addr, mul-double/2addr, div-double/2addr.
     * form: <op> f0, f0, f1
     */
    /* binop/2addr vA, vB */
    EXT     a2, rINST, 8, 4             # a2 <- A
    EXT     a3, rINST, 12, 4            # a3 <- B
    GET_VREG_DOUBLE f0, a2              # f0 <- vA
    GET_VREG_DOUBLE f1, a3              # f1 <- vB
    $instr                              # f0 <- f0 op f1
    FETCH_ADVANCE_INST 1                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_DOUBLE f0, a2              # vA <- f0
    GOTO_OPCODE t4                      # jump to next instruction

%def fcmp(gt_bias=""):
    /*
     * Compare two floating-point values.  Puts 0, 1, or -1 into the
     * destination register based on the results of the comparison.
     *
     * For: cmpl-float, cmpg-float
     */
    /* op vAA, vBB, vCC */
    srli     a4, rINST, 8                # a4 <- AA
    lbu     a2, 2(rPC)                  # a2 <- BB
    lbu     a3, 3(rPC)                  # a3 <- CC
    GET_VREG_FLOAT f0, a2               # f0 <- vBB
    GET_VREG_FLOAT f1, a3               # f1 <- vCC
    feq.s t4, f0, f1
    li      a0, 0
    bnez    t4, 1f                      # done if vBB == vCC (ordered)

    .if $gt_bias
    flt.s t4, f0, f1
    li      a0, -1
    bnez    t4, 1f                      # done if vBB < vCC (ordered)
    li      a0, 1                       # vBB > vCC or unordered
    .else
    flt.s t4, f1, f0
    li      a0, 1
    bnez  t4, 1f                        # done if vBB > vCC (ordered)
    li      a0, -1                      # vBB < vCC or unordered
    .endif
1:
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG a0, a4                     # vAA <- a0
    GOTO_OPCODE t4                      # jump to next instruction

%def fcmpWide(gt_bias=""):
    /*
     * Compare two floating-point values.  Puts 0, 1, or -1 into the
     * destination register based on the results of the comparison.
     *
     * For: cmpl-double, cmpg-double
     */
    /* op vAA, vBB, vCC */
    srli    a4, rINST, 8                # a4 <- AA
    lbu     a2, 2(rPC)                  # a2 <- BB
    lbu     a3, 3(rPC)                  # a3 <- CC
    GET_VREG_DOUBLE f0, a2              # f0 <- vBB
    GET_VREG_DOUBLE f1, a3              # f1 <- vCC
    feq.d   t4, f0, f1
    li      a0, 0
    bnez    t4, 1f                      # done if vBB == vCC (ordered)

    .if $gt_bias
    flt.d   t4, f0, f1
    li      a0, -1
    bnez    t4, 1f                      # done if vBB < vCC (ordered)
    li      a0, 1                       # vBB > vCC or unordered
    .else
    flt.d   t4, f1, f0
    li      a0, 1
    bnez    t4, 1f                      # done if vBB > vCC (ordered)
    li      a0, -1                      # vBB < vCC or unordered
    .endif
1:
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG a0, a4                     # vAA <- a0
    GOTO_OPCODE t4                      # jump to next instruction

%def fcvtFooter(suffix="", valreg=""):
    /*
     * Stores a specified register containing the result of conversion
     * from or to a floating-point type and jumps to the next instruction.
     *
     * Expects a1 to contain the destination Dalvik register number.
     * a1 is set up by fcvtHeader.S.
     *
     * For: int-to-float, int-to-double, long-to-float, long-to-double,
     *      float-to-int, float-to-long, float-to-double, double-to-int,
     *      double-to-long, double-to-float, neg-float, neg-double.
     *
     * Note that this file can't be included after a break in other files
     * and in those files its contents appear as a copy.
     * See: float-to-int, float-to-long, double-to-int, double-to-long.
     */
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG$suffix $valreg, a1
    GOTO_OPCODE t4                      # jump to next instruction

%def fcvtHeader(suffix="", valreg=""):
    /*
     * Loads a specified register from vB. Used primarily for conversions
     * from or to a floating-point type.
     *
     * Sets up a1 = A and a2 = B. a2 is later used by fcvtFooter.S to
     * store the result in vA and jump to the next instruction.
     *
     * For: int-to-float, int-to-double, long-to-float, long-to-double,
     *      float-to-int, float-to-long, float-to-double, double-to-int,
     *      double-to-long, double-to-float, neg-float, neg-double.
     */
    EXT     a1, rINST, 8, 4             # a1 <- A
    srli     a2, rINST, 12               # a2 <- B
    GET_VREG$suffix $valreg, a2
    FETCH_ADVANCE_INST 1                # advance rPC, load rINST

%def op_add_double():
%  fbinopWide(instr="fadd.d f0, f0, f1")

%def op_add_double_2addr():
%  fbinopWide2addr(instr="fadd.d f0, f0, f1")

%def op_add_float():
%  fbinop(instr="fadd.s f0, f0, f1")

%def op_add_float_2addr():
%  fbinop2addr(instr="fadd.s f0, f0, f1")

%def op_cmpg_double():
%  fcmpWide(gt_bias="1")

%def op_cmpg_float():
%  fcmp(gt_bias="1")

%def op_cmpl_double():
%  fcmpWide(gt_bias="0")

%def op_cmpl_float():
%  fcmp(gt_bias="0")

%def op_div_double():
%  fbinopWide(instr="fdiv.d f0, f0, f1")

%def op_div_double_2addr():
%  fbinopWide2addr(instr="fdiv.d f0, f0, f1")

%def op_div_float():
%  fbinop(instr="fdiv.s f0, f0, f1")

%def op_div_float_2addr():
%  fbinop2addr(instr="fdiv.s f0, f0, f1")

%def op_double_to_float():
    /*
     * Conversion from or to floating-point happens in a floating-point register.
     * Therefore we load the input and store the output into or from a
     * floating-point register irrespective of the type.
     */
%  fcvtHeader(suffix="_DOUBLE", valreg="f0")
   fcvt.s.d f0, f0
%  fcvtFooter(suffix="_FLOAT", valreg="f0")

%def op_double_to_int():
%  fcvtHeader(suffix="_DOUBLE", valreg="f0")
   xor     a0, a0, a0
   feq.d   t4, f0, f0        # check if fp(B) is NaN
   beqz    t4, 1f            # return 0 if fp(B) is NaN
   fcvt.w.d a0, f0, rtz
1:
%  fcvtFooter(suffix="", valreg="a0")

%def op_double_to_long():
%  fcvtHeader(suffix="_DOUBLE", valreg="f0")
   xor     a0, a0, a0
   feq.d   t4, f0, f0        # check if fp(B) is NaN
   beqz    t4, 1f            # return 0 if fp(B) is NaN
   fcvt.l.d a0, f0, rtz
1:
%  fcvtFooter(suffix="_WIDE", valreg="a0")

%def op_float_to_double():
    /*
     * Conversion from or to floating-point happens in a floating-point register.
     * Therefore we load the input and store the output into or from a
     * floating-point register irrespective of the type.
     */
%  fcvtHeader(suffix="_FLOAT", valreg="f0")
    fcvt.d.s f0, f0
%  fcvtFooter(suffix="_DOUBLE", valreg="f0")

%def op_float_to_int():
%  fcvtHeader(suffix="_FLOAT", valreg="f0")
   xor     a0, a0, a0
   feq.s   t4, f0, f0        # check if fp(B) is NaN
   beqz    t4, 1f            # return 0 if fp(B) is NaN
   fcvt.w.s a0, f0, rtz
1:
%  fcvtFooter(suffix="", valreg="a0")

%def op_float_to_long():
%  fcvtHeader(suffix="_FLOAT", valreg="f0")
   xor     a0, a0, a0
   feq.s   t4, f0, f0        # check if fp(B) is NaN
   beqz    t4, 1f            # return 0 if fp(B) is NaN
   fcvt.l.s a0, f0, rtz
1:
%  fcvtFooter(suffix="_WIDE", valreg="a0")

%def op_int_to_double():
    /*
     * Conversion from or to floating-point happens in a floating-point register.
     * Therefore we load the input and store the output into or from a
     * floating-point register irrespective of the type.
     */
%  fcvtHeader(suffix="", valreg="a0")
   fcvt.d.w f0, a0
%  fcvtFooter(suffix="_DOUBLE", valreg="f0")

%def op_int_to_float():
    /*
     * Conversion from or to floating-point happens in a floating-point register.
     * Therefore we load the input and store the output into or from a
     * floating-point register irrespective of the type.
     */
%  fcvtHeader(suffix="", valreg="a0")
   fcvt.s.w f0, a0
%  fcvtFooter(suffix="_FLOAT", valreg="f0")

%def op_long_to_double():
    /*
     * Conversion from or to floating-point happens in a floating-point register.
     * Therefore we load the input and store the output into or from a
     * floating-point register irrespective of the type.
     */
%  fcvtHeader(suffix="_WIDE", valreg="a0")
   fcvt.d.l f0, a0
%  fcvtFooter(suffix="_DOUBLE", valreg="f0")

%def op_long_to_float():
    /*
     * Conversion from or to floating-point happens in a floating-point register.
     * Therefore we load the input and store the output into or from a
     * floating-point register irrespective of the type.
     */
%  fcvtHeader(suffix="_WIDE", valreg="a0")
   fcvt.s.l f0, a0
%  fcvtFooter(suffix="_FLOAT", valreg="f0")

%def op_mul_double():
%  fbinopWide(instr="fmul.d f0, f0, f1")

%def op_mul_double_2addr():
%  fbinopWide2addr(instr="fmul.d f0, f0, f1")

%def op_mul_float():
%  fbinop(instr="fmul.s f0, f0, f1")

%def op_mul_float_2addr():
%  fbinop2addr(instr="fmul.s f0, f0, f1")

%def op_neg_double():
%  fcvtHeader(suffix="_DOUBLE", valreg="f0")
    fneg.d   f0, f0
%  fcvtFooter(suffix="_DOUBLE", valreg="f0")

%def op_neg_float():
%  fcvtHeader(suffix="_FLOAT", valreg="f0")
    fneg.s   f0, f0
%  fcvtFooter(suffix="_FLOAT", valreg="f0")

%def op_rem_double():
    /* rem-double vAA, vBB, vCC */
    .extern fmod
    lbu     a2, 2(rPC)                  # a2 <- BB
    lbu     a3, 3(rPC)                  # a3 <- CC
    GET_VREG_DOUBLE f10, a2             # fa0 <- vBB
    GET_VREG_DOUBLE f11, a3             # fa1 <- vCC
    call     fmod@plt                    # f0 <- f12 op f13
    srli     a4, rINST, 8                # a4 <- AA
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_DOUBLE f10, a4              # vAA <- fa0
    GOTO_OPCODE t4                      # jump to next instruction

%def op_rem_double_2addr():
    /* rem-double/2addr vA, vB */
    .extern fmod
    EXT     a2, rINST, 8, 4             # a2 <- A
    EXT     a3, rINST, 12, 4            # a3 <- B
    GET_VREG_DOUBLE f10, a2             # fa0 <- vA
    GET_VREG_DOUBLE f11, a3             # fa1 <- vB
    call     fmod@plt                   # fa0 <- fa0 op fa1
    EXT     a2, rINST, 8, 4             # a2 <- A
    FETCH_ADVANCE_INST 1                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_DOUBLE f10, a2              # vA <- fa0
    GOTO_OPCODE t4                      # jump to next instruction

%def op_rem_float():
    /* rem-float vAA, vBB, vCC */
    .extern fmodf
    lbu     a2, 2(rPC)                  # a2 <- BB
    lbu     a3, 3(rPC)                  # a3 <- CC
    GET_VREG_FLOAT f10, a2              # fa0 <- vBB
    GET_VREG_FLOAT f11, a3              # fa1 <- vCC
    call     fmodf@plt                   # fa0 <- fa0 op fa1
    srli     a4, rINST, 8                # a4 <- AA
    FETCH_ADVANCE_INST 2                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_FLOAT f10, a4               # vAA <- fa0
    GOTO_OPCODE t4                      # jump to next instruction

%def op_rem_float_2addr():
    /* rem-float/2addr vA, vB */
    .extern fmodf
    EXT     a2, rINST, 8, 4             # a2 <- A
    EXT     a3, rINST, 12, 4            # a3 <- B
    GET_VREG_FLOAT f10, a2              # fa0 <- vA
    GET_VREG_FLOAT f11, a3              # fa1 <- vB
    call     fmodf@plt                  # fa0 <- f12 op f13
    EXT     a2, rINST, 8, 4             # a2 <- A
    FETCH_ADVANCE_INST 1                # advance rPC, load rINST
    GET_INST_OPCODE t4                  # extract opcode from rINST
    SET_VREG_FLOAT f10, a2               # vA <- fa0
    GOTO_OPCODE t4                      # jump to next instruction

%def op_sub_double():
%  fbinopWide(instr="fsub.d f0, f0, f1")

%def op_sub_double_2addr():
%  fbinopWide2addr(instr="fsub.d f0, f0, f1")

%def op_sub_float():
%  fbinop(instr="fsub.s f0, f0, f1")

%def op_sub_float_2addr():
%  fbinop2addr(instr="fsub.s f0, f0, f1")
