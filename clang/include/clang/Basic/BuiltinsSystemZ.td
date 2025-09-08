//===--- BuiltinsSystemZ.td - SystemZ Builtin function database -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

def Tbegin : TargetBuiltin {
  let Spellings = ["__builtin_tbegin"];
  let Prototype = "int(void *)";
  let Attributes = [ReturnsTwice];
  let Features = "transactional-execution";
}

def TbeginNofloat : TargetBuiltin {
  let Spellings = ["__builtin_tbegin_nofloat"];
  let Prototype = "int(void *)";
  let Attributes = [ReturnsTwice];
  let Features = "transactional-execution";
}

def Tbeginc : TargetBuiltin {
  let Spellings = ["__builtin_tbeginc"];
  let Prototype = "void()";
  let Attributes = [NoThrow, ReturnsTwice];
  let Features = "transactional-execution";
}

def Tabort : TargetBuiltin {
  let Spellings = ["__builtin_tabort"];
  let Prototype = "void(int)";
  let Attributes = [NoReturn];
  let Features = "transactional-execution";
}

def Tend : TargetBuiltin {
  let Spellings = ["__builtin_tend"];
  let Prototype = "int()";
  let Attributes = [NoThrow];
  let Features = "transactional-execution";
}

def TxNestingDepth : TargetBuiltin {
  let Spellings = ["__builtin_tx_nesting_depth"];
  let Prototype = "int()";
  let Attributes = [NoThrow, Const];
  let Features = "transactional-execution";
}

def TxAssist : TargetBuiltin {
  let Spellings = ["__builtin_tx_assist"];
  let Prototype = "void(int)";
  let Attributes = [NoThrow];
  let Features = "transactional-execution";
}

def NonTxStore : TargetBuiltin {
  let Spellings = ["__builtin_non_tx_store"];
  let Prototype = "void(unsigned long int *, unsigned long int)";
  let Features = "transactional-execution";
}

def Lcbb : TargetBuiltin {
  let Spellings = ["__builtin_s390_lcbb"];
  let Prototype = "unsigned int(void const *, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vlbb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vlbb"];
  let Prototype = "_Vector<16, signed char>(void const *, int)";
  let Features = "vector";
}

def Vll : TargetBuiltin {
  let Spellings = ["__builtin_s390_vll"];
  let Prototype = "_Vector<16, signed char>(unsigned int, void const *)";
  let Features = "vector";
}

def Vstl : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstl"];
  let Prototype = "void(_Vector<16, signed char>, unsigned int, void *)";
  let Features = "vector";
}

def Vperm : TargetBuiltin {
  let Spellings = ["__builtin_s390_vperm"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vpdi : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpdi"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vpksh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpksh"];
  let Prototype = "_Vector<16, signed char>(_Vector<8, signed short>, _Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vpkshs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpkshs"];
  let Prototype = "_Vector<16, signed char>(_Vector<8, signed short>, _Vector<8, signed short>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vpksf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpksf"];
  let Prototype = "_Vector<8, signed short>(_Vector<4, signed int>, _Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vpksfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpksfs"];
  let Prototype = "_Vector<8, signed short>(_Vector<4, signed int>, _Vector<4, signed int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vpksg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpksg"];
  let Prototype = "_Vector<4, signed int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vpksgs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpksgs"];
  let Prototype = "_Vector<4, signed int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vpklsh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpklsh"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vpklshs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpklshs"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vpklsf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpklsf"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vpklsfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpklsfs"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vpklsg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpklsg"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vpklsgs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpklsgs"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vuphb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuphb"];
  let Prototype = "_Vector<8, signed short>(_Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vuphh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuphh"];
  let Prototype = "_Vector<4, signed int>(_Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vuphf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuphf"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vuplb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplb"];
  let Prototype = "_Vector<8, signed short>(_Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vuplhw : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplhw"];
  let Prototype = "_Vector<4, signed int>(_Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vuplf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplf"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vuplhb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplhb"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vuplhh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplhh"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vuplhf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplhf"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vupllb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vupllb"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vupllh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vupllh"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vupllf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vupllf"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vaq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vaq"];
  let Prototype = "signed long long int(signed long long int, signed long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vacq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vacq"];
  let Prototype = "unsigned long long int(unsigned long long int, unsigned long long int, unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vaccb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vaccb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vacch : TargetBuiltin {
  let Spellings = ["__builtin_s390_vacch"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vaccf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vaccf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vaccg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vaccg"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vaccq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vaccq"];
  let Prototype = "unsigned long long int(unsigned long long int, unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vacccq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vacccq"];
  let Prototype = "unsigned long long int(unsigned long long int, unsigned long long int, unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vavgb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavgb"];
  let Prototype = "_Vector<16, signed char>(_Vector<16, signed char>, _Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vavgh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavgh"];
  let Prototype = "_Vector<8, signed short>(_Vector<8, signed short>, _Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vavgf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavgf"];
  let Prototype = "_Vector<4, signed int>(_Vector<4, signed int>, _Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vavgg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavgg"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vavglb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavglb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vavglh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavglh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vavglf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavglf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vavglg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavglg"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vceqbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vceqbs"];
  let Prototype = "_Vector<16, signed char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vceqhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vceqhs"];
  let Prototype = "_Vector<8, signed short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vceqfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vceqfs"];
  let Prototype = "_Vector<4, signed int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vceqgs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vceqgs"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vchbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchbs"];
  let Prototype = "_Vector<16, signed char>(_Vector<16, signed char>, _Vector<16, signed char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vchhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchhs"];
  let Prototype = "_Vector<8, signed short>(_Vector<8, signed short>, _Vector<8, signed short>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vchfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchfs"];
  let Prototype = "_Vector<4, signed int>(_Vector<4, signed int>, _Vector<4, signed int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vchgs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchgs"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vchlbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchlbs"];
  let Prototype = "_Vector<16, signed char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vchlhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchlhs"];
  let Prototype = "_Vector<8, signed short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vchlfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchlfs"];
  let Prototype = "_Vector<4, signed int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vchlgs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchlgs"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vcksm : TargetBuiltin {
  let Spellings = ["__builtin_s390_vcksm"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vclzb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclzb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vclzh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclzh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vclzf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclzf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vclzg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclzg"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vctzb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vctzb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vctzh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vctzh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vctzf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vctzf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vctzg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vctzg"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Verimb : TargetBuiltin {
  let Spellings = ["__builtin_s390_verimb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Verimh : TargetBuiltin {
  let Spellings = ["__builtin_s390_verimh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Verimf : TargetBuiltin {
  let Spellings = ["__builtin_s390_verimf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Verimg : TargetBuiltin {
  let Spellings = ["__builtin_s390_verimg"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Verllb : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, unsigned char)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Verllh : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, unsigned char)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Verllf : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, unsigned char)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Verllg : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllg"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, unsigned char)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Verllvb : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllvb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Verllvh : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllvh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Verllvf : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllvf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Verllvg : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllvg"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vgfmb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmb"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vgfmh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmh"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vgfmf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmf"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vgfmg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmg"];
  let Prototype = "unsigned long long int(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vgfmab : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmab"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vgfmah : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmah"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vgfmaf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmaf"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vgfmag : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmag"];
  let Prototype = "unsigned long long int(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmahb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmahb"];
  let Prototype = "_Vector<16, signed char>(_Vector<16, signed char>, _Vector<16, signed char>, _Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmahh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmahh"];
  let Prototype = "_Vector<8, signed short>(_Vector<8, signed short>, _Vector<8, signed short>, _Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmahf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmahf"];
  let Prototype = "_Vector<4, signed int>(_Vector<4, signed int>, _Vector<4, signed int>, _Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmalhb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalhb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmalhh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalhh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmalhf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalhf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmaeb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaeb"];
  let Prototype = "_Vector<8, signed short>(_Vector<16, signed char>, _Vector<16, signed char>, _Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmaeh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaeh"];
  let Prototype = "_Vector<4, signed int>(_Vector<8, signed short>, _Vector<8, signed short>, _Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmaef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaef"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<4, signed int>, _Vector<4, signed int>, _Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmaleb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaleb"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmaleh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaleh"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmalef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalef"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmaob : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaob"];
  let Prototype = "_Vector<8, signed short>(_Vector<16, signed char>, _Vector<16, signed char>, _Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmaoh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaoh"];
  let Prototype = "_Vector<4, signed int>(_Vector<8, signed short>, _Vector<8, signed short>, _Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmaof : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaof"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<4, signed int>, _Vector<4, signed int>, _Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmalob : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalob"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmaloh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaloh"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmalof : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalof"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmhb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmhb"];
  let Prototype = "_Vector<16, signed char>(_Vector<16, signed char>, _Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmhh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmhh"];
  let Prototype = "_Vector<8, signed short>(_Vector<8, signed short>, _Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmhf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmhf"];
  let Prototype = "_Vector<4, signed int>(_Vector<4, signed int>, _Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmlhb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlhb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmlhh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlhh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmlhf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlhf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmeb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmeb"];
  let Prototype = "_Vector<8, signed short>(_Vector<16, signed char>, _Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmeh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmeh"];
  let Prototype = "_Vector<4, signed int>(_Vector<8, signed short>, _Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmef"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<4, signed int>, _Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmleb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmleb"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmleh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmleh"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmlef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlef"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmob : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmob"];
  let Prototype = "_Vector<8, signed short>(_Vector<16, signed char>, _Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmoh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmoh"];
  let Prototype = "_Vector<4, signed int>(_Vector<8, signed short>, _Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmof : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmof"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<4, signed int>, _Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmlob : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlob"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmloh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmloh"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vmlof : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlof"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsq"];
  let Prototype = "signed long long int(signed long long int, signed long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsbcbiq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsbcbiq"];
  let Prototype = "unsigned long long int(unsigned long long int, unsigned long long int, unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsbiq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsbiq"];
  let Prototype = "unsigned long long int(unsigned long long int, unsigned long long int, unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vscbib : TargetBuiltin {
  let Spellings = ["__builtin_s390_vscbib"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vscbih : TargetBuiltin {
  let Spellings = ["__builtin_s390_vscbih"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vscbif : TargetBuiltin {
  let Spellings = ["__builtin_s390_vscbif"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vscbig : TargetBuiltin {
  let Spellings = ["__builtin_s390_vscbig"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vscbiq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vscbiq"];
  let Prototype = "unsigned long long int(unsigned long long int, unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsl : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsl"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vslb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vslb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsldb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsldb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsra : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsra"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsrab : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsrab"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsrl : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsrl"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsrlb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsrlb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsumb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsumb"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsumh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsumh"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsumgh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsumgh"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsumgf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsumgf"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsumqf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsumqf"];
  let Prototype = "unsigned long long int(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vsumqg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsumqg"];
  let Prototype = "unsigned long long int(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vtm : TargetBuiltin {
  let Spellings = ["__builtin_s390_vtm"];
  let Prototype = "int(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfaeb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaeb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfaebs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaebs"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfaeh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaeh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfaehs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaehs"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfaef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaef"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfaefs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaefs"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfaezb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaezb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfaezbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaezbs"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfaezh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaezh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfaezhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaezhs"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfaezf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaezf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfaezfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaezfs"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeeb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeeb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeebs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeebs"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeeh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeeh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeehs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeehs"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeef"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeefs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeefs"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeezb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeezb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeezbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeezbs"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeezh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeezh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeezhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeezhs"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeezf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeezf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeezfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeezfs"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeneb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeneb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfenebs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenebs"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfeneh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeneh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfenehs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenehs"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfenef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenef"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfenefs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenefs"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfenezb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenezb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfenezbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenezbs"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfenezh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenezh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfenezhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenezhs"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfenezf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenezf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfenezfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenezfs"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vistrb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vistrb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vistrbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vistrbs"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vistrh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vistrh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vistrhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vistrhs"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vistrf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vistrf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vistrfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vistrfs"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vstrcb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrcb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vstrcbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrcbs"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vstrch : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrch"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vstrchs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrchs"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vstrcf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrcf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vstrcfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrcfs"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vstrczb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrczb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vstrczbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrczbs"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vstrczh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrczh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vstrczhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrczhs"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vstrczf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrczf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vstrczfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrczfs"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfcedbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfcedbs"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<2, double>, _Vector<2, double>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfchdbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfchdbs"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<2, double>, _Vector<2, double>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfchedbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfchedbs"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<2, double>, _Vector<2, double>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfidb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfidb"];
  let Prototype = "_Vector<2, double>(_Vector<2, double>, int, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vflndb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vflndb"];
  let Prototype = "_Vector<2, double>(_Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vflpdb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vflpdb"];
  let Prototype = "_Vector<2, double>(_Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfmadb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmadb"];
  let Prototype = "_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfmsdb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmsdb"];
  let Prototype = "_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vfsqdb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfsqdb"];
  let Prototype = "_Vector<2, double>(_Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vftcidb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vftcidb"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<2, double>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vlrlr : TargetBuiltin {
  let Spellings = ["__builtin_s390_vlrlr"];
  let Prototype = "_Vector<16, signed char>(unsigned int, void const *)";
  let Features = "vector-enhancements-1";
}

def Vstrlr : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrlr"];
  let Prototype = "void(_Vector<16, signed char>, unsigned int, void *)";
  let Features = "vector-enhancements-1";
}

def Vbperm : TargetBuiltin {
  let Spellings = ["__builtin_s390_vbperm"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vmslg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmslg"];
  let Prototype = "unsigned long long int(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, unsigned long long int, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfmaxdb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmaxdb"];
  let Prototype = "_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfmindb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmindb"];
  let Prototype = "_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfnmadb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfnmadb"];
  let Prototype = "_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfnmsdb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfnmsdb"];
  let Prototype = "_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfcesbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfcesbs"];
  let Prototype = "_Vector<4, signed int>(_Vector<4, float>, _Vector<4, float>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfchsbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfchsbs"];
  let Prototype = "_Vector<4, signed int>(_Vector<4, float>, _Vector<4, float>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfchesbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfchesbs"];
  let Prototype = "_Vector<4, signed int>(_Vector<4, float>, _Vector<4, float>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfisb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfisb"];
  let Prototype = "_Vector<4, float>(_Vector<4, float>, int, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfmaxsb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmaxsb"];
  let Prototype = "_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfminsb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfminsb"];
  let Prototype = "_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vflnsb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vflnsb"];
  let Prototype = "_Vector<4, float>(_Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vflpsb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vflpsb"];
  let Prototype = "_Vector<4, float>(_Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfmasb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmasb"];
  let Prototype = "_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfmssb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmssb"];
  let Prototype = "_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfnmasb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfnmasb"];
  let Prototype = "_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfnmssb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfnmssb"];
  let Prototype = "_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vfsqsb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfsqsb"];
  let Prototype = "_Vector<4, float>(_Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vftcisb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vftcisb"];
  let Prototype = "_Vector<4, signed int>(_Vector<4, float>, int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-1";
}

def Vsld : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsld"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-2";
}

def Vsrd : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsrd"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-2";
}

def Vstrsb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrsb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-2";
}

def Vstrsh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrsh"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<16, unsigned char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-2";
}

def Vstrsf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrsf"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<16, unsigned char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-2";
}

def Vstrszb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrszb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-2";
}

def Vstrszh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrszh"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<16, unsigned char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-2";
}

def Vstrszf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrszf"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<16, unsigned char>, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-2";
}

def Vlbrh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vlbrh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vlbrf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vlbrf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vlbrg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vlbrg"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vlbrq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vlbrq"];
  let Prototype = "unsigned long long int(unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector";
}

def Vclfnhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclfnhs"];
  let Prototype = "_Vector<4, float>(_Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "nnp-assist";
}

def Vclfnls : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclfnls"];
  let Prototype = "_Vector<4, float>(_Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "nnp-assist";
}

def Vcrnfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vcrnfs"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<4, float>, _Vector<4, float>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "nnp-assist";
}

def Vcfn : TargetBuiltin {
  let Spellings = ["__builtin_s390_vcfn"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "nnp-assist";
}

def Vcnf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vcnf"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "nnp-assist";
}

def Bdepg : TargetBuiltin {
  let Spellings = ["__builtin_s390_bdepg"];
  let Prototype = "unsigned long int(unsigned long int, unsigned long int)";
  let Attributes = [NoThrow, Const];
  let Features = "miscellaneous-extensions-4";
}

def Bextg : TargetBuiltin {
  let Spellings = ["__builtin_s390_bextg"];
  let Prototype = "unsigned long int(unsigned long int, unsigned long int)";
  let Attributes = [NoThrow, Const];
  let Features = "miscellaneous-extensions-4";
}

def Vgemb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgemb"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vgemh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgemh"];
  let Prototype = "_Vector<8, unsigned short>(_Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vgemf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgemf"];
  let Prototype = "_Vector<4, unsigned int>(_Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vgemg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgemg"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vgemq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgemq"];
  let Prototype = "unsigned long long int(_Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vuplg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplg"];
  let Prototype = "signed long long int(_Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vupllg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vupllg"];
  let Prototype = "unsigned long long int(_Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vuphg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuphg"];
  let Prototype = "signed long long int(_Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vuplhg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplhg"];
  let Prototype = "unsigned long long int(_Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vavgq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavgq"];
  let Prototype = "signed long long int(signed long long int, signed long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vavglq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavglq"];
  let Prototype = "unsigned long long int(unsigned long long int, unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Veval : TargetBuiltin {
  let Spellings = ["__builtin_s390_veval"];
  let Prototype = "_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmahg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmahg"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>, _Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmahq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmahq"];
  let Prototype = "signed long long int(signed long long int, signed long long int, signed long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmalhg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalhg"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmalhq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalhq"];
  let Prototype = "unsigned long long int(unsigned long long int, unsigned long long int, unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmaeg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaeg"];
  let Prototype = "signed long long int(_Vector<2, signed long long int>, _Vector<2, signed long long int>, signed long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmaleg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaleg"];
  let Prototype = "unsigned long long int(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmaog : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaog"];
  let Prototype = "signed long long int(_Vector<2, signed long long int>, _Vector<2, signed long long int>, signed long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmalog : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalog"];
  let Prototype = "unsigned long long int(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmhg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmhg"];
  let Prototype = "_Vector<2, signed long long int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmhq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmhq"];
  let Prototype = "signed long long int(signed long long int, signed long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmlhg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlhg"];
  let Prototype = "_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmlhq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlhq"];
  let Prototype = "unsigned long long int(unsigned long long int, unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmeg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmeg"];
  let Prototype = "signed long long int(_Vector<2, signed long long int>, _Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmleg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmleg"];
  let Prototype = "unsigned long long int(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmog : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmog"];
  let Prototype = "signed long long int(_Vector<2, signed long long int>, _Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vmlog : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlog"];
  let Prototype = "unsigned long long int(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vceqqs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vceqqs"];
  let Prototype = "signed long long int(unsigned long long int, unsigned long long int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vchqs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchqs"];
  let Prototype = "signed long long int(signed long long int, signed long long int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vchlqs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchlqs"];
  let Prototype = "signed long long int(unsigned long long int, unsigned long long int, int *)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vclzq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclzq"];
  let Prototype = "unsigned long long int(unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}

def Vctzq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vctzq"];
  let Prototype = "unsigned long long int(unsigned long long int)";
  let Attributes = [NoThrow, Const];
  let Features = "vector-enhancements-3";
}
