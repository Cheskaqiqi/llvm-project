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
  let Prototype = "int(void*)";
  let Attributes = [ReturnsTwice];
  let Features = ["transactional-execution"];
}

def TbeginNofloat : TargetBuiltin {
  let Spellings = ["__builtin_tbegin_nofloat"];
  let Prototype = "int(void*)";
  let Attributes = [ReturnsTwice];
  let Features = ["transactional-execution"];
}

def Tbeginc : TargetBuiltin {
  let Spellings = ["__builtin_tbeginc"];
  let Prototype = "void()";
  let Attributes = [NoThrow, ReturnsTwice];
  let Features = ["transactional-execution"];
}

def Tabort : TargetBuiltin {
  let Spellings = ["__builtin_tabort"];
  let Prototype = "void(int)";
  let Attributes = [NoReturn];
  let Features = ["transactional-execution"];
}

def Tend : TargetBuiltin {
  let Spellings = ["__builtin_tend"];
  let Prototype = "int()";
  let Attributes = [NoThrow];
  let Features = ["transactional-execution"];
}

def TxNestingDepth : TargetBuiltin {
  let Spellings = ["__builtin_tx_nesting_depth"];
  let Prototype = "int()";
  let Attributes = [NoThrow, Const];
  let Features = ["transactional-execution"];
}

def TxAssist : TargetBuiltin {
  let Spellings = ["__builtin_tx_assist"];
  let Prototype = "void(int)";
  let Attributes = [NoThrow];
  let Features = ["transactional-execution"];
}

def NonTxStore : TargetBuiltin {
  let Spellings = ["__builtin_non_tx_store"];
  let Prototype = "void(unsigned long int*, unsigned long int)";
  let Features = ["transactional-execution"];
}

def Lcbb : TargetBuiltin {
  let Spellings = ["__builtin_s390_lcbb"];
  let Prototype = "unsigned int(const void*, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vlbb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vlbb"];
  let Prototype = "Vector<16, signed char>(const void*, int)";
  let Features = ["vector"];
}

def Vll : TargetBuiltin {
  let Spellings = ["__builtin_s390_vll"];
  let Prototype = "Vector<16, signed char>(unsigned int, const void*)";
  let Features = ["vector"];
}

def Vstl : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstl"];
  let Prototype = "void(Vector<16, signed char>, unsigned int, void*)";
  let Features = ["vector"];
}

def Vperm : TargetBuiltin {
  let Spellings = ["__builtin_s390_vperm"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vpdi : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpdi"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vpksh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpksh"];
  let Prototype = "Vector<16, signed char>(Vector<8, signed short>, Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vpkshs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpkshs"];
  let Prototype = "Vector<16, signed char>(Vector<8, signed short>, Vector<8, signed short>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vpksf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpksf"];
  let Prototype = "Vector<8, signed short>(Vector<4, signed int>, Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vpksfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpksfs"];
  let Prototype = "Vector<8, signed short>(Vector<4, signed int>, Vector<4, signed int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vpksg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpksg"];
  let Prototype = "Vector<4, signed int>(Vector<2, signed long long int>, Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vpksgs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpksgs"];
  let Prototype = "Vector<4, signed int>(Vector<2, signed long long int>, Vector<2, signed long long int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vpklsh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpklsh"];
  let Prototype = "Vector<16, unsigned char>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vpklshs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpklshs"];
  let Prototype = "Vector<16, unsigned char>(Vector<8, unsigned short>, Vector<8, unsigned short>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vpklsf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpklsf"];
  let Prototype = "Vector<8, unsigned short>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vpklsfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpklsfs"];
  let Prototype = "Vector<8, unsigned short>(Vector<4, unsigned int>, Vector<4, unsigned int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vpklsg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpklsg"];
  let Prototype = "Vector<4, unsigned int>(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vpklsgs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vpklsgs"];
  let Prototype = "Vector<4, unsigned int>(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vuphb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuphb"];
  let Prototype = "Vector<8, signed short>(Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vuphh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuphh"];
  let Prototype = "Vector<4, signed int>(Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vuphf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuphf"];
  let Prototype = "Vector<2, signed long long int>(Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vuplb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplb"];
  let Prototype = "Vector<8, signed short>(Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vuplhw : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplhw"];
  let Prototype = "Vector<4, signed int>(Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vuplf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplf"];
  let Prototype = "Vector<2, signed long long int>(Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vuplhb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplhb"];
  let Prototype = "Vector<8, unsigned short>(Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vuplhh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplhh"];
  let Prototype = "Vector<4, unsigned int>(Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vuplhf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplhf"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vupllb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vupllb"];
  let Prototype = "Vector<8, unsigned short>(Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vupllh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vupllh"];
  let Prototype = "Vector<4, unsigned int>(Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vupllf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vupllf"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vaq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vaq"];
  let Prototype = "signed __int128 int(signed __int128 int, signed __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vacq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vacq"];
  let Prototype = "unsigned __int128 int(unsigned __int128 int, unsigned __int128 int, unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vaccb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vaccb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vacch : TargetBuiltin {
  let Spellings = ["__builtin_s390_vacch"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vaccf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vaccf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vaccg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vaccg"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vaccq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vaccq"];
  let Prototype = "unsigned __int128 int(unsigned __int128 int, unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vacccq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vacccq"];
  let Prototype = "unsigned __int128 int(unsigned __int128 int, unsigned __int128 int, unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vavgb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavgb"];
  let Prototype = "Vector<16, signed char>(Vector<16, signed char>, Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vavgh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavgh"];
  let Prototype = "Vector<8, signed short>(Vector<8, signed short>, Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vavgf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavgf"];
  let Prototype = "Vector<4, signed int>(Vector<4, signed int>, Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vavgg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavgg"];
  let Prototype = "Vector<2, signed long long int>(Vector<2, signed long long int>, Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vavglb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavglb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vavglh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavglh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vavglf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavglf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vavglg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavglg"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vceqbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vceqbs"];
  let Prototype = "Vector<16, signed char>(Vector<16, unsigned char>, Vector<16, unsigned char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vceqhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vceqhs"];
  let Prototype = "Vector<8, signed short>(Vector<8, unsigned short>, Vector<8, unsigned short>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vceqfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vceqfs"];
  let Prototype = "Vector<4, signed int>(Vector<4, unsigned int>, Vector<4, unsigned int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vceqgs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vceqgs"];
  let Prototype = "Vector<2, signed long long int>(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vchbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchbs"];
  let Prototype = "Vector<16, signed char>(Vector<16, signed char>, Vector<16, signed char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vchhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchhs"];
  let Prototype = "Vector<8, signed short>(Vector<8, signed short>, Vector<8, signed short>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vchfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchfs"];
  let Prototype = "Vector<4, signed int>(Vector<4, signed int>, Vector<4, signed int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vchgs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchgs"];
  let Prototype = "Vector<2, signed long long int>(Vector<2, signed long long int>, Vector<2, signed long long int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vchlbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchlbs"];
  let Prototype = "Vector<16, signed char>(Vector<16, unsigned char>, Vector<16, unsigned char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vchlhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchlhs"];
  let Prototype = "Vector<8, signed short>(Vector<8, unsigned short>, Vector<8, unsigned short>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vchlfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchlfs"];
  let Prototype = "Vector<4, signed int>(Vector<4, unsigned int>, Vector<4, unsigned int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vchlgs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchlgs"];
  let Prototype = "Vector<2, signed long long int>(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vcksm : TargetBuiltin {
  let Spellings = ["__builtin_s390_vcksm"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vclzb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclzb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vclzh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclzh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vclzf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclzf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vclzg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclzg"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vctzb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vctzb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vctzh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vctzh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vctzf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vctzf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vctzg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vctzg"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Verimb : TargetBuiltin {
  let Spellings = ["__builtin_s390_verimb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Verimh : TargetBuiltin {
  let Spellings = ["__builtin_s390_verimh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Verimf : TargetBuiltin {
  let Spellings = ["__builtin_s390_verimf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, Vector<4, unsigned int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Verimg : TargetBuiltin {
  let Spellings = ["__builtin_s390_verimg"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>, Vector<2, unsigned long long int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Verllb : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, unsigned char)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Verllh : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, unsigned char)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Verllf : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, unsigned char)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Verllg : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllg"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<2, unsigned long long int>, unsigned char)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Verllvb : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllvb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Verllvh : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllvh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Verllvf : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllvf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Verllvg : TargetBuiltin {
  let Spellings = ["__builtin_s390_verllvg"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vgfmb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmb"];
  let Prototype = "Vector<8, unsigned short>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vgfmh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmh"];
  let Prototype = "Vector<4, unsigned int>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vgfmf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmf"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vgfmg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmg"];
  let Prototype = "unsigned __int128 int(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vgfmab : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmab"];
  let Prototype = "Vector<8, unsigned short>(Vector<16, unsigned char>, Vector<16, unsigned char>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vgfmah : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmah"];
  let Prototype = "Vector<4, unsigned int>(Vector<8, unsigned short>, Vector<8, unsigned short>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vgfmaf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmaf"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<4, unsigned int>, Vector<4, unsigned int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vgfmag : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgfmag"];
  let Prototype = "unsigned __int128 int(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>, unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmahb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmahb"];
  let Prototype = "Vector<16, signed char>(Vector<16, signed char>, Vector<16, signed char>, Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmahh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmahh"];
  let Prototype = "Vector<8, signed short>(Vector<8, signed short>, Vector<8, signed short>, Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmahf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmahf"];
  let Prototype = "Vector<4, signed int>(Vector<4, signed int>, Vector<4, signed int>, Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmalhb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalhb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmalhh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalhh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmalhf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalhf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmaeb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaeb"];
  let Prototype = "Vector<8, signed short>(Vector<16, signed char>, Vector<16, signed char>, Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmaeh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaeh"];
  let Prototype = "Vector<4, signed int>(Vector<8, signed short>, Vector<8, signed short>, Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmaef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaef"];
  let Prototype = "Vector<2, signed long long int>(Vector<4, signed int>, Vector<4, signed int>, Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmaleb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaleb"];
  let Prototype = "Vector<8, unsigned short>(Vector<16, unsigned char>, Vector<16, unsigned char>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmaleh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaleh"];
  let Prototype = "Vector<4, unsigned int>(Vector<8, unsigned short>, Vector<8, unsigned short>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmalef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalef"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<4, unsigned int>, Vector<4, unsigned int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmaob : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaob"];
  let Prototype = "Vector<8, signed short>(Vector<16, signed char>, Vector<16, signed char>, Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmaoh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaoh"];
  let Prototype = "Vector<4, signed int>(Vector<8, signed short>, Vector<8, signed short>, Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmaof : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaof"];
  let Prototype = "Vector<2, signed long long int>(Vector<4, signed int>, Vector<4, signed int>, Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmalob : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalob"];
  let Prototype = "Vector<8, unsigned short>(Vector<16, unsigned char>, Vector<16, unsigned char>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmaloh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaloh"];
  let Prototype = "Vector<4, unsigned int>(Vector<8, unsigned short>, Vector<8, unsigned short>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmalof : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalof"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<4, unsigned int>, Vector<4, unsigned int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmhb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmhb"];
  let Prototype = "Vector<16, signed char>(Vector<16, signed char>, Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmhh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmhh"];
  let Prototype = "Vector<8, signed short>(Vector<8, signed short>, Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmhf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmhf"];
  let Prototype = "Vector<4, signed int>(Vector<4, signed int>, Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmlhb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlhb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmlhh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlhh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmlhf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlhf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmeb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmeb"];
  let Prototype = "Vector<8, signed short>(Vector<16, signed char>, Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmeh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmeh"];
  let Prototype = "Vector<4, signed int>(Vector<8, signed short>, Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmef"];
  let Prototype = "Vector<2, signed long long int>(Vector<4, signed int>, Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmleb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmleb"];
  let Prototype = "Vector<8, unsigned short>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmleh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmleh"];
  let Prototype = "Vector<4, unsigned int>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmlef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlef"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmob : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmob"];
  let Prototype = "Vector<8, signed short>(Vector<16, signed char>, Vector<16, signed char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmoh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmoh"];
  let Prototype = "Vector<4, signed int>(Vector<8, signed short>, Vector<8, signed short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmof : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmof"];
  let Prototype = "Vector<2, signed long long int>(Vector<4, signed int>, Vector<4, signed int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmlob : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlob"];
  let Prototype = "Vector<8, unsigned short>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmloh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmloh"];
  let Prototype = "Vector<4, unsigned int>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vmlof : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlof"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsq"];
  let Prototype = "signed __int128 int(signed __int128 int, signed __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsbcbiq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsbcbiq"];
  let Prototype = "unsigned __int128 int(unsigned __int128 int, unsigned __int128 int, unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsbiq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsbiq"];
  let Prototype = "unsigned __int128 int(unsigned __int128 int, unsigned __int128 int, unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vscbib : TargetBuiltin {
  let Spellings = ["__builtin_s390_vscbib"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vscbih : TargetBuiltin {
  let Spellings = ["__builtin_s390_vscbih"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vscbif : TargetBuiltin {
  let Spellings = ["__builtin_s390_vscbif"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vscbig : TargetBuiltin {
  let Spellings = ["__builtin_s390_vscbig"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vscbiq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vscbiq"];
  let Prototype = "unsigned __int128 int(unsigned __int128 int, unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsl : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsl"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vslb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vslb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsldb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsldb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsra : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsra"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsrab : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsrab"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsrl : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsrl"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsrlb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsrlb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsumb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsumb"];
  let Prototype = "Vector<4, unsigned int>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsumh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsumh"];
  let Prototype = "Vector<4, unsigned int>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsumgh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsumgh"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsumgf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsumgf"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsumqf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsumqf"];
  let Prototype = "unsigned __int128 int(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vsumqg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsumqg"];
  let Prototype = "unsigned __int128 int(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vtm : TargetBuiltin {
  let Spellings = ["__builtin_s390_vtm"];
  let Prototype = "int(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfaeb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaeb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfaebs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaebs"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfaeh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaeh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfaehs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaehs"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfaef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaef"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfaefs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaefs"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfaezb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaezb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfaezbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaezbs"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfaezh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaezh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfaezhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaezhs"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfaezf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaezf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfaezfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfaezfs"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeeb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeeb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeebs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeebs"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeeh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeeh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeehs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeehs"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeef"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeefs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeefs"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeezb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeezb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeezbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeezbs"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeezh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeezh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeezhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeezhs"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeezf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeezf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeezfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeezfs"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeneb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeneb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfenebs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenebs"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfeneh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfeneh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfenehs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenehs"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfenef : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenef"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfenefs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenefs"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfenezb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenezb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfenezbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenezbs"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfenezh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenezh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfenezhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenezhs"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfenezf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenezf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfenezfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfenezfs"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vistrb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vistrb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vistrbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vistrbs"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vistrh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vistrh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vistrhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vistrhs"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vistrf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vistrf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vistrfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vistrfs"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vstrcb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrcb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vstrcbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrcbs"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, Vector<16, unsigned char>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vstrch : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrch"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vstrchs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrchs"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, Vector<8, unsigned short>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vstrcf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrcf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, Vector<4, unsigned int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vstrcfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrcfs"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, Vector<4, unsigned int>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vstrczb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrczb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vstrczbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrczbs"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, Vector<16, unsigned char>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vstrczh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrczh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vstrczhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrczhs"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, Vector<8, unsigned short>, Vector<8, unsigned short>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vstrczf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrczf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, Vector<4, unsigned int>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vstrczfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrczfs"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>, Vector<4, unsigned int>, Vector<4, unsigned int>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfcedbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfcedbs"];
  let Prototype = "Vector<2, signed long long int>(Vector<2, double>, Vector<2, double>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfchdbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfchdbs"];
  let Prototype = "Vector<2, signed long long int>(Vector<2, double>, Vector<2, double>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfchedbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfchedbs"];
  let Prototype = "Vector<2, signed long long int>(Vector<2, double>, Vector<2, double>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfidb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfidb"];
  let Prototype = "Vector<2, double>(Vector<2, double>, int, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vflndb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vflndb"];
  let Prototype = "Vector<2, double>(Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vflpdb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vflpdb"];
  let Prototype = "Vector<2, double>(Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfmadb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmadb"];
  let Prototype = "Vector<2, double>(Vector<2, double>, Vector<2, double>, Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfmsdb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmsdb"];
  let Prototype = "Vector<2, double>(Vector<2, double>, Vector<2, double>, Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vfsqdb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfsqdb"];
  let Prototype = "Vector<2, double>(Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vftcidb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vftcidb"];
  let Prototype = "Vector<2, signed long long int>(Vector<2, double>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vlrlr : TargetBuiltin {
  let Spellings = ["__builtin_s390_vlrlr"];
  let Prototype = "Vector<16, signed char>(unsigned int, const void*)";
  let Features = ["vector-enhancements-1"];
}

def Vstrlr : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrlr"];
  let Prototype = "void(Vector<16, signed char>, unsigned int, void*)";
  let Features = ["vector-enhancements-1"];
}

def Vbperm : TargetBuiltin {
  let Spellings = ["__builtin_s390_vbperm"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<16, unsigned char>, Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vmslg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmslg"];
  let Prototype = "unsigned __int128 int(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>, unsigned __int128 int, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfmaxdb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmaxdb"];
  let Prototype = "Vector<2, double>(Vector<2, double>, Vector<2, double>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfmindb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmindb"];
  let Prototype = "Vector<2, double>(Vector<2, double>, Vector<2, double>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfnmadb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfnmadb"];
  let Prototype = "Vector<2, double>(Vector<2, double>, Vector<2, double>, Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfnmsdb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfnmsdb"];
  let Prototype = "Vector<2, double>(Vector<2, double>, Vector<2, double>, Vector<2, double>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfcesbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfcesbs"];
  let Prototype = "Vector<4, signed int>(Vector<4, float>, Vector<4, float>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfchsbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfchsbs"];
  let Prototype = "Vector<4, signed int>(Vector<4, float>, Vector<4, float>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfchesbs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfchesbs"];
  let Prototype = "Vector<4, signed int>(Vector<4, float>, Vector<4, float>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfisb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfisb"];
  let Prototype = "Vector<4, float>(Vector<4, float>, int, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfmaxsb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmaxsb"];
  let Prototype = "Vector<4, float>(Vector<4, float>, Vector<4, float>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfminsb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfminsb"];
  let Prototype = "Vector<4, float>(Vector<4, float>, Vector<4, float>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vflnsb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vflnsb"];
  let Prototype = "Vector<4, float>(Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vflpsb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vflpsb"];
  let Prototype = "Vector<4, float>(Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfmasb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmasb"];
  let Prototype = "Vector<4, float>(Vector<4, float>, Vector<4, float>, Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfmssb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfmssb"];
  let Prototype = "Vector<4, float>(Vector<4, float>, Vector<4, float>, Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfnmasb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfnmasb"];
  let Prototype = "Vector<4, float>(Vector<4, float>, Vector<4, float>, Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfnmssb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfnmssb"];
  let Prototype = "Vector<4, float>(Vector<4, float>, Vector<4, float>, Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vfsqsb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vfsqsb"];
  let Prototype = "Vector<4, float>(Vector<4, float>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vftcisb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vftcisb"];
  let Prototype = "Vector<4, signed int>(Vector<4, float>, int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-1"];
}

def Vsld : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsld"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-2"];
}

def Vsrd : TargetBuiltin {
  let Spellings = ["__builtin_s390_vsrd"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-2"];
}

def Vstrsb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrsb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, Vector<16, unsigned char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-2"];
}

def Vstrsh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrsh"];
  let Prototype = "Vector<16, unsigned char>(Vector<8, unsigned short>, Vector<8, unsigned short>, Vector<16, unsigned char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-2"];
}

def Vstrsf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrsf"];
  let Prototype = "Vector<16, unsigned char>(Vector<4, unsigned int>, Vector<4, unsigned int>, Vector<16, unsigned char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-2"];
}

def Vstrszb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrszb"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, Vector<16, unsigned char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-2"];
}

def Vstrszh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrszh"];
  let Prototype = "Vector<16, unsigned char>(Vector<8, unsigned short>, Vector<8, unsigned short>, Vector<16, unsigned char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-2"];
}

def Vstrszf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vstrszf"];
  let Prototype = "Vector<16, unsigned char>(Vector<4, unsigned int>, Vector<4, unsigned int>, Vector<16, unsigned char>, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-2"];
}

def Vlbrh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vlbrh"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vlbrf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vlbrf"];
  let Prototype = "Vector<4, unsigned int>(Vector<4, unsigned int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vlbrg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vlbrg"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vlbrq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vlbrq"];
  let Prototype = "unsigned __int128 int(unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector"];
}

def Vclfnhs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclfnhs"];
  let Prototype = "Vector<4, float>(Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["nnp-assist"];
}

def Vclfnls : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclfnls"];
  let Prototype = "Vector<4, float>(Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["nnp-assist"];
}

def Vcrnfs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vcrnfs"];
  let Prototype = "Vector<8, unsigned short>(Vector<4, float>, Vector<4, float>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["nnp-assist"];
}

def Vcfn : TargetBuiltin {
  let Spellings = ["__builtin_s390_vcfn"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["nnp-assist"];
}

def Vcnf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vcnf"];
  let Prototype = "Vector<8, unsigned short>(Vector<8, unsigned short>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["nnp-assist"];
}

def Bdepg : TargetBuiltin {
  let Spellings = ["__builtin_s390_bdepg"];
  let Prototype = "unsigned long int(unsigned long int, unsigned long int)";
  let Attributes = [NoThrow, Const];
  let Features = ["miscellaneous-extensions-4"];
}

def Bextg : TargetBuiltin {
  let Spellings = ["__builtin_s390_bextg"];
  let Prototype = "unsigned long int(unsigned long int, unsigned long int)";
  let Attributes = [NoThrow, Const];
  let Features = ["miscellaneous-extensions-4"];
}

def Vgemb : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgemb"];
  let Prototype = "Vector<16, unsigned char>(Vector<8, unsigned short>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vgemh : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgemh"];
  let Prototype = "Vector<8, unsigned short>(Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vgemf : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgemf"];
  let Prototype = "Vector<4, unsigned int>(Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vgemg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgemg"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vgemq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vgemq"];
  let Prototype = "unsigned __int128 int(Vector<16, unsigned char>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vuplg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplg"];
  let Prototype = "signed __int128 int(Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vupllg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vupllg"];
  let Prototype = "unsigned __int128 int(Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vuphg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuphg"];
  let Prototype = "signed __int128 int(Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vuplhg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vuplhg"];
  let Prototype = "unsigned __int128 int(Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vavgq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavgq"];
  let Prototype = "signed __int128 int(signed __int128 int, signed __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vavglq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vavglq"];
  let Prototype = "unsigned __int128 int(unsigned __int128 int, unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Veval : TargetBuiltin {
  let Spellings = ["__builtin_s390_veval"];
  let Prototype = "Vector<16, unsigned char>(Vector<16, unsigned char>, Vector<16, unsigned char>, Vector<16, unsigned char>, int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmahg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmahg"];
  let Prototype = "Vector<2, signed long long int>(Vector<2, signed long long int>, Vector<2, signed long long int>, Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmahq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmahq"];
  let Prototype = "signed __int128 int(signed __int128 int, signed __int128 int, signed __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmalhg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalhg"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmalhq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalhq"];
  let Prototype = "unsigned __int128 int(unsigned __int128 int, unsigned __int128 int, unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmaeg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaeg"];
  let Prototype = "signed __int128 int(Vector<2, signed long long int>, Vector<2, signed long long int>, signed __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmaleg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaleg"];
  let Prototype = "unsigned __int128 int(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>, unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmaog : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmaog"];
  let Prototype = "signed __int128 int(Vector<2, signed long long int>, Vector<2, signed long long int>, signed __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmalog : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmalog"];
  let Prototype = "unsigned __int128 int(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>, unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmhg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmhg"];
  let Prototype = "Vector<2, signed long long int>(Vector<2, signed long long int>, Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmhq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmhq"];
  let Prototype = "signed __int128 int(signed __int128 int, signed __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmlhg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlhg"];
  let Prototype = "Vector<2, unsigned long long int>(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmlhq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlhq"];
  let Prototype = "unsigned __int128 int(unsigned __int128 int, unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmeg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmeg"];
  let Prototype = "signed __int128 int(Vector<2, signed long long int>, Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmleg : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmleg"];
  let Prototype = "unsigned __int128 int(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmog : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmog"];
  let Prototype = "signed __int128 int(Vector<2, signed long long int>, Vector<2, signed long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vmlog : TargetBuiltin {
  let Spellings = ["__builtin_s390_vmlog"];
  let Prototype = "unsigned __int128 int(Vector<2, unsigned long long int>, Vector<2, unsigned long long int>)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vceqqs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vceqqs"];
  let Prototype = "signed __int128 int(unsigned __int128 int, unsigned __int128 int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vchqs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchqs"];
  let Prototype = "signed __int128 int(signed __int128 int, signed __int128 int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vchlqs : TargetBuiltin {
  let Spellings = ["__builtin_s390_vchlqs"];
  let Prototype = "signed __int128 int(unsigned __int128 int, unsigned __int128 int, int*)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vclzq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vclzq"];
  let Prototype = "unsigned __int128 int(unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}

def Vctzq : TargetBuiltin {
  let Spellings = ["__builtin_s390_vctzq"];
  let Prototype = "unsigned __int128 int(unsigned __int128 int)";
  let Attributes = [NoThrow, Const];
  let Features = ["vector-enhancements-3"];
}
