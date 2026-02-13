//===--- BuiltinsSystemZ.td - SystemZ Builtin function database -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

def SystemZPrefix : NamePrefix<"__builtin_s390_">;

class SystemZBuiltin<string prototype, list<Attribute> Attr = []> : TargetBuiltin {
  let Spellings = [NAME];
  let Prototype = prototype;
  let Attributes = Attr;
  let RequiredNamePrefix = SystemZPrefix;
}

class SystemZNoPrefixBuiltin<string prototype, list<Attribute> Attr = []> : TargetBuiltin {
  let Spellings = [NAME];
  let Prototype = prototype;
  let Attributes = Attr;
}

let Features = "transactional-execution" in {
  def __builtin_tbegin : SystemZNoPrefixBuiltin<"int(void *)", [ReturnsTwice]>;
  def __builtin_tbegin_nofloat : SystemZNoPrefixBuiltin<"int(void *)", [ReturnsTwice]>;
  def __builtin_tbeginc : SystemZNoPrefixBuiltin<"void()", [NoThrow, ReturnsTwice]>;
  def __builtin_tabort : SystemZNoPrefixBuiltin<"void(int)", [NoReturn]>;
  def __builtin_tend : SystemZNoPrefixBuiltin<"int()", [NoThrow]>;
  def __builtin_tx_nesting_depth : SystemZNoPrefixBuiltin<"int()", [NoThrow, Const]>;
  def __builtin_tx_assist : SystemZNoPrefixBuiltin<"void(int)", [NoThrow]>;
  def __builtin_non_tx_store : SystemZNoPrefixBuiltin<"void(unsigned long int *, unsigned long int)">;
}

let Features = "vector" in {
  def lcbb : SystemZBuiltin<"unsigned int(void const *, int)", [NoThrow, Const]>;
  def vlbb : SystemZBuiltin<"_Vector<16, signed char>(void const *, int)">;
  def vll : SystemZBuiltin<"_Vector<16, signed char>(unsigned int, void const *)">;
  def vstl : SystemZBuiltin<"void(_Vector<16, signed char>, unsigned int, void *)">;
  def vperm : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vpdi : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int)", [NoThrow, Const]>;
  def vpksh : SystemZBuiltin<"_Vector<16, signed char>(_Vector<8, signed short>, _Vector<8, signed short>)", [NoThrow, Const]>;
  def vpkshs : SystemZBuiltin<"_Vector<16, signed char>(_Vector<8, signed short>, _Vector<8, signed short>, int *)", [NoThrow, Const]>;
  def vpksf : SystemZBuiltin<"_Vector<8, signed short>(_Vector<4, signed int>, _Vector<4, signed int>)", [NoThrow, Const]>;
  def vpksfs : SystemZBuiltin<"_Vector<8, signed short>(_Vector<4, signed int>, _Vector<4, signed int>, int *)", [NoThrow, Const]>;
  def vpksg : SystemZBuiltin<"_Vector<4, signed int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>)", [NoThrow, Const]>;
  def vpksgs : SystemZBuiltin<"_Vector<4, signed int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>, int *)", [NoThrow, Const]>;
  def vpklsh : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vpklshs : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)", [NoThrow, Const]>;
  def vpklsf : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vpklsfs : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)", [NoThrow, Const]>;
  def vpklsg : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vpklsgs : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int *)", [NoThrow, Const]>;
  def vuphb : SystemZBuiltin<"_Vector<8, signed short>(_Vector<16, signed char>)", [NoThrow, Const]>;
  def vuphh : SystemZBuiltin<"_Vector<4, signed int>(_Vector<8, signed short>)", [NoThrow, Const]>;
  def vuphf : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<4, signed int>)", [NoThrow, Const]>;
  def vuplb : SystemZBuiltin<"_Vector<8, signed short>(_Vector<16, signed char>)", [NoThrow, Const]>;
  def vuplhw : SystemZBuiltin<"_Vector<4, signed int>(_Vector<8, signed short>)", [NoThrow, Const]>;
  def vuplf : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<4, signed int>)", [NoThrow, Const]>;
  def vuplhb : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vuplhh : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vuplhf : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vupllb : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vupllh : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vupllf : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vaq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t)", [NoThrow, Const]>;
  def vacq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t, __int128_t)", [NoThrow, Const]>;
  def vaccb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vacch : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vaccf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vaccg : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vaccq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t)", [NoThrow, Const]>;
  def vacccq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t, __int128_t)", [NoThrow, Const]>;
  def vavgb : SystemZBuiltin<"_Vector<16, signed char>(_Vector<16, signed char>, _Vector<16, signed char>)", [NoThrow, Const]>;
  def vavgh : SystemZBuiltin<"_Vector<8, signed short>(_Vector<8, signed short>, _Vector<8, signed short>)", [NoThrow, Const]>;
  def vavgf : SystemZBuiltin<"_Vector<4, signed int>(_Vector<4, signed int>, _Vector<4, signed int>)", [NoThrow, Const]>;
  def vavgg : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>)", [NoThrow, Const]>;
  def vavglb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vavglh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vavglf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vavglg : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vceqbs : SystemZBuiltin<"_Vector<16, signed char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)", [NoThrow, Const]>;
  def vceqhs : SystemZBuiltin<"_Vector<8, signed short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)", [NoThrow, Const]>;
  def vceqfs : SystemZBuiltin<"_Vector<4, signed int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)", [NoThrow, Const]>;
  def vceqgs : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int *)", [NoThrow, Const]>;
  def vchbs : SystemZBuiltin<"_Vector<16, signed char>(_Vector<16, signed char>, _Vector<16, signed char>, int *)", [NoThrow, Const]>;
  def vchhs : SystemZBuiltin<"_Vector<8, signed short>(_Vector<8, signed short>, _Vector<8, signed short>, int *)", [NoThrow, Const]>;
  def vchfs : SystemZBuiltin<"_Vector<4, signed int>(_Vector<4, signed int>, _Vector<4, signed int>, int *)", [NoThrow, Const]>;
  def vchgs : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>, int *)", [NoThrow, Const]>;
  def vchlbs : SystemZBuiltin<"_Vector<16, signed char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)", [NoThrow, Const]>;
  def vchlhs : SystemZBuiltin<"_Vector<8, signed short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)", [NoThrow, Const]>;
  def vchlfs : SystemZBuiltin<"_Vector<4, signed int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)", [NoThrow, Const]>;
  def vchlgs : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int *)", [NoThrow, Const]>;
  def vcksm : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vclzb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vclzh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vclzf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vclzg : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vctzb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vctzh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vctzf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vctzg : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def verimb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int)", [NoThrow, Const]>;
  def verimh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int)", [NoThrow, Const]>;
  def verimf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int)", [NoThrow, Const]>;
  def verimg : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int)", [NoThrow, Const]>;
  def verllb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, unsigned char)", [NoThrow, Const]>;
  def verllh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, unsigned char)", [NoThrow, Const]>;
  def verllf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, unsigned char)", [NoThrow, Const]>;
  def verllg : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, unsigned char)", [NoThrow, Const]>;
  def verllvb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def verllvh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def verllvf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def verllvg : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vgfmb : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vgfmh : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vgfmf : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vgfmg : SystemZBuiltin<"__int128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vgfmab : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vgfmah : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vgfmaf : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vgfmag : SystemZBuiltin<"__int128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, __int128_t)", [NoThrow, Const]>;
  def vmahb : SystemZBuiltin<"_Vector<16, signed char>(_Vector<16, signed char>, _Vector<16, signed char>, _Vector<16, signed char>)", [NoThrow, Const]>;
  def vmahh : SystemZBuiltin<"_Vector<8, signed short>(_Vector<8, signed short>, _Vector<8, signed short>, _Vector<8, signed short>)", [NoThrow, Const]>;
  def vmahf : SystemZBuiltin<"_Vector<4, signed int>(_Vector<4, signed int>, _Vector<4, signed int>, _Vector<4, signed int>)", [NoThrow, Const]>;
  def vmalhb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vmalhh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vmalhf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vmaeb : SystemZBuiltin<"_Vector<8, signed short>(_Vector<16, signed char>, _Vector<16, signed char>, _Vector<8, signed short>)", [NoThrow, Const]>;
  def vmaeh : SystemZBuiltin<"_Vector<4, signed int>(_Vector<8, signed short>, _Vector<8, signed short>, _Vector<4, signed int>)", [NoThrow, Const]>;
  def vmaef : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<4, signed int>, _Vector<4, signed int>, _Vector<2, signed long long int>)", [NoThrow, Const]>;
  def vmaleb : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vmaleh : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vmalef : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vmaob : SystemZBuiltin<"_Vector<8, signed short>(_Vector<16, signed char>, _Vector<16, signed char>, _Vector<8, signed short>)", [NoThrow, Const]>;
  def vmaoh : SystemZBuiltin<"_Vector<4, signed int>(_Vector<8, signed short>, _Vector<8, signed short>, _Vector<4, signed int>)", [NoThrow, Const]>;
  def vmaof : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<4, signed int>, _Vector<4, signed int>, _Vector<2, signed long long int>)", [NoThrow, Const]>;
  def vmalob : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vmaloh : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vmalof : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vmhb : SystemZBuiltin<"_Vector<16, signed char>(_Vector<16, signed char>, _Vector<16, signed char>)", [NoThrow, Const]>;
  def vmhh : SystemZBuiltin<"_Vector<8, signed short>(_Vector<8, signed short>, _Vector<8, signed short>)", [NoThrow, Const]>;
  def vmhf : SystemZBuiltin<"_Vector<4, signed int>(_Vector<4, signed int>, _Vector<4, signed int>)", [NoThrow, Const]>;
  def vmlhb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vmlhh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vmlhf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vmeb : SystemZBuiltin<"_Vector<8, signed short>(_Vector<16, signed char>, _Vector<16, signed char>)", [NoThrow, Const]>;
  def vmeh : SystemZBuiltin<"_Vector<4, signed int>(_Vector<8, signed short>, _Vector<8, signed short>)", [NoThrow, Const]>;
  def vmef : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<4, signed int>, _Vector<4, signed int>)", [NoThrow, Const]>;
  def vmleb : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vmleh : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vmlef : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vmob : SystemZBuiltin<"_Vector<8, signed short>(_Vector<16, signed char>, _Vector<16, signed char>)", [NoThrow, Const]>;
  def vmoh : SystemZBuiltin<"_Vector<4, signed int>(_Vector<8, signed short>, _Vector<8, signed short>)", [NoThrow, Const]>;
  def vmof : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<4, signed int>, _Vector<4, signed int>)", [NoThrow, Const]>;
  def vmlob : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vmloh : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vmlof : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vsq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t)", [NoThrow, Const]>;
  def vsbcbiq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t, __int128_t)", [NoThrow, Const]>;
  def vsbiq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t, __int128_t)", [NoThrow, Const]>;
  def vscbib : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vscbih : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vscbif : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vscbig : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vscbiq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t)", [NoThrow, Const]>;
  def vsl : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vslb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vsldb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)", [NoThrow, Const]>;
  def vsra : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vsrab : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vsrl : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vsrlb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vsumb : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vsumh : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vsumgh : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vsumgf : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vsumqf : SystemZBuiltin<"__int128_t(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vsumqg : SystemZBuiltin<"__int128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vtm : SystemZBuiltin<"int(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vfaeb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)", [NoThrow, Const]>;
  def vfaebs : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int, int *)", [NoThrow, Const]>;
  def vfaeh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int)", [NoThrow, Const]>;
  def vfaehs : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int, int *)", [NoThrow, Const]>;
  def vfaef : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int)", [NoThrow, Const]>;
  def vfaefs : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int, int *)", [NoThrow, Const]>;
  def vfaezb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)", [NoThrow, Const]>;
  def vfaezbs : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int, int *)", [NoThrow, Const]>;
  def vfaezh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int)", [NoThrow, Const]>;
  def vfaezhs : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int, int *)", [NoThrow, Const]>;
  def vfaezf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int)", [NoThrow, Const]>;
  def vfaezfs : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int, int *)", [NoThrow, Const]>;
  def vfeeb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vfeebs : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)", [NoThrow, Const]>;
  def vfeeh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vfeehs : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)", [NoThrow, Const]>;
  def vfeef : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vfeefs : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)", [NoThrow, Const]>;
  def vfeezb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vfeezbs : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)", [NoThrow, Const]>;
  def vfeezh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vfeezhs : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)", [NoThrow, Const]>;
  def vfeezf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vfeezfs : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)", [NoThrow, Const]>;
  def vfeneb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vfenebs : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)", [NoThrow, Const]>;
  def vfeneh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vfenehs : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)", [NoThrow, Const]>;
  def vfenef : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vfenefs : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)", [NoThrow, Const]>;
  def vfenezb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vfenezbs : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)", [NoThrow, Const]>;
  def vfenezh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vfenezhs : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)", [NoThrow, Const]>;
  def vfenezf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vfenezfs : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)", [NoThrow, Const]>;
  def vistrb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vistrbs : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, int *)", [NoThrow, Const]>;
  def vistrh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vistrhs : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, int *)", [NoThrow, Const]>;
  def vistrf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vistrfs : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, int *)", [NoThrow, Const]>;
  def vstrcb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int)", [NoThrow, Const]>;
  def vstrcbs : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int, int *)", [NoThrow, Const]>;
  def vstrch : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int)", [NoThrow, Const]>;
  def vstrchs : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int, int *)", [NoThrow, Const]>;
  def vstrcf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int)", [NoThrow, Const]>;
  def vstrcfs : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int, int *)", [NoThrow, Const]>;
  def vstrczb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int)", [NoThrow, Const]>;
  def vstrczbs : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int, int *)", [NoThrow, Const]>;
  def vstrczh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int)", [NoThrow, Const]>;
  def vstrczhs : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int, int *)", [NoThrow, Const]>;
  def vstrczf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int)", [NoThrow, Const]>;
  def vstrczfs : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int, int *)", [NoThrow, Const]>;
  def vfcedbs : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<2, double>, _Vector<2, double>, int *)", [NoThrow, Const]>;
  def vfchdbs : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<2, double>, _Vector<2, double>, int *)", [NoThrow, Const]>;
  def vfchedbs : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<2, double>, _Vector<2, double>, int *)", [NoThrow, Const]>;
  def vfidb : SystemZBuiltin<"_Vector<2, double>(_Vector<2, double>, int, int)", [NoThrow, Const]>;
  def vflndb : SystemZBuiltin<"_Vector<2, double>(_Vector<2, double>)", [NoThrow, Const]>;
  def vflpdb : SystemZBuiltin<"_Vector<2, double>(_Vector<2, double>)", [NoThrow, Const]>;
  def vfmadb : SystemZBuiltin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)", [NoThrow, Const]>;
  def vfmsdb : SystemZBuiltin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)", [NoThrow, Const]>;
  def vfsqdb : SystemZBuiltin<"_Vector<2, double>(_Vector<2, double>)", [NoThrow, Const]>;
  def vftcidb : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<2, double>, int, int *)", [NoThrow, Const]>;
  def vlbrh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vlbrf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>)", [NoThrow, Const]>;
  def vlbrg : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vlbrq : SystemZBuiltin<"__int128_t(__int128_t)", [NoThrow, Const]>;
}

let Features = "vector-enhancements-1" in {
  def vlrlr : SystemZBuiltin<"_Vector<16, signed char>(unsigned int, void const *)">;
  def vstrlr : SystemZBuiltin<"void(_Vector<16, signed char>, unsigned int, void *)">;
  def vbperm : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vmslg : SystemZBuiltin<"__int128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, __int128_t, int)", [NoThrow, Const]>;
  def vfmaxdb : SystemZBuiltin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, int)", [NoThrow, Const]>;
  def vfmindb : SystemZBuiltin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, int)", [NoThrow, Const]>;
  def vfnmadb : SystemZBuiltin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)", [NoThrow, Const]>;
  def vfnmsdb : SystemZBuiltin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)", [NoThrow, Const]>;
  def vfcesbs : SystemZBuiltin<"_Vector<4, signed int>(_Vector<4, float>, _Vector<4, float>, int *)", [NoThrow, Const]>;
  def vfchsbs : SystemZBuiltin<"_Vector<4, signed int>(_Vector<4, float>, _Vector<4, float>, int *)", [NoThrow, Const]>;
  def vfchesbs : SystemZBuiltin<"_Vector<4, signed int>(_Vector<4, float>, _Vector<4, float>, int *)", [NoThrow, Const]>;
  def vfisb : SystemZBuiltin<"_Vector<4, float>(_Vector<4, float>, int, int)", [NoThrow, Const]>;
  def vfmaxsb : SystemZBuiltin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, int)", [NoThrow, Const]>;
  def vfminsb : SystemZBuiltin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, int)", [NoThrow, Const]>;
  def vflnsb : SystemZBuiltin<"_Vector<4, float>(_Vector<4, float>)", [NoThrow, Const]>;
  def vflpsb : SystemZBuiltin<"_Vector<4, float>(_Vector<4, float>)", [NoThrow, Const]>;
  def vfmasb : SystemZBuiltin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)", [NoThrow, Const]>;
  def vfmssb : SystemZBuiltin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)", [NoThrow, Const]>;
  def vfnmasb : SystemZBuiltin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)", [NoThrow, Const]>;
  def vfnmssb : SystemZBuiltin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)", [NoThrow, Const]>;
  def vfsqsb : SystemZBuiltin<"_Vector<4, float>(_Vector<4, float>)", [NoThrow, Const]>;
  def vftcisb : SystemZBuiltin<"_Vector<4, signed int>(_Vector<4, float>, int, int *)", [NoThrow, Const]>;
}

let Features = "vector-enhancements-2" in {
  def vsld : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)", [NoThrow, Const]>;
  def vsrd : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)", [NoThrow, Const]>;
  def vstrsb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)", [NoThrow, Const]>;
  def vstrsh : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<16, unsigned char>, int *)", [NoThrow, Const]>;
  def vstrsf : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<16, unsigned char>, int *)", [NoThrow, Const]>;
  def vstrszb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)", [NoThrow, Const]>;
  def vstrszh : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<16, unsigned char>, int *)", [NoThrow, Const]>;
  def vstrszf : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<16, unsigned char>, int *)", [NoThrow, Const]>;
}

let Features = "nnp-assist" in {
  def vclfnhs : SystemZBuiltin<"_Vector<4, float>(_Vector<8, unsigned short>, int)", [NoThrow, Const]>;
  def vclfnls : SystemZBuiltin<"_Vector<4, float>(_Vector<8, unsigned short>, int)", [NoThrow, Const]>;
  def vcrnfs : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<4, float>, _Vector<4, float>, int)", [NoThrow, Const]>;
  def vcfn : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, int)", [NoThrow, Const]>;
  def vcnf : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, int)", [NoThrow, Const]>;
}

let Features = "miscellaneous-extensions-4" in {
  def bdepg : SystemZBuiltin<"unsigned long int(unsigned long int, unsigned long int)", [NoThrow, Const]>;
  def bextg : SystemZBuiltin<"unsigned long int(unsigned long int, unsigned long int)", [NoThrow, Const]>;
}

let Features = "vector-enhancements-3" in {
  def vgemb : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<8, unsigned short>)", [NoThrow, Const]>;
  def vgemh : SystemZBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vgemf : SystemZBuiltin<"_Vector<4, unsigned int>(_Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vgemg : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vgemq : SystemZBuiltin<"__int128_t(_Vector<16, unsigned char>)", [NoThrow, Const]>;
  def vuplg : SystemZBuiltin<"__int128_t(_Vector<2, signed long long int>)", [NoThrow, Const]>;
  def vupllg : SystemZBuiltin<"__int128_t(_Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vuphg : SystemZBuiltin<"__int128_t(_Vector<2, signed long long int>)", [NoThrow, Const]>;
  def vuplhg : SystemZBuiltin<"__int128_t(_Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vavgq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t)", [NoThrow, Const]>;
  def vavglq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t)", [NoThrow, Const]>;
  def veval : SystemZBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int)", [NoThrow, Const]>;
  def vmahg : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>, _Vector<2, signed long long int>)", [NoThrow, Const]>;
  def vmahq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t, __int128_t)", [NoThrow, Const]>;
  def vmalhg : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vmalhq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t, __int128_t)", [NoThrow, Const]>;
  def vmaeg : SystemZBuiltin<"__int128_t(_Vector<2, signed long long int>, _Vector<2, signed long long int>, __int128_t)", [NoThrow, Const]>;
  def vmaleg : SystemZBuiltin<"__int128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, __int128_t)", [NoThrow, Const]>;
  def vmaog : SystemZBuiltin<"__int128_t(_Vector<2, signed long long int>, _Vector<2, signed long long int>, __int128_t)", [NoThrow, Const]>;
  def vmalog : SystemZBuiltin<"__int128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, __int128_t)", [NoThrow, Const]>;
  def vmhg : SystemZBuiltin<"_Vector<2, signed long long int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>)", [NoThrow, Const]>;
  def vmhq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t)", [NoThrow, Const]>;
  def vmlhg : SystemZBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vmlhq : SystemZBuiltin<"__int128_t(__int128_t, __int128_t)", [NoThrow, Const]>;
  def vmeg : SystemZBuiltin<"__int128_t(_Vector<2, signed long long int>, _Vector<2, signed long long int>)", [NoThrow, Const]>;
  def vmleg : SystemZBuiltin<"__int128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vmog : SystemZBuiltin<"__int128_t(_Vector<2, signed long long int>, _Vector<2, signed long long int>)", [NoThrow, Const]>;
  def vmlog : SystemZBuiltin<"__int128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)", [NoThrow, Const]>;
  def vceqqs : SystemZBuiltin<"__int128_t(__int128_t, __int128_t, int *)", [NoThrow, Const]>;
  def vchqs : SystemZBuiltin<"__int128_t(__int128_t, __int128_t, int *)", [NoThrow, Const]>;
  def vchlqs : SystemZBuiltin<"__int128_t(__int128_t, __int128_t, int *)", [NoThrow, Const]>;
  def vclzq : SystemZBuiltin<"__int128_t(__int128_t)", [NoThrow, Const]>;
  def vctzq : SystemZBuiltin<"__int128_t(__int128_t)", [NoThrow, Const]>;
}
