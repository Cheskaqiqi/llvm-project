; RUN: llvm-as -disable-output <%s 2>&1 | FileCheck %s
; CHECK: invalid #dbg record expression
; CHECK-NEXT: #dbg_value({{.*}})
; CHECK: warning: ignoring invalid debug info

define void @foo(i32 %a) {
entry:
  %s = alloca i32
  call void @llvm.dbg.value(metadata ptr %s, i64 0, metadata !DILocalVariable(scope: !1), metadata !""), !dbg !DILocation(scope: !1)
  ret void
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DISubprogram()
