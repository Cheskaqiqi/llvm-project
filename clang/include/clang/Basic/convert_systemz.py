#!/usr/bin/env python3

import re
import sys
from typing import List, Tuple, Optional, Dict
from collections import OrderedDict


class SystemZConverter:
    def __init__(self):
        self.base_types = {
            'v': 'void',
            'b': 'bool',
            'c': 'char',
            's': 'short',
            'i': 'int',
            'h': '__fp16',
            'x': '_Float16',
            'y': '__bf16',
            'f': 'float',
            'd': 'double',
            'z': 'size_t',
            'w': 'wchar_t',
            'F': 'CFString',
            'G': 'id',
            'H': 'SEL',
            'M': 'struct objc_super',
            'a': '__builtin_va_list',
            'A': '__builtin_va_list&',
            'Y': 'ptrdiff_t',
            'P': 'FILE*',
            'J': 'jmp_buf',
            'p': 'pid_t',
        }
       
        self.attributes = {
            'n': 'NoThrow',
            'r': 'NoReturn',
            'U': 'Pure',
            'c': 'Const',
            't': 'CustomTypeChecking',
            'T': 'TypeGeneric',
            'F': 'LibBuiltin',
            'f': 'LibFunction',
            'h': 'RequiresHeader',
            'i': 'RuntimeLibFunction',
            'e': 'ConstWithoutErrnoAndExceptions',
            'g': 'ConstWithoutExceptions',
            'j': 'ReturnsTwice',
            'u': 'NoSideEffects',
            'z': 'CXXNamespaceStd',
            'E': 'ConstantEvaluated',
            'G': 'CXXConsteval',
        }


    def parse_builtin_line(self, line: str) -> Optional[Tuple[str, str, str, str]]:
        pattern = r'TARGET_BUILTIN\(([^,]+),\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)"\)'
        match = re.match(pattern, line.strip())
        if match:
            return match.group(1), match.group(2), match.group(3), match.group(4)
        return None


    def parse_type_encoding(self, encoding: str) -> Tuple[str, List[str]]:
        if not encoding:
            return "void", []
       
        i = 0
        return_type = self._parse_single_type(encoding, i)
        i = return_type[1]
       
        params = []
        while i < len(encoding):
            if encoding[i] == '.':
                params.append("...")
                break
            param_type = self._parse_single_type(encoding, i)
            params.append(param_type[0])
            i = param_type[1]
       
        return return_type[0], params


    def _parse_single_type(self, encoding: str, start_pos: int) -> Tuple[str, int]:
        i = start_pos
        if i >= len(encoding):
            return "void", i
    
        prefixes = []
        is_int128 = False
    
        while i < len(encoding):
            if encoding[i:i+3] == 'LLL':
                is_int128 = True
                i += 3
                continue
            elif encoding[i:i+2] == 'LL':
                prefixes.append('long long')
                i += 2
            elif encoding[i] == 'L':
                prefixes.append('long')
                i += 1
            elif encoding[i] == 'U':
                prefixes.append('unsigned')
                i += 1
            elif encoding[i] == 'S':
                prefixes.append('signed')
                i += 1
            elif encoding[i] in 'ZWNOI':
                i += 1
            else:
                break
    
        if i >= len(encoding):
            return "void", i
    
        if is_int128:
            base_type = "__int128_t"
            prefixes = [p for p in prefixes if p != 'unsigned']

            if i < len(encoding):
                i += 1
    
            cv = []
            ptrs = []
            while i < len(encoding):
                ch = encoding[i]
                if ch == '*':
                    ptrs.append('*')
                    i += 1
                    if i < len(encoding) and encoding[i].isdigit():
                        i += 1
                elif ch == '&':
                    ptrs.append('&')
                    i += 1
                    if i < len(encoding) and encoding[i].isdigit():
                        i += 1
                elif ch == 'C':
                    cv.append('const')
                    i += 1
                elif ch == 'D':
                    cv.append('volatile')
                    i += 1
                elif ch == 'R':
                    cv.append('restrict')
                    i += 1
                else:
                    break
    
            cv_str = (" " + " ".join(cv)) if cv else ""
            ptr_str = "".join((" *" if p == "*" else " &") for p in ptrs)
            return f"{base_type}{cv_str}{ptr_str}".strip(), i
    
        base_type = ""
        if encoding[i] == 'V':
            i += 1
            num_str = ""
            while i < len(encoding) and encoding[i].isdigit():
                num_str += encoding[i]
                i += 1
            elem_type, i2 = self._parse_single_type(encoding, i)
            base_type = f"_Vector<{num_str}, {elem_type}>"
            i = i2
        elif encoding[i] == 'q':
            i += 1
            num_str = ""
            while i < len(encoding) and encoding[i].isdigit():
                num_str += encoding[i]
                i += 1
            elem_type, i2 = self._parse_single_type(encoding, i)
            base_type = f"ScalableVector<{num_str}, {elem_type}>"
            i = i2
        elif encoding[i] == 'E':
            i += 1
            num_str = ""
            while i < len(encoding) and encoding[i].isdigit():
                num_str += encoding[i]
                i += 1
            elem_type, i2 = self._parse_single_type(encoding, i)
            base_type = f"_Vector<{num_str}, {elem_type}>"
            i = i2
        elif encoding[i] == 'X':
            i += 1
            elem_type, i2 = self._parse_single_type(encoding, i)
            base_type = f"_Complex {elem_type}"
            i = i2
        elif encoding[i] == 'Q':
            i += 1
            if i < len(encoding):
                if encoding[i] == 'a':
                    base_type = "svcount_t"
                elif encoding[i] == 'b':
                    base_type = "__amdgpu_buffer_rsrc_t"
                else:
                    base_type = f"TargetBuiltinType_{encoding[i]}"
                i += 1
            else:
                base_type = "void"
        else:
            base_type = self.base_types.get(encoding[i], f"UnknownType_{encoding[i]}")
            i += 1
    
        cv = []
        ptrs = []
        while i < len(encoding):
            ch = encoding[i]
            if ch == '*':
                ptrs.append('*')
                i += 1
                if i < len(encoding) and encoding[i].isdigit():
                    i += 1
            elif ch == '&':
                ptrs.append('&')
                i += 1
                if i < len(encoding) and encoding[i].isdigit():
                    i += 1
            elif ch == 'C':
                cv.append('const')
                i += 1
            elif ch == 'D':
                cv.append('volatile')
                i += 1
            elif ch == 'R':
                cv.append('restrict')
                i += 1
            else:
                break
    
        prefix_str = (" ".join(prefixes) + " ") if prefixes else ""
        cv_str = (" " + " ".join(cv)) if cv else ""
        ptr_str = "".join((" *" if p == "*" else " &") for p in ptrs)
        return f"{prefix_str}{base_type}{cv_str}{ptr_str}".strip(), i


    def decode_attributes(self, attr_str: str) -> List[str]:
        attrs = []
        i = 0
        while i < len(attr_str):
            char = attr_str[i]
            if char in self.attributes:
                attrs.append(self.attributes[char])
            elif char == 'p' and i + 1 < len(attr_str) and attr_str[i + 1] == ':':
                j = i + 2
                while j < len(attr_str) and attr_str[j] != ':':
                    j += 1
                if j < len(attr_str):
                    num = attr_str[i + 2:j]
                    attrs.append(f'PrintfFormat<{num}>')
                    i = j
            elif char == 'V' and i + 1 < len(attr_str) and attr_str[i + 1] == ':':
                j = i + 2
                while j < len(attr_str) and attr_str[j] != ':':
                    j += 1
                if j < len(attr_str):
                    num = attr_str[i + 2:j]
                    attrs.append(f'RequiresVectorWidth<{num}>')
                    i = j
            i += 1
        return attrs


    def generate_compact_def(self, name: str, return_type: str, param_types: List[str],
                             attributes: List[str]) -> str:
        """Generate compact TableGen definition using base classes"""
        
        # Determine which base class to use and the def name
        if '__builtin_s390_' in name:
            # Remove the __builtin_s390_ prefix for the def name
            def_name = name.replace('__builtin_s390_', '')
            base_class = 'SystemZBuiltin'
        else:
            # Keep the full name for non-s390 builtins
            def_name = name
            base_class = 'SystemZNoPrefixBuiltin'
        
        # Build prototype
        if not param_types:
            prototype = f"{return_type}()"
        elif param_types == ["..."]:
            prototype = f"{return_type}(...)"
        else:
            prototype = f"{return_type}({', '.join(param_types)})"
        
        # Build attributes list
        if attributes:
            attr_str = f", [{', '.join(attributes)}]"
        else:
            attr_str = ""
        
        # Generate compact definition
        return f'def {def_name} : {base_class}<"{prototype}"{attr_str}>;'


    def group_builtins_by_feature(self, builtins_data: List[Tuple]) -> Dict[str, List]:
        """Group builtins by their feature attribute"""
        groups = OrderedDict()
        
        for name, proto_encoding, attr_encoding, feature in builtins_data:
            feature = feature.strip()
            if feature not in groups:
                groups[feature] = []
            groups[feature].append((name, proto_encoding, attr_encoding))
        
        return groups


    def convert_file(self, input_file: str, output_file: str = None):
        try:
            with open(input_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: File not found {input_file}")
            return
        
        # Parse all builtins first
        builtins_data = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith('TARGET_BUILTIN'):
                parsed = self.parse_builtin_line(line)
                if parsed:
                    builtins_data.append(parsed)
        
        # Group by feature
        feature_groups = self.group_builtins_by_feature(builtins_data)
        
        # Generate output
        converted_lines = []
        
        # Header
        converted_lines.append("//===--- BuiltinsSystemZ.td - SystemZ Builtin function database -*- C++ -*-===//")
        converted_lines.append("//")
        converted_lines.append("// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.")
        converted_lines.append("// See https://llvm.org/LICENSE.txt for license information.")
        converted_lines.append("// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception")
        converted_lines.append("//")
        converted_lines.append("//===----------------------------------------------------------------------===//")
        converted_lines.append("")
        converted_lines.append('include "clang/Basic/BuiltinsBase.td"')
        converted_lines.append("")
        
        # Add base class definitions
        converted_lines.append('def SystemZPrefix : NamePrefix<"__builtin_s390_">;')
        converted_lines.append("")
        converted_lines.append("class SystemZBuiltin<string prototype, list<Attribute> Attr = []> : TargetBuiltin {")
        converted_lines.append("  let Spellings = [NAME];")
        converted_lines.append("  let Prototype = prototype;")
        converted_lines.append("  let Attributes = Attr;")
        converted_lines.append("  let RequiredNamePrefix = SystemZPrefix;")
        converted_lines.append("}")
        converted_lines.append("")
        converted_lines.append("class SystemZNoPrefixBuiltin<string prototype, list<Attribute> Attr = []> : TargetBuiltin {")
        converted_lines.append("  let Spellings = [NAME];")
        converted_lines.append("  let Prototype = prototype;")
        converted_lines.append("  let Attributes = Attr;")
        converted_lines.append("}")
        converted_lines.append("")
        
        # Generate grouped definitions
        conversion_count = 0
        error_count = 0
        
        for feature, builtins in feature_groups.items():
            # Start feature group
            converted_lines.append(f'let Features = "{feature}" in {{')
            
            for name, proto_encoding, attr_encoding in builtins:
                try:
                    return_type, param_types = self.parse_type_encoding(proto_encoding)
                    attributes = self.decode_attributes(attr_encoding)
                    
                    compact_def = self.generate_compact_def(name, return_type, param_types, attributes)
                    converted_lines.append(f"  {compact_def}")
                    conversion_count += 1
                    
                except Exception as e:
                    error_line = f"  // ERROR converting {name}: {e}"
                    converted_lines.append(error_line)
                    error_count += 1
            
            # End feature group
            converted_lines.append("}")
            converted_lines.append("")
        
        # Write output
        output_content = '\n'.join(converted_lines)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(output_content)
            print(f"Conversion completed!")
            print(f"Output file: {output_file}")
            print(f"Successfully converted: {conversion_count} functions")
            print(f"Feature groups: {len(feature_groups)}")
            if error_count > 0:
                print(f"Conversion errors: {error_count}")
        else:
            print(output_content)


    def test_conversion(self):
        test_cases = [
            ('TARGET_BUILTIN(__builtin_tbegin, "iv*", "j", "transactional-execution")',
             "Transaction builtin (no s390 prefix)"),
            ('TARGET_BUILTIN(__builtin_s390_lcbb, "UivC*Ii", "nc", "vector")',
             "Vector builtin (with s390 prefix)"),
            ('TARGET_BUILTIN(__builtin_s390_vperm, "V16UcV16UcV16UcV16Uc", "nc", "vector")',
             "Vector types"),
        ]
       
        print("=== Testing Compact Conversion ===\n")
        for test_case, description in test_cases:
            print(f"Test: {description}")
            print(f"Input: {test_case}")
           
            parsed = self.parse_builtin_line(test_case)
            if parsed:
                name, proto, attrs, feature = parsed
                print(f"Parsed:")
                print(f"  Name: {name}")
                print(f"  Feature: {feature}")
               
                return_type, param_types = self.parse_type_encoding(proto)
                attributes = self.decode_attributes(attrs)
                
                compact_def = self.generate_compact_def(name, return_type, param_types, attributes)
                print(f"Result:")
                print(f"  {compact_def}")
            else:
                print("Parse failed!")
            print("-" * 60)


def main():
    converter = SystemZConverter()
   
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            converter.test_conversion()
        else:
            input_file = sys.argv[1]
            output_file = sys.argv[2] if len(sys.argv) > 2 else 'BuiltinsSystemZ.td'
            converter.convert_file(input_file, output_file)
    else:
        print("SystemZ Builtin Function Converter (Compact Version)")
        print("Usage:")
        print("  python convert_systemz.py --test                    # Run tests")
        print("  python convert_systemz.py input.def [output.td]     # Convert file")
        print("")
        print("Example:")
        print("  python convert_systemz.py BuiltinsSystemZ.def BuiltinsSystemZ.td")


if __name__ == "__main__":
    main()