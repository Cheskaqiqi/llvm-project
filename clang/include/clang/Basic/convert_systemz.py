#!/usr/bin/env python3

import re
import sys
from typing import List, Tuple, Optional

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
        while i < len(encoding):
            if encoding[i:i+3] == 'LLL':
                prefixes.append('__int128')
                i += 3
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
        
        base_type = ""
        if encoding[i] == 'V':
            i += 1
            num_str = ""
            while i < len(encoding) and encoding[i].isdigit():
                num_str += encoding[i]
                i += 1
            elem_type = self._parse_single_type(encoding, i)
            base_type = f"Vector<{num_str}, {elem_type[0]}>"
            i = elem_type[1]
        elif encoding[i] == 'q':
            i += 1
            num_str = ""
            while i < len(encoding) and encoding[i].isdigit():
                num_str += encoding[i]
                i += 1
            elem_type = self._parse_single_type(encoding, i)
            base_type = f"ScalableVector<{num_str}, {elem_type[0]}>"
            i = elem_type[1]
        elif encoding[i] == 'E':
            i += 1
            num_str = ""
            while i < len(encoding) and encoding[i].isdigit():
                num_str += encoding[i]
                i += 1
            elem_type = self._parse_single_type(encoding, i)
            base_type = f"ExtVector<{num_str}, {elem_type[0]}>"
            i = elem_type[1]
        elif encoding[i] == 'X':
            i += 1
            elem_type = self._parse_single_type(encoding, i)
            base_type = f"_Complex {elem_type[0]}"
            i = elem_type[1]
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
            base_type = self.base_types.get(encoding[i], f"UnknownType_{encoding[i]}")
            i += 1
        
        suffixes = []
        while i < len(encoding):
            if encoding[i] == '*':
                suffixes.append('*')
                i += 1
                if i < len(encoding) and encoding[i].isdigit():
                    i += 1
            elif encoding[i] == '&':
                suffixes.append('&')
                i += 1
                if i < len(encoding) and encoding[i].isdigit():
                    i += 1
            elif encoding[i] == 'C':
                suffixes.insert(0, 'const ')
                i += 1
            elif encoding[i] == 'D':
                suffixes.insert(0, 'volatile ')
                i += 1
            elif encoding[i] == 'R':
                suffixes.insert(0, 'restrict ')
                i += 1
            else:
                break
        
        full_type = ""
        if prefixes:
            full_type += " ".join(prefixes) + " "
        
        const_volatile = [s for s in suffixes if s in ['const ', 'volatile ', 'restrict ']]
        pointers = [s for s in suffixes if s in ['*', '&']]
        
        full_type += "".join(const_volatile) + base_type + "".join(pointers)
        
        return full_type.strip(), i

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

    def generate_tablegen_def(self, name: str, return_type: str, param_types: List[str], 
                            attributes: List[str], feature: str) -> str:
        def_name = name.replace('__builtin_s390_', '').replace('__builtin_', '')
        def_name = ''.join(word.capitalize() for word in def_name.split('_'))
        
        if def_name.startswith('390'):
            def_name = 'S' + def_name
        
        if not param_types:
            prototype = f"{return_type}()"
        elif param_types == ["..."]:
            prototype = f"{return_type}(...)"
        else:
            prototype = f"{return_type}({', '.join(param_types)})"
        
        result = f"""def {def_name} : TargetBuiltin {{
  let Spellings = ["{name}"];
  let Prototype = "{prototype}";"""
        
        if attributes:
            attr_str = ', '.join(attributes)
            result += f"\n  let Attributes = [{attr_str}];"
        
        if feature:
            result += f'\n  let Features = ["{feature}"];'
        
        result += "\n}"
        
        return result

    def convert_builtin(self, name: str, proto_encoding: str, attr_encoding: str, feature: str) -> str:
        try:
            return_type, param_types = self.parse_type_encoding(proto_encoding)
            attributes = self.decode_attributes(attr_encoding)
            return self.generate_tablegen_def(name, return_type, param_types, attributes, feature)
        except Exception as e:
            return f"// ERROR converting {name}: {e}\n// Original: TARGET_BUILTIN({name}, \"{proto_encoding}\", \"{attr_encoding}\", \"{feature}\")"

    def convert_file(self, input_file: str, output_file: str = None):
        try:
            with open(input_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: File not found {input_file}")
            return
        
        converted_lines = []
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
        
        conversion_count = 0
        error_count = 0
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith('TARGET_BUILTIN'):
                parsed = self.parse_builtin_line(line)
                if parsed:
                    name, proto, attrs, feature = parsed
                    converted = self.convert_builtin(name, proto, attrs, feature)
                    converted_lines.append(converted)
                    converted_lines.append("")
                    
                    if converted.startswith("// ERROR"):
                        error_count += 1
                    else:
                        conversion_count += 1
                else:
                    converted_lines.append(f"// ERROR: Could not parse line {line_num}: {line}")
                    error_count += 1
        
        output_content = '\n'.join(converted_lines)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(output_content)
            print(f"Conversion completed!")
            print(f"Output file: {output_file}")
            print(f"Successfully converted: {conversion_count} functions")
            if error_count > 0:
                print(f"Conversion errors: {error_count}")
        else:
            print(output_content)

    def test_conversion(self):
        test_cases = [
            ('TARGET_BUILTIN(__builtin_tbegin, "iv*", "j", "transactional-execution")', 
             "Simple pointer type"),
            ('TARGET_BUILTIN(__builtin_s390_lcbb, "UivC*Ii", "nc", "vector")', 
             "Complex mixed types"),
            ('TARGET_BUILTIN(__builtin_s390_vperm, "V16UcV16UcV16UcV16Uc", "nc", "vector")', 
             "Vector types"),
            ('TARGET_BUILTIN(__builtin_s390_vfidb, "V2dV2dIiIi", "nc", "vector")', 
             "Multi-param vectors"),
        ]
        
        print("=== Testing Conversion ===\n")
        for test_case, description in test_cases:
            print(f"Test: {description}")
            print(f"Input: {test_case}")
            
            parsed = self.parse_builtin_line(test_case)
            if parsed:
                name, proto, attrs, feature = parsed
                print(f"Parsed:")
                print(f"  Name: {name}")
                print(f"  Prototype: {proto}")
                print(f"  Attributes: {attrs}")
                print(f"  Feature: {feature}")
                
                converted = self.convert_builtin(name, proto, attrs, feature)
                print(f"Result:")
                print(converted)
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
        print("SystemZ Builtin Function Converter")
        print("Usage:")
        print("  python convert_systemz.py --test                    # Run tests")
        print("  python convert_systemz.py input.def [output.td]     # Convert file")
        print("")
        print("Example:")
        print("  python convert_systemz.py BuiltinsSystemZ.def BuiltinsSystemZ.td")

if __name__ == "__main__":
    main()

