#!/usr/bin/env python3
"""
Python 3 Syntax Conversion Script

Automates Phase 1 syntax conversions:
1. print statements -> print() functions
2. dict.iteritems/iterkeys/itervalues -> .items()/.keys()/.values()
3. xrange() -> range()
4. except Exception, e: -> except Exception as e:
5. file() -> open()
"""

import re
import os
import sys
from pathlib import Path

def convert_print_statements(content):
    """Convert print statements to print() functions"""
    lines = content.split('\n')
    converted = []
    changes = 0

    for line in lines:
        # Skip lines that already use print()
        if re.match(r'^\s*print\s*\(', line):
            converted.append(line)
            continue

        # Match print statements (not already function calls)
        match = re.match(r'^(\s*)print\s+(.+)$', line)
        if match:
            indent = match.group(1)
            args = match.group(2)
            # Remove trailing comments
            if '#' in args:
                # Be careful not to remove # inside strings
                parts = args.split('#', 1)
                args = parts[0].rstrip()
                comment = ' #' + parts[1]
            else:
                comment = ''
            converted.append(f'{indent}print({args}){comment}')
            changes += 1
        else:
            converted.append(line)

    return '\n'.join(converted), changes

def convert_dict_methods(content):
    """Convert dict.iteritems/iterkeys/itervalues to items/keys/values"""
    changes = 0

    # .iteritems() -> .items()
    new_content, count = re.subn(r'\.iteritems\(\)', '.items()', content)
    changes += count

    # .iterkeys() -> .keys()
    new_content, count = re.subn(r'\.iterkeys\(\)', '.keys()', new_content)
    changes += count

    # .itervalues() -> .values()
    new_content, count = re.subn(r'\.itervalues\(\)', '.values()', new_content)
    changes += count

    return new_content, changes

def convert_xrange(content):
    """Convert xrange() to range()"""
    changes = 0
    new_content, count = re.subn(r'\bxrange\b', 'range', content)
    changes += count
    return new_content, changes

def convert_exception_syntax(content):
    """Convert except Exception, e: to except Exception as e:"""
    changes = 0
    # Pattern: except <exception>, <var>:
    new_content, count = re.subn(
        r'except\s+([\w\.]+)\s*,\s*(\w+)\s*:',
        r'except \1 as \2:',
        content
    )
    changes += count
    return new_content, changes

def convert_file_builtin(content):
    """Convert file() to open()"""
    changes = 0
    new_content, count = re.subn(r'\bfile\s*\(', 'open(', content)
    changes += count
    return new_content, changes

def add_future_imports(content, filename):
    """Add from __future__ import division, print_function to top of file"""
    lines = content.split('\n')

    # Check if already has future imports
    if 'from __future__ import' in content:
        return content, 0

    # Find where to insert: after module docstring, before other imports
    insert_pos = 0
    in_docstring = False
    docstring_char = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip shebang
        if i == 0 and stripped.startswith('#!'):
            insert_pos = i + 1
            continue

        # Skip encoding declarations
        if stripped.startswith('#') and ('coding' in stripped or 'encoding' in stripped):
            insert_pos = i + 1
            continue

        # Handle module docstrings
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = True
                docstring_char = stripped[:3]
                # Check if single-line docstring
                if stripped.endswith(docstring_char) and len(stripped) > 6:
                    in_docstring = False
                    insert_pos = i + 1
            elif stripped.startswith('"') or stripped.startswith("'"):
                # Single-line docstring with single quotes
                insert_pos = i + 1
            elif stripped == '' or stripped.startswith('#'):
                # Empty line or comment, can skip
                if insert_pos == i:
                    insert_pos = i + 1
            else:
                # Hit actual code, insert before this
                break
        else:
            # Inside docstring, look for end
            if docstring_char in line:
                in_docstring = False
                insert_pos = i + 1

    # Insert the future import
    future_import = 'from __future__ import division, print_function'

    # Add blank line before if needed
    if insert_pos > 0 and insert_pos < len(lines) and lines[insert_pos-1].strip() != '':
        lines.insert(insert_pos, '')
        insert_pos += 1

    lines.insert(insert_pos, future_import)

    # Add blank line after if next line is not blank
    if insert_pos + 1 < len(lines) and lines[insert_pos + 1].strip() != '':
        lines.insert(insert_pos + 1, '')

    return '\n'.join(lines), 1

def convert_file(filepath):
    """Convert a single Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read()

    original = content
    total_changes = 0
    changes_by_type = {}

    # Apply conversions
    content, changes = convert_print_statements(content)
    if changes:
        changes_by_type['print'] = changes
        total_changes += changes

    content, changes = convert_dict_methods(content)
    if changes:
        changes_by_type['dict_methods'] = changes
        total_changes += changes

    content, changes = convert_xrange(content)
    if changes:
        changes_by_type['xrange'] = changes
        total_changes += changes

    content, changes = convert_exception_syntax(content)
    if changes:
        changes_by_type['exception'] = changes
        total_changes += changes

    content, changes = convert_file_builtin(content)
    if changes:
        changes_by_type['file'] = changes
        total_changes += changes

    content, changes = add_future_imports(content, filepath)
    if changes:
        changes_by_type['future_imports'] = changes
        total_changes += changes

    # Write back if changes made
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes_by_type

    return False, {}

def main():
    """Main conversion routine"""
    base_dir = Path('/home/dude/statarb')

    # Find all .py files (excluding salamander directory for now)
    py_files = []
    for py_file in base_dir.glob('*.py'):
        if py_file.name != 'convert_syntax_py3.py':  # Skip this script
            py_files.append(py_file)

    print(f"Found {len(py_files)} Python files to convert")
    print()

    total_files_changed = 0
    all_changes = {
        'print': 0,
        'dict_methods': 0,
        'xrange': 0,
        'exception': 0,
        'file': 0,
        'future_imports': 0
    }

    for py_file in sorted(py_files):
        changed, changes = convert_file(py_file)
        if changed:
            total_files_changed += 1
            print(f"✓ {py_file.name}")
            for change_type, count in changes.items():
                print(f"  - {change_type}: {count} changes")
                all_changes[change_type] += count
        else:
            print(f"  {py_file.name} (no changes)")

    print()
    print("="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Files modified: {total_files_changed}/{len(py_files)}")
    print()
    print("Changes by type:")
    for change_type, count in all_changes.items():
        if count > 0:
            print(f"  - {change_type}: {count}")
    print()

if __name__ == '__main__':
    main()
