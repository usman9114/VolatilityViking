#!/usr/bin/env python3
"""Quick fix for the df error in async_runner.py"""

with open('src/live/async_runner.py', 'r') as f:
    lines = f.readlines()

# Find and replace the buggy section (around line 1069)
for i in range(len(lines)):
    if 'last_row = df.iloc[-1]' in lines[i]:
        # Found the bug - replace this section with simple version
        # Backtrack to find start of try block
        start = i
        while start > 0 and 'tech_context_str = "{}"' not in lines[start]:
            start -= 1
        
        # Find end of this try/except
        end = i
        while end < len(lines) and 'except Exception as exc:' not in lines[end]:
            end += 1
        end += 2  # Include the exception handler
        
        # Replace with fixed version
        indent = '                          '
        new_code = [
            f'{indent}# --- TECHNICAL CONTEXT (simplified) ---\n',
            f'{indent}try:\n',
            f'{indent}    tech_data = {{\n',
            f'{indent}        "price": float(current_price),\n',
            f'{indent}        "rsi": float(self.rsi.get(symbol, 0)),\n',
            f'{indent}        "atr": float(atr_val),\n',
            f'{indent}        "adx": float(adx_val)\n',
            f'{indent}    }}\n',
            f'{indent}    tech_context_str = ""\n',
            f'{indent}except Exception as exc:\n',
            f'{indent}    tech_context_str = ""\n',
            f'{indent}\n',
        ]
        
        lines[start:end+1] = new_code
        break

with open('src/live/async_runner.py', 'w') as f:
    f.writelines(lines)

print("Fixed df error!")
