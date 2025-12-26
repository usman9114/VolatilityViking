#!/usr/bin/env python3
import os
import glob
import time
import re
import sys

# Color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'

LOG_DIR = "logs"

def get_latest_log():
    list_of_files = glob.glob(f'{LOG_DIR}/*.log') 
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def parse_log(filepath):
    print(f"{BOLD}Checking Log:{RESET} {filepath}")
    
    # helper to strip ANSI colors if present in log (not common in file, but good practice)
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    last_status = {}
    errors = []
    last_update_time = 0
    warnings = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            # Check file age first
            file_mod_time = os.path.getmtime(filepath)
            age = time.time() - file_mod_time
            if age > 120:  # 2 minutes without write
                print(f"{RED}[CRITICAL] Log is STALE! Last write was {int(age)}s ago. Bot may be frozen.{RESET}")
            else:
                print(f"{GREEN}[OK] Log is fresh (updated {int(age)}s ago).{RESET}")

            # Parse lines (reverse for speed on big files?) No, simple read is fine for <10MB
            for line in lines[-200:]: # Only check last 200 lines
                clean_line = ansi_escape.sub('', line).strip()
                
                # Timestamp check (approximate from log line)
                # 2025-12-20 18:14:00,397 ...
                
                if "ERROR" in clean_line:
                    errors.append(clean_line)
                if "WARNING" in clean_line:
                    warnings.append(clean_line)
                    
                # [ETH/USDT] Price: ...
                if "] Price:" in clean_line and "ADX:" in clean_line:
                    # Look for [SYMBOL] immediately before "Price:"
                    # Example: ... [INFO] [ETH/USDT] Price: ...
                    match = re.search(r'\[([A-Z0-9/]+)\]\s*Price:\s*([\d\.]+).*ADX:\s*([\d\.]+) .*GARCH:\s*([\d\.]+)%', clean_line)
                    if match:
                        sym, price, adx, garch = match.groups()
                        if sym == "INFO": continue # Skip the log level tag if caught by mistake
                        last_status[sym] = {
                            'price': price,
                            'adx': adx,
                            'garch': garch,
                            'line': clean_line
                        }

    except Exception as e:
        print(f"{RED}Error reading log: {e}{RESET}")
        return

    print(f"\n{BOLD}--- Market Status ---{RESET}")
    if not last_status:
        print(f"{YELLOW}No recent status lines found.{RESET}")
    else:
        for sym, data in last_status.items():
            # Check logic health
            adx = float(data['adx'])
            garch = float(data['garch'])
            
            # Formatting
            adx_color = GREEN if adx < 25 else YELLOW
            if adx > 50: adx_color = RED
            
            print(f"{BOLD}{sym:<10}{RESET} | Price: {data['price']:<8} | ADX: {adx_color}{data['adx']:<5}{RESET} | GARCH: {data['garch']}%")

    print(f"\n{BOLD}--- Health Check ---{RESET}")
    if errors:
        print(f"{RED}Found {len(errors)} ERRORS in last 200 lines:{RESET}")
        for e in errors[-3:]:
            print(f"  {e}")
    else:
        print(f"{GREEN}No ERRORS found in recent logs.{RESET}")
        
    if warnings:
        print(f"{YELLOW}Found {len(warnings)} WARNINGS.{RESET}")

if __name__ == "__main__":
    latest = get_latest_log()
    if latest:
        parse_log(latest)
    else:
        print(f"{RED}No log files found in {LOG_DIR}{RESET}")
