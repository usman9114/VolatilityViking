#!/usr/bin/env python3
"""
Complete restoration of async_runner.py with all session changes
"""

with open('src/live/async_runner.py', 'r') as f:
    lines = f.readlines()

# Find and update key sections
new_lines = []
skip_until = None

for i, line in enumerate(lines):
    # Skip lines if we're in a replacement block
    if skip_until and i < skip_until:
        continue
    skip_until = None
    
    # 1. Fix logging setup (lines 16-29)
    if i == 15 and 'os.makedirs' in line:
        new_lines.extend([
            '\n',
            '# Setup logging with profile support\n',
            'def setup_logging(profile_name="default"):\n',
            '    os.makedirs("logs", exist_ok=True)\n',
            '    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")\n',
            '    log_filename = f"logs/{profile_name}_{timestamp}.log"\n',
            '    \n',
            '    logging.getLogger().handlers = []\n',
            '    logging.basicConfig(\n',
            '        level=logging.INFO,\n',
            '        format="%(asctime)s [%(levelname)s] %(message)s",\n',
            '        handlers=[\n',
            '            logging.FileHandler(log_filename),\n',
            '            logging.StreamHandler()\n',
            '        ]\n',
            '    )\n',
            '    logger = logging.getLogger(__name__)\n',
            '    logger.info(f"Log file: {log_filename}")\n',
            '    return logger\n',
            '\n',
            'logger = logging.getLogger(__name__)\n',
            '\n',
        ])
        # Skip original logging setup
        skip_until = 30
        continue
    
    # 2. Update main() at bottom
    if 'if __name__ == "__main__":' in line:
        new_lines.append(line)
        new_lines.extend([
            '    import argparse\n',
            '    parser = argparse.ArgumentParser()\n',
            '    parser.add_argument("--config", required=True)\n',
            '    parser.add_argument("--profile", default="default")\n',
            '    args = parser.parse_args()\n',
            '    \n',
            '    # Setup logging with profile name\n',
            '    logger = setup_logging(args.profile)\n',
            '    \n',
            '    # Load profile .env\n',
            '    env_path = f"profiles/{args.profile}/.env"\n',
            '    if os.path.exists(env_path):\n',
            '        logger.info(f"Loading env from {env_path}")\n',
            '        load_dotenv(env_path, override=True)\n',
            '    \n',
        ])
        continue
    
    # 3. Skip old argparse if it exists
    if 'parser = argparse.ArgumentParser' in line and i > 1000:
        # Skip until we find asyncio.run
        while i < len(lines) and 'asyncio.run' not in lines[i]:
            i += 1
        continue
    
    new_lines.append(line)

# Write restored file
with open('src/live/async_runner.py', 'w') as f:
    f.writelines(new_lines)

print("Step 1/3: Profile support restored ✅")
