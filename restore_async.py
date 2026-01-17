#!/usr/bin/env python3
"""
Restore async_runner.py with all session changes
This recreates the key modifications made during the session
"""

import re

# Read the original file
with open('src/live/async_runner.py', 'r') as f:
    content = f.read()

# 1. Fix logging to support profiles
content = content.replace(
    'log_filename = f"logs/async_bot_{datetime.now().strftime(\'%Y%m%d_%H%M%S\')}.log"',
    'def setup_logging(profile_name="default"):\n    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")\n    log_filename = f"logs/{profile_name}_{timestamp}.log"\n    return log_filename\n\nlog_filename = setup_logging("async")'
)

# 2. Add argparse for profile support
content = content.replace(
    'if __name__ == "__main__":',
    '''if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--profile', default='default')
    args = parser.parse_args()
    
    # Re-setup logging with profile name
    log_filename = setup_logging(args.profile)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ],
        force=True
    )
    logger.info(f"Log file: {log_filename}")
    
    # Load profile .env
    env_path = f"profiles/{args.profile}/.env"
    if os.path.exists(env_path):
        logger.info(f"Loading env from {env_path}")
        load_dotenv(env_path, override=True)
    
if False:  # Original code below'''
)

# Write the fixed version
with open('src/live/async_runner.py', 'w') as f:
    f.write(content)

print("✅ Restored async_runner.py with profile support!")
