
import argparse
import os
import sys
import subprocess
import shutil

PROFILES_DIR = "profiles"

def list_profiles():
    if not os.path.exists(PROFILES_DIR):
        print("No profiles found.")
        return
    profiles = [d for d in os.listdir(PROFILES_DIR) if os.path.isdir(os.path.join(PROFILES_DIR, d))]
    print(f"Available Profiles: {profiles}")

def create_profile(name):
    path = os.path.join(PROFILES_DIR, name)
    if os.path.exists(path):
        print(f"Profile '{name}' already exists.")
        return
    
    os.makedirs(path)
    # Copy defaults (Only .env, as config is shared)
    if os.path.exists(".env"):
        shutil.copy(".env", os.path.join(path, ".env"))
    
    print(f"✅ Created profile '{name}' at {path}")

def run_profile(name, args):
    path = os.path.join(PROFILES_DIR, name)
    if not os.path.exists(path):
        print(f"Profile '{name}' not found.")
        return

    # User Request: Shared Config (Default), but allow Override
    profile_config = os.path.join(path, "config.json")
    if os.path.exists(profile_config):
        config_file = profile_config
        print(f"⚙️ Using Custom Profile Config: {config_file}")
    else:
        config_file = "config.json" 
    env_file = os.path.join(path, ".env")
    
    cmd = [sys.executable, "main.py", "--mode", "live", "--config", config_file, "--env-file", env_file, "--profile", name]
    if args.async_mode:
         cmd.append("--async-mode")
         
    print(f"🚀 Launching Bot for User: {name.upper()}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopped.")

def main():
    parser = argparse.ArgumentParser(description="VolatilityViking Profile Manager")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List
    subparsers.add_parser("list", help="List all profiles")
    
    # Create
    create_parser = subparsers.add_parser("create", help="Create a new profile")
    create_parser.add_argument("name", help="Profile name (e.g. 'alice')")
    
    # Run (Explicit)
    run_parser = subparsers.add_parser("run", help="Run bot for a profile")
    run_parser.add_argument("name", help="Profile name")
    run_parser.add_argument("--async-mode", action="store_true", help="Use Asian Mode (Async)")
    
    # Global args for flags
    parser.add_argument("--async-mode", action="store_true", help="Use Async Mode (for flag usage)")
    
    # Dynamic Profile Flags (e.g. --usman)
    existing_profiles = []
    if os.path.exists(PROFILES_DIR):
        existing_profiles = [d for d in os.listdir(PROFILES_DIR) if os.path.isdir(os.path.join(PROFILES_DIR, d))]
        for p in existing_profiles:
            parser.add_argument(f"--{p}", action="store_true", help=f"Run bot for {p}")

    args = parser.parse_args()
    
    # Handle Dynamic Flags
    for p in existing_profiles:
        if getattr(args, p):
            run_profile(p, args)
            return

    if args.command == "list":
        list_profiles()
    elif args.command == "create":
        create_profile(args.name)
    elif args.command == "run":
        run_profile(args.name, args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
