#!/bin/bash
# ETH-Bot Directory Cleanup Script
# This will free up ~12 GB of space!

echo "🧹 ETH-Bot Cleanup Script"
echo "========================="
echo ""
echo "This will delete:"
echo "  - eth-bot.zip (3.4 GB)"
echo "  - passivbot/ (50 MB)"
echo "  - deployment.tar.gz (114 MB)"
echo "  - src/dashboard/ (old dashboard)"
echo "  - .pytest_cache/"
echo "  - grid_bot (old script)"
echo ""
echo "Total space to free: ~3.6 GB"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "🗑️  Deleting files..."

# Delete large files
if [ -f "eth-bot.zip" ]; then
    echo "  Removing eth-bot.zip (3.4 GB)..."
    rm eth-bot.zip
fi

if [ -d "passivbot" ]; then
    echo "  Removing passivbot/ (50 MB)..."
    rm -rf passivbot/
fi

if [ -f "deployment.tar.gz" ]; then
    echo "  Removing deployment.tar.gz (114 MB)..."
    rm deployment.tar.gz
fi

if [ -d "src/dashboard" ]; then
    echo "  Removing src/dashboard/ (old dashboard)..."
    rm -rf src/dashboard/
fi

if [ -d ".pytest_cache" ]; then
    echo "  Removing .pytest_cache/..."
    rm -rf .pytest_cache/
fi

if [ -f "grid_bot" ]; then
    echo "  Removing grid_bot (old script)..."
    rm grid_bot
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Space freed: ~3.6 GB"
echo ""
echo "Kept:"
echo "  ✓ dashboard.py (new Streamlit dashboard)"
echo "  ✓ venv/ (your active virtual environment)"
echo "  ✓ .venv/ (alternative virtual environment)"
echo "  ✓ src/ (core source code)"
echo "  ✓ data/ (trading data)"
echo "  ✓ logs/ (bot logs)"
echo "  ✓ models/ (trained models)"
echo ""
