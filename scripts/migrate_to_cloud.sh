#!/bin/bash
# Migration Script to 34.63.90.232

echo "Starting Migration to Cloud..."
echo "Target: usman.qureshi@34.63.90.232"

rsync -avz \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'data/raw' \
  --exclude 'data/logs/*.bak' \
  /home/usman/eth-bot/ \
  usman.qureshi@34.63.90.232:~/eth-bot

echo "Migration Command Finished."
