# ETH Trading Bot - Docker Quick Reference

## 🚀 Quick Start

### Load Docker Image
```bash
docker load < eth-bot-docker.tar.gz
```

### Verify Image Loaded
```bash
docker images | grep eth-bot
```

---

## 📋 All Run Commands

### 1️⃣ MAINNET - Conservative Mode (Default)
**60-second checks, reactive grid management**
```bash
docker run -d \
  --name eth-bot \
  --env-file .env \
  --restart unless-stopped \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  eth-bot:latest
```

### 2️⃣ MAINNET - Aggressive Mode
**5-second checks, Passivbot-style proactive grid**
```bash
docker run -d \
  --name eth-bot-aggressive \
  --env-file .env \
  --restart unless-stopped \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  eth-bot:latest \
  python main.py --mode live --symbol ETH/USDT --aggressive
```

### 3️⃣ TESTNET - Conservative Mode
**For testing with testnet API keys**
```bash
docker run -d \
  --name eth-bot-testnet \
  --env-file .env \
  --restart unless-stopped \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  eth-bot:latest \
  python src/live/binance_bot.py --symbol ETH/USDT
```

### 4️⃣ TESTNET - Aggressive Mode
```bash
docker run -d \
  --name eth-bot-testnet-aggressive \
  --env-file .env \
  --restart unless-stopped \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  eth-bot:latest \
  python src/live/binance_bot.py --symbol ETH/USDT --aggressive
```

### 5️⃣ Dashboard Only
**Streamlit dashboard for monitoring**
```bash
docker run -d \
  --name eth-dashboard \
  --env-file .env \
  --restart unless-stopped \
  -p 8501:8501 \
  eth-bot:latest \
  streamlit run dashboard.py
```

Access at: `http://YOUR_VM_IP:8501`

### 6️⃣ Bot + Dashboard (Recommended)
**Run both bot and dashboard together**

```bash
# Start bot
docker run -d \
  --name eth-bot \
  --env-file .env \
  --restart unless-stopped \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  eth-bot:latest

# Start dashboard
docker run -d \
  --name eth-dashboard \
  --env-file .env \
  --restart unless-stopped \
  -p 8501:8501 \
  eth-bot:latest \
  streamlit run dashboard.py
```

---

## 🎛️ Command Options Explained

| Option | Description |
|--------|-------------|
| `--mode live` | Run in live trading mode (mainnet) |
| `--symbol ETH/USDT` | Trading pair |
| `--aggressive` | Enable 5s checks + proactive grid |
| `--env-file .env` | Load API keys from .env file |
| `-v $(pwd)/data:/app/data` | Persist trading data |
| `-v $(pwd)/logs:/app/logs` | Persist logs |
| `-p 8501:8501` | Expose dashboard port |
| `--restart unless-stopped` | Auto-restart on crash |
| `-d` | Run in background (detached) |

---

## 📊 Monitoring & Management

### View Logs
```bash
# Follow logs in real-time
docker logs -f eth-bot

# Last 100 lines
docker logs --tail 100 eth-bot

# Last 50 lines with timestamps
docker logs --tail 50 -t eth-bot
```

### Container Management
```bash
# Stop bot
docker stop eth-bot

# Start bot
docker start eth-bot

# Restart bot
docker restart eth-bot

# Remove container
docker rm -f eth-bot

# Check status
docker ps | grep eth-bot
```

### Resource Usage
```bash
# View CPU/Memory usage
docker stats eth-bot

# View all containers
docker ps -a
```

---

## 🔧 Troubleshooting

### Check if container is running
```bash
docker ps | grep eth-bot
```

### Enter container shell
```bash
docker exec -it eth-bot bash
```

### View environment variables
```bash
docker exec eth-bot env | grep BINANCE
```

### Check .env file
```bash
cat .env
```

### Restart with fresh logs
```bash
docker rm -f eth-bot
# Then run your preferred command again
```

---

## 🔐 Security Notes

1. **Never commit .env file** - Contains API keys
2. **Use read-only API keys** for dashboard
3. **Firewall port 8501** if dashboard is public
4. **Regularly update** Docker image

---

## 📁 File Structure

```
/home/usman.qureshi/
├── eth-bot-docker.tar.gz  # Docker image (715 MB)
├── .env                    # API keys
├── data/                   # Trading data (created by bot)
└── logs/                   # Bot logs (created by bot)
```

---

## ⚡ Performance

| Mode | RAM Usage | CPU Usage | Check Interval |
|------|-----------|-----------|----------------|
| Conservative | ~50 MB | Low | 60 seconds |
| Aggressive | ~100 MB | Medium | 5 seconds |
| Dashboard | ~150 MB | Low | On-demand |

---

## 🆘 Common Issues

### "Permission denied" error
```bash
sudo usermod -aG docker $USER
# Then logout and login again
```

### "Port already in use"
```bash
# Find what's using port 8501
sudo lsof -i :8501
# Kill it or use different port
docker run -p 8502:8501 ...
```

### "Cannot connect to Docker daemon"
```bash
sudo systemctl start docker
```

---

## 📝 Example: Full Deployment

```bash
# 1. Load image
docker load < eth-bot-docker.tar.gz

# 2. Verify .env exists
cat .env

# 3. Run bot (conservative)
docker run -d --name eth-bot --env-file .env \
  --restart unless-stopped \
  -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs \
  eth-bot:latest

# 4. Run dashboard
docker run -d --name eth-dashboard --env-file .env \
  --restart unless-stopped -p 8501:8501 \
  eth-bot:latest streamlit run dashboard.py

# 5. Check logs
docker logs -f eth-bot

# 6. Access dashboard
# Open browser: http://YOUR_VM_IP:8501
```

---

## 🔄 Updating the Bot

```bash
# 1. Stop current bot
docker stop eth-bot && docker rm eth-bot

# 2. Load new image
docker load < eth-bot-docker-new.tar.gz

# 3. Run with same command
docker run -d --name eth-bot --env-file .env \
  --restart unless-stopped \
  -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs \
  eth-bot:latest
```

---

**Need help?** Check logs with `docker logs -f eth-bot`
