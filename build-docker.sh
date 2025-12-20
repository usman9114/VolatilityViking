#!/bin/bash
# Build and Export Docker Image for ETH Trading Bot

set -e  # Exit on error

echo "🐳 Building ETH Trading Bot Docker Image"
echo "========================================"
echo ""

# Build the image
echo "📦 Building image..."
docker build -t eth-bot:latest .

echo ""
echo "✅ Build complete!"
echo ""

# Save to tar file
echo "💾 Saving image to tar file..."
docker save eth-bot:latest | gzip > eth-bot-docker.tar.gz

echo ""
echo "✅ Image saved to: eth-bot-docker.tar.gz"
echo ""

# Show size
SIZE=$(du -h eth-bot-docker.tar.gz | cut -f1)
echo "📊 Image size: $SIZE"
echo ""

echo "🚀 Next steps:"
echo "1. Transfer to other VM:"
echo "   scp eth-bot-docker.tar.gz user@other-vm:/path/to/destination/"
echo ""
echo "2. On the other VM, load the image:"
echo "   docker load < eth-bot-docker.tar.gz"
echo ""
echo "3. Run the bot:"
echo "   docker run -d --name eth-bot --env-file .env eth-bot:latest"
echo ""
