import asyncio
import ccxt.async_support as ccxt_async
import os
import pandas as pd
from src.agents.base_agent import BaseAgent, logger

class ScoutAgent(BaseAgent):
    """
    The Scout: Scans the market for new opportunities and detects market regimes.
    """
    def __init__(self):
        super().__init__()
        
    async def get_top_volume_pairs(self, limit=20):
        """Fetch top volume USDT pairs from Binance Futures."""
        exchange = None
        try:
            # Use Public endpoint (no keys needed for market data usually, but we have them)
            exchange = ccxt_async.binance({
                'options': {'defaultType': 'future'}
            })
            
            # Fetch ALL tickers (more robust than filtering keys first)
            tickers = await exchange.fetch_tickers()
            
            # Filter for USDT futures and valid symbols
            # Note: Binance Futures symbols often look like 'BTC/USDT:USDT'
            usdt_tickers = [
                t for s, t in tickers.items() 
                if '/USDT' in s and 'quoteVolume' in t and t['quoteVolume'] is not None
            ]
            
            # Sort by Quote Volume
            sorted_tickers = sorted(usdt_tickers, key=lambda x: x['quoteVolume'], reverse=True)
            
            top_pairs = []
            for t in sorted_tickers[:limit]:
                # Normalize symbol: "BTC/USDT:USDT" -> "BTC/USDT"
                raw_sym = t['symbol']
                norm_sym = raw_sym.split(':')[0]
                
                top_pairs.append({
                    'symbol': norm_sym,
                    'volume': t['quoteVolume'],
                    'price': t['last'],
                    'change_24h': t.get('percentage', 0)
                })
                
            return top_pairs
            
        except Exception as e:
            logger.error(f"[{self.name}] Scan Error: {e}")
            return []
        finally:
            if exchange: await exchange.close()

    async def analyze_regime(self, symbol):
        """
        Analyze logic for a single symbol to determine regime (Bull/Bear).
        Uses simple ADX/RSI check (can reuse async_runner logic later).
        For now, returns 'neutral', 'bull', or 'bear'.
        """
        # Placeholder for deeper analysis.
        # In a real impl, we would fetch OHLCV here.
        return "neutral"

    async def run(self):
        logger.info(f"[{self.name}] Scanning market...")
        top_coins = await self.get_top_volume_pairs()
        
        # Simple Regime Check on BTC
        btc_regime = await self.analyze_regime("BTC/USDT")
        
        report = {
            "agent": "Scout",
            "market_regime": btc_regime,
            "top_candidates": top_coins[:5]  # Top 5
        }
        
        logger.info(f"[{self.name}] Report Generated: {len(top_coins)} coins scanned.")
        return report

if __name__ == "__main__":
    scan = ScoutAgent()
    loop = asyncio.get_event_loop()
    rep = loop.run_until_complete(scan.run())
    print(rep)
