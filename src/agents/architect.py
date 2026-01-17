import asyncio
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
from src.agents.base_agent import BaseAgent, logger

class ArchitectAgent(BaseAgent):
    """
    The Architect: The Math Quant.
    Optimizes grid geometry (span, spacing) based on Market Texture (Vol + Trend).
    """
    def __init__(self):
        super().__init__()
        
    async def fetch_ohlcv(self, symbol, timeframe='4h', limit=100):
        exchange = None
        try:
            # Revert to public for market data to save rate limits
            exchange = ccxt_async.binance({
                'options': {'defaultType': 'future'}
            })
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except Exception as e:
            logger.error(f"[{self.name}] Data Fetch Error for {symbol}: {e}")
            return pd.DataFrame()
        finally:
            if exchange: await exchange.close()

    def calculate_indicators(self, df):
        if df.empty: return 0, 0
        
        # ATR (14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        # Simple ADX Approximation (DX) for speed
        # Real ADX is complex, we use High-Low expansion as proxy for now or simple trend strength
        # Using simple Slope of EMA(20) as Trend Strength proxy if ADX too complex for vanilla pandas
        # Let's do a simplified ADX
        
        up = df['high'] - df['high'].shift(1)
        down = df['low'].shift(1) - df['low']
        
        # ... actually, let's stick to standard Volatility for now to be safe.
        # Volatility = ATR / Price
        current_price = df['close'].iloc[-1]
        vol_pct = (atr / current_price) * 100
        
        return vol_pct, 0 # Placeholder for ADX or implement later if needed

    async def optimize_symbol(self, symbol):
        df = await self.fetch_ohlcv(symbol)
        if df.empty: return None

        vol_pct, _ = self.calculate_indicators(df)
        
        # Logic Matrix
        # Base Grid Span = 30% (0.30)
        # Low Vol (< 2% daily move) -> Tighten to 15%
        # High Vol (> 5% daily move) -> Widen to 50%
        
        rec_span = 0.30
        mode = "NEUTRAL"
        
        if vol_pct < 1.5:
            rec_span = 0.15
            mode = "COMPRESSED (Low Vol)"
        elif vol_pct > 5.0:
            rec_span = 0.50
            mode = "EXPANDED (High Vol)"
        elif vol_pct > 3.0:
            rec_span = 0.40
            mode = "ELEVATED"
            
        return {
            "symbol": symbol,
            "volatility_4h": round(vol_pct, 2),
            "mode": mode,
            "recommended_span": rec_span
        }

    async def run(self):
        logger.info(f"[{self.name}] Analyzing Market Structure...")
        report = {
            "agent": "The Architect",
            "optimizations": []
        }
        
        symbols = self.config.get('symbols', [])
        for s in symbols:
            opt = await self.optimize_symbol(s)
            if opt:
                report['optimizations'].append(opt)
                
        logger.info(f"[{self.name}] Generated {len(report['optimizations'])} optimizations.")
        return report

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('profiles/usman/.env')
    arch = ArchitectAgent()
    loop = asyncio.get_event_loop()
    print(loop.run_until_complete(arch.run()))
