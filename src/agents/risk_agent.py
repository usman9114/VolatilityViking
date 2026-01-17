import asyncio
import ccxt.async_support as ccxt_async
import os
from src.agents.base_agent import BaseAgent, logger

class RiskAgent(BaseAgent):
    """
    The Risk Officer: Audits portfolio health, checks drawdown, and recommends safety measures.
    """
    def __init__(self):
        super().__init__()

    async def audit_portfolio(self):
        """
        Connects to exchange and checks real equity vs balance.
        """
        exchange = None
        try:
            live_mode = self.config.get('live_mode', False)
            if live_mode:
                api_key = os.getenv('BINANCE_API_KEY')
                secret = os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET')
                if not api_key: logger.warning("Live Mode enabled but BINANCE_API_KEY missing!")
            else:
                api_key = os.getenv('BINANCE_TESTNET_API_KEY')
                secret = os.getenv('BINANCE_TESTNET_SECRET_KEY')
                if not api_key: logger.warning("Testnet Mode enabled but BINANCE_TESTNET_API_KEY missing!")
            
            # Default to checking Futures if enabled
            futures_enabled = self.config.get('futures_settings', {}).get('enabled', False)
            default_type = 'future' if futures_enabled else 'spot'
            
            exchange = ccxt_async.binance({
                'apiKey': api_key,
                'secret': secret,
                'options': {'defaultType': default_type}
            })
            
            # Fetch Balance & Positions
            balance = await exchange.fetch_balance()
            total_equity = balance.get('USDT', {}).get('total', 0)
            
            positions = []
            if default_type == 'future':
                raw_pos = await exchange.fetch_positions()
                
                # Fetch Open Orders Counts (Parallel)
                # We need to know if grids are active or stuck
                active_symbols = [p['symbol'] for p in raw_pos if float(p.get('contracts', 0)) != 0]
                
                open_orders_map = {}
                if active_symbols:
                    # quick fetch of open orders for active positions
                    tasks = [exchange.fetch_open_orders(sym) for sym in active_symbols]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for sym, res in zip(active_symbols, results):
                        if isinstance(res, list):
                            open_orders_map[sym] = len(res)
                        else:
                            logger.warning(f"[{self.name}] Failed to fetch orders for {sym}: {res}")
                            open_orders_map[sym] = 0

                for p in raw_pos:
                    amt = float(p.get('contracts', 0))
                    if amt != 0:
                        entry_price = float(p.get('entryPrice', 0))
                        mark_price = float(p.get('markPrice', 0))
                        liq_price = float(p.get('liquidationPrice', 0))
                        
                        dist_to_liq = 0
                        if liq_price > 0 and mark_price > 0:
                            dist_to_liq = abs(mark_price - liq_price) / mark_price

                        positions.append({
                            'symbol': p['symbol'],
                            'amount': amt,
                            'pnl': float(p['unrealizedPnl']),
                            'side': p['side'],
                            'grid_orders': open_orders_map.get(p['symbol'], 0),
                            'dist_to_liq_pct': dist_to_liq
                        })
            
            # Calculate Total PnL
            total_pnl = sum(p['pnl'] for p in positions)
            
            # Safety Score (Simple Heuristic)
            # 100 = No positions or high profit
            # < 50 = High drawdown
            safety_score = 100
            if total_equity > 0:
                pnl_pct = total_pnl / total_equity
                if pnl_pct < -0.10: safety_score = 80
                if pnl_pct < -0.20: safety_score = 60
                if pnl_pct < -0.30: safety_score = 40
            
            return {
                "total_equity": total_equity,
                "total_unrealized_pnl": total_pnl,
                "safety_score": safety_score,
                "positions_count": len(positions),
                "flagged_positions": [p for p in positions if p['pnl'] < -50] # Flag losers > $50
            }
            
        except Exception as e:
            logger.error(f"[{self.name}] Audit Error: {e}")
            return {"error": str(e)}
        finally:
            if exchange: await exchange.close()

    async def run(self):
        logger.info(f"[{self.name}] Auditing portfolio...")
        audit = await self.audit_portfolio()
        
        report = {
            "agent": "RiskOfficer",
            "audit": audit,
            "recommendation": "Maintain" if audit.get('safety_score', 100) > 70 else "Reduce Exposure"
        }
        
        return report

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('profiles/usman/.env')
    agent = RiskAgent()
    loop = asyncio.get_event_loop()
    rep = loop.run_until_complete(agent.run())
    print(rep)
