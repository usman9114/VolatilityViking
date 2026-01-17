"""
State Manager - Tracks persisted state like average entry prices.
Identical to Passivbot's state management philosophy but simplified.
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class StateManager:
    def __init__(self, file_path="data/bot_state.json"):
        self.file_path = file_path
        self.state = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    self.state = json.load(f)
            else:
                self.state = {"positions": {}}
                self._save()
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            self.state = {"positions": {}}

    def reload(self):
        """Force reload state from disk (for hot-swapping)."""
        self._load()

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def update_position(self, symbol, side, price, qty):
        """
        Update weighted average entry price.
        side: 'buy' (increases pos) or 'sell' (decreases pos) [Spot logic]
        """
        pos = self.state.get("positions", {}).get(symbol, {"amount": 0.0, "avg_price": 0.0})
        
        current_amt = float(pos.get("amount", 0.0))
        current_price = float(pos.get("avg_price", 0.0))
        
        qty = float(qty)
        price = float(price)
        
        if side == 'buy':
            # Weighted Average
            new_amt = current_amt + qty
            if new_amt > 0:
                new_price = ((current_amt * current_price) + (qty * price)) / new_amt
            else:
                new_price = price
            
            pos["amount"] = new_amt
            pos["avg_price"] = new_price
            
        elif side == 'sell':
            # Selling reduces amount but doesn't change avg entry price (FIFO/LIFO doesn't matter for avg)
            new_amt = max(0, current_amt - qty)
            pos["amount"] = new_amt
            if new_amt == 0:
                pos["avg_price"] = 0.0 # Reset if closed
        
        self.state.setdefault("positions", {})[symbol] = pos
        self._save()
        logger.info(f"[{symbol}] State Updated: Amt={pos['amount']:.4f}, Entry={pos['avg_price']:.4f}")

    def get_avg_entry(self, symbol):
        """Get average entry price for a symbol."""
        return self.state.get("positions", {}).get(symbol, {}).get("avg_price", 0.0)

    async def sync_from_history(self, exchange, symbols):
        """
        Reconstruct average entry price from exchange trade history.
        Logic: Matches current balance against recent BUY history to find weighted avg cost.
        """
        try:
            # 1. Get current balances
            balances = await exchange.fetch_balance()
            
            for symbol in symbols:
                base_currency = symbol.split('/')[0]
                total_balance = float(balances.get(base_currency, {}).get('total', 0.0))
                
                if total_balance <= 0:
                    continue
                    
                logger.info(f"[{symbol}] Syncing history for balance: {total_balance} {base_currency}")
                
                # 2. Fetch trades (last 90 days? or limit=1000)
                # We need enough trades to cover the current balance
                trades = await exchange.fetch_my_trades(symbol, limit=200) 
                
                # 3. Calculate Weighted Avg Entry
                # Iterate backwards (LIFO) to find which trades make up the current balance
                remaining_qty = total_balance
                total_cost = 0.0
                covered_qty = 0.0
                
                # Sort trades newest first
                sorted_trades = sorted(trades, key=lambda x: x['timestamp'], reverse=True)
                
                for trade in sorted_trades:
                    if remaining_qty <= 0:
                        break
                        
                    side = trade['side']
                    price = float(trade['price'])
                    qty = float(trade['amount'])
                    
                    if side == 'buy':
                        # This buy contributes to our holding
                        take_qty = min(remaining_qty, qty)
                        total_cost += take_qty * price
                        covered_qty += take_qty
                        remaining_qty -= take_qty
                        
                    elif side == 'sell':
                        # Sells reduce the balance we are looking for? 
                        # Actually, simplified Logic: "What is the average price of the coins I HOLD?"
                        # Standard accounting matches current balance to MOST RECENT buys.
                        # We ignore sells encountered in history because they removed OLDER coins (FIFO) 
                        # or SPECIFIC coins (LIFO). 
                        # For "Average Cost Basis", usually we assume the remaining coins come from the most recent buys.
                        pass
                
                if covered_qty > 0:
                    avg_price = total_cost / covered_qty
                    logger.info(f"[{symbol}] Calculated Avg Entry: {avg_price:.4f} (based on {covered_qty:.4f} coins)")
                    
                    # Update State
                    self.state.setdefault("positions", {})[symbol] = {
                        "amount": total_balance,
                        "avg_price": avg_price
                    }
                else:
                    logger.warning(f"[{symbol}] Could not find enough buy history to cover balance.")
            
            self._save()
            return True
            
        except Exception as e:
            logger.error(f"Sync history error: {e}")
            return False

    def set_manual_override(self, symbol, price, amount):
        """Manually set position state (useful for fixing sync issues)."""
        self.state.setdefault("positions", {})[symbol] = {
            "amount": float(amount),
            "avg_price": float(price)
        }
        self._save()
