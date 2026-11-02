import numpy as np
import logging

logger = logging.getLogger(__name__)

class RecursiveGrid:
    """
    Implements recursive grid logic similar to Passivbot.
    Calculates the ideal grid structure based on:
    - Current Price (Reference)
    - Available Balance (Equity)
    - Grid Parameters (Spacing, DDown Factor, Wallet Exposure)
    """

    @staticmethod
    def calc_recursive_grid(
        current_price: float,
        balance: float,
        params: dict,
        side: str = 'long'
    ):
        """
        Calculate grid levels recursively.
        
        Args:
            current_price: Current market price (or last entry price).
            balance: Total equity available for this symbol.
            params: Dictionary containing grid parameters:
                - grid_span (float): Total coverage (e.g., 0.3 for 30%)
                - wallet_exposure_limit (float): Max exposure (e.g., 1.0)
                - max_orders (int): Number of grid orders
                - spacing_weight (float): Geometric spacing factor
                - qty_step_multiplier (float): Martingale multiplier
                - ddown_factor (float): Dynamic shaper (optional)
        
        Returns:
            list of dicts: [{'price': p, 'qty': q, 'dist': d}, ...]
        """
        try:
            n_orders = int(params.get('max_orders', 10))
            grid_span = float(params.get('grid_span', 0.2))
            wel = float(params.get('wallet_exposure_limit', 1.0))
            spacing_weight = float(params.get('grid_spacing_weight', 1.2))
            qty_multiplier = float(params.get('qty_step_multiplier', 1.4))
            
            # Initial Order Size (Start small)
            # Concept: Total Wallet Exposure = Sum of all orders
            # We solve for 'initial_qty' such that Sum(initial * mult^i) = Total * WEL
            
            # Geometric Sum Formula: S_n = a * (r^n - 1) / (r - 1)
            if qty_multiplier != 1.0:
                geo_sum_factor = (pow(qty_multiplier, n_orders) - 1) / (qty_multiplier - 1)
            else:
                geo_sum_factor = n_orders
                
            total_budget = balance * wel
            initial_cost = total_budget / geo_sum_factor
            
            # Distance Calculation
            if spacing_weight != 1.0:
                # Solve for base_step such that accumulated distance = grid_span
                # dist_i = base * weight^i
                dist_sum_factor = (pow(spacing_weight, n_orders) - 1) / (spacing_weight - 1)
                base_dist = grid_span / dist_sum_factor
            else:
                base_dist = grid_span / n_orders

            orders = []
            current_p = current_price
            current_q_cost = initial_cost
            cum_dist = 0.0

            for i in range(n_orders):
                # Calculate Price
                step_dist = base_dist * pow(spacing_weight, i)
                cum_dist += step_dist
                
                if side == 'long':
                    price = current_price * (1 - cum_dist)
                else: 
                    price = current_price * (1 + cum_dist)
                
                # Calculate Qty
                qty_cost = initial_cost * pow(qty_multiplier, i)
                qty_base = qty_cost / price
                
                orders.append({
                    'id': i+1,
                    'price': price,
                    'qty_base': qty_base,
                    'qty_quote': qty_cost,
                    'dist_pct': cum_dist
                })
                
            return orders

        except Exception as e:
            logger.error(f"Recursive Grid Error: {e}")
            return []

if __name__ == "__main__":
    # Test
    orders = RecursiveGrid.calc_recursive_grid(
        1000.0, 
        1000.0, 
        {
            'grid_span': 0.15, 
            'max_orders': 5, 
            'wallet_exposure_limit': 1.0,
            'qty_step_multiplier': 1.5,
            'grid_spacing_weight': 1.2
        }
    )
    import json
    print(json.dumps(orders, indent=2))
