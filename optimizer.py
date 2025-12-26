import random
import json
import os
import sys
import copy

# Add root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backtest.garch_backtester import GarchBacktester

class GeneticOptimizer:
    def __init__(self, data_path, pop_size=10, generations=5):
        self.data_path = data_path
        self.pop_size = pop_size
        self.generations = generations
        self.population = []
        
        # Base config
        with open('config.json', 'r') as f:
            self.base_config = json.load(f)
            
    def init_population(self):
        self.population = []
        for _ in range(self.pop_size):
            individual = {
                'grid_span': random.uniform(0.01, 0.15),
                'qty_step_multiplier': random.uniform(1.1, 1.6), # Constrain to safe range
                'volatility_scale': random.uniform(0.5, 2.0),
                'trend_bias_strength': random.uniform(0.1, 1.0)
            }
            self.population.append(individual)
            
    def fitness(self, individual):
        # Create config for this individual
        config = copy.deepcopy(self.base_config)
        config['grid_settings']['grid_span'] = individual['grid_span']
        config['grid_settings']['qty_step_multiplier'] = individual['qty_step_multiplier']
        config['smart_grid']['volatility_scale'] = individual['volatility_scale']
        config['smart_grid']['trend_bias_strength'] = individual['trend_bias_strength']
        
        # Write temp config
        temp_conf = 'config_opt.json'
        with open(temp_conf, 'w') as f:
            json.dump(config, f)
            
        # Run Backtest
        bt = GarchBacktester(temp_conf)
        bt.load_data(self.data_path)
        
        # SLICE DATA: Last 30 days only for speed (30 * 24 * 60 = 43200 rows)
        if len(bt.df) > 43200:
            bt.df = bt.df.iloc[-43200:]
        
        # Suppress print
        import sys
        import io
        suppress_text = io.StringIO()
        sys.stdout = suppress_text 
        
        try:
            history = bt.run()
            final_equity = history[-1]['equity']
            profit_pct = (final_equity - bt.initial_balance) / bt.initial_balance
        except Exception:
            profit_pct = -1.0 # Penalize errors
            
        sys.stdout = sys.__stdout__
        return profit_pct
        
    def run(self):
        self.init_population()
        
        for g in range(self.generations):
            print(f"--- Generation {g+1}/{self.generations} ---")
            
            # Evaluate
            scores = []
            for ind in self.population:
                score = self.fitness(ind)
                scores.append((score, ind))
                print(f"Score: {score*100:.2f}% | Span: {ind['grid_span']:.3f}, Vol: {ind['volatility_scale']:.2f}")
                
            # Sort
            scores.sort(key=lambda x: x[0], reverse=True)
            best_score, best_ind = scores[0]
            print(f"Gen {g+1} Best: {best_score*100:.2f}%")
            print(f"Best Params: {best_ind}")
            
            # Selection (Top 50%)
            survivors = [x[1] for x in scores[:self.pop_size//2]]
            
            # Mutation / Crossover (fill rest)
            new_pop = survivors[:]
            while len(new_pop) < self.pop_size:
                parent = random.choice(survivors)
                child = parent.copy()
                # Mutate
                if random.random() < 0.3: child['grid_span'] *= random.uniform(0.9, 1.1)
                if random.random() < 0.3: child['volatility_scale'] *= random.uniform(0.9, 1.1)
                new_pop.append(child)
            
            self.population = new_pop
            
        return scores[0][1]

if __name__ == "__main__":
    # Check for data
    import os
    if os.path.exists("data/ETHUSDT_1m.csv"):
        opt = GeneticOptimizer("data/ETHUSDT_1m.csv", pop_size=6, generations=3)
        best = opt.run()
        print("\nOptimization Complete. Best Parameters:")
        print(json.dumps(best, indent=4))
    else:
        print("Error: No data found at data/ETHUSDT_1m.csv")
        print("Please fetch data first.")
