import random
import matplotlib.pyplot as plt
import numpy as np

class RiskManagementSimulator:
    def __init__(self, initial_balance=1000, threshold_pct=0.005, max_risk_pool_pct=0.05, k_value=20):
        # Constants
        self.THRESHOLD_PCT = threshold_pct  # 0.5%
        self.MAX_RISK_POOL_PCT = max_risk_pool_pct  # 5%
        self.K_VALUE = k_value  # K value for risk adjustment formula
        
        # Initial state
        self.initial_balance = initial_balance
        self.account_size = initial_balance
        self.risk_pool = initial_balance * self.THRESHOLD_PCT  # Start at 0.5% of account
        
        # Win rate tracking for last 8 trades
        self.last_8_trades = []
        self.win_rate = 0
        self.trading_well = False
        
        # Trade history
        self.balance_history = [initial_balance]
        self.risk_pool_history = [self.risk_pool]
        self.win_rate_history = [0]
        
        # Logs
        self.logs = []
        self.log(f"Initial Balance: ${self.account_size:.2f}")
        self.log(f"Initial Risk Pool: ${self.risk_pool:.2f} ({(self.risk_pool/self.account_size*100):.2f}%)")
    
    def log(self, message):
        self.logs.append(message)
        print(message)
    
    def calculate_reduced_risk_pool(self, risk_pool, loss_amount):
        """Formula for reducing risk pool when below threshold"""
        reduction = loss_amount * risk_pool / (risk_pool + self.K_VALUE)
        return risk_pool - reduction
    
    def calculate_increased_risk_pool(self, risk_pool, win_amount):
        """Formula for increasing risk pool when below threshold"""
        increase = win_amount * risk_pool / (risk_pool + self.K_VALUE)
        return risk_pool + increase
    
    def run_simulation(self, num_trades, avg_gain, avg_loss, win_rate):
        """
        Run a simulation with the given parameters
        
        Parameters:
        - num_trades: number of trades to simulate
        - avg_gain: average gain per winning trade (as a percentage of risk)
        - avg_loss: average loss per losing trade (as a percentage of risk)
        - win_rate: probability of a trade being a win (0.0 to 1.0)
        """
        self.log(f"\n===== Starting Simulation =====")
        self.log(f"Trades: {num_trades} | Avg Gain: {avg_gain:.2f}% | Avg Loss: {avg_loss:.2f}% | Win Rate: {win_rate:.2f}")
        
        for trade_num in range(1, num_trades + 1):
            # Determine if this trade is a win based on the win rate
            is_win = random.random() < win_rate
            
            # Update the win rate tracking
            if len(self.last_8_trades) >= 8:
                self.last_8_trades.pop(0)  # Remove oldest trade
            self.last_8_trades.append(is_win)
            
            # Calculate current win rate
            wins = sum(self.last_8_trades)
            self.win_rate = wins / len(self.last_8_trades) if len(self.last_8_trades) > 0 else 0
            
            # Determine if we are trading well
            previous_trading_well = self.trading_well
            self.trading_well = self.win_rate >= 0.375  # At least 3/8 wins (37.5%)
            
            self.log(f"\n--- Trade {trade_num} ---")
            
            # Check if win rate just crossed the threshold
            if not previous_trading_well and self.trading_well:
                self.log(f"WIN RATE THRESHOLD CROSSED: Win rate is now {(self.win_rate * 100):.2f}%, trading improved!")
                # Set risk pool to 0.5% when crossing threshold only if below 0.5%
                if self.risk_pool < self.account_size * self.THRESHOLD_PCT:
                    old_risk_pool = self.risk_pool
                    self.risk_pool = self.account_size * self.THRESHOLD_PCT
                    self.log(f"RISK POOL ADJUSTED: Setting risk pool to 0.5% of account: {old_risk_pool:.4f} → {self.risk_pool:.4f}")
            elif previous_trading_well and not self.trading_well:
                self.log(f"WIN RATE DROPPED BELOW THRESHOLD: Win rate is now {(self.win_rate * 100):.2f}%, trading worse!")
            
            self.log(f"Win Rate: {(self.win_rate * 100):.2f}% ({wins}/{len(self.last_8_trades)})")
            
            # Calculate the amount of money risked on this trade (full risk pool)
            risk_amount = self.risk_pool
            self.log(f"Risk Amount: ${risk_amount:.2f}")
            
            # Process the trade result
            if is_win:
                # Calculate the return amount based on percentage of risked amount
                return_amount = risk_amount * (avg_gain / 100)
                self.log(f"Win Amount: ${return_amount:.2f}")
                self.log(f"Balance Before: ${self.account_size:.2f}")
                self.account_size += return_amount
                self.log(f"Balance After: ${self.account_size:.2f}")
                
                # Update threshold based on new account size
                new_threshold_amount = self.account_size * self.THRESHOLD_PCT
                
                # Update risk pool based on threshold
                old_risk_pool = self.risk_pool
                
                if self.risk_pool < new_threshold_amount:
                    # If below threshold, handle differently
                    potential_new_risk_pool = self.calculate_increased_risk_pool(old_risk_pool, return_amount)
                    
                    if potential_new_risk_pool < new_threshold_amount:
                        # Even applying the formula to the entire win wouldn't reach the threshold
                        self.risk_pool = potential_new_risk_pool
                        increased_amount = self.risk_pool - old_risk_pool
                        
                        formula_calculation = return_amount * old_risk_pool / (old_risk_pool + self.K_VALUE)
                        
                        self.log(f"Risk Pool Update: Formula used for entire win amount (${return_amount:.2f})")
                        self.log(f"Formula calculation: ${return_amount:.2f} * ${old_risk_pool:.2f} / (${old_risk_pool:.2f} + {self.K_VALUE}) = ${formula_calculation:.4f}")
                        self.log(f"Formula added: ${increased_amount:.4f} to pool")
                    else:
                        # Applying the formula to the entire win would exceed the threshold
                        # Find the amount that would reach the threshold exactly
                        amount_needed_for_threshold = (new_threshold_amount - old_risk_pool) * (old_risk_pool + self.K_VALUE) / old_risk_pool
                        
                        # Use formula for the threshold portion
                        risk_pool_at_threshold = self.calculate_increased_risk_pool(old_risk_pool, amount_needed_for_threshold)
                        increased_by_formula = risk_pool_at_threshold - old_risk_pool
                        
                        # Add the rest directly
                        remaining_win = return_amount - amount_needed_for_threshold
                        
                        self.log(f"Risk Pool Update: Formula used for ${amount_needed_for_threshold:.4f}, adding ${increased_by_formula:.4f}")
                        self.log(f"Formula calculation: ${amount_needed_for_threshold:.4f} * ${old_risk_pool:.2f} / (${old_risk_pool:.2f} + {self.K_VALUE}) = ${increased_by_formula:.4f}")
                        self.log(f"Full addition for remaining ${remaining_win:.2f}")
                        
                        # Apply both parts
                        self.risk_pool = risk_pool_at_threshold + remaining_win
                else:
                    # If already above threshold, add full amount
                    self.risk_pool += return_amount
                    self.log(f"Risk Pool Update: Full win amount ${return_amount:.2f} added")
                
                # Cap risk pool at maximum percentage
                max_risk_pool = self.account_size * self.MAX_RISK_POOL_PCT
                if self.risk_pool > max_risk_pool:
                    self.log(f"Risk Pool Capped: ${self.risk_pool:.2f} → ${max_risk_pool:.2f} (5% limit)")
                    self.risk_pool = max_risk_pool
            else:
                # Calculate the loss amount based on percentage of risked amount
                loss_amount = risk_amount * (avg_loss / 100)
                old_account_size = self.account_size
                self.account_size -= loss_amount
                
                self.log(f"Loss Amount: ${loss_amount:.2f}")
                self.log(f"Balance Before: ${old_account_size:.2f}")
                self.log(f"Balance After: ${self.account_size:.2f}")
                
                # Update threshold based on new account size
                new_threshold_amount = self.account_size * self.THRESHOLD_PCT
                
                # Store the old risk pool for logging
                old_risk_pool = self.risk_pool
                
                # Update risk pool based on threshold
                if self.risk_pool > new_threshold_amount:
                    # If above threshold, subtract full amount down to threshold
                    amount_above_threshold = self.risk_pool - new_threshold_amount
                    
                    if loss_amount <= amount_above_threshold:
                        # Can subtract full loss without going below threshold
                        self.risk_pool -= loss_amount
                        self.log(f"Risk Pool Update: Full loss of ${loss_amount:.2f} subtracted")
                    else:
                        # Need to reduce to threshold and then apply formula
                        reduced_by_direct = amount_above_threshold
                        excess_loss = loss_amount - amount_above_threshold
                        
                        # First reduce to threshold
                        self.risk_pool = new_threshold_amount
                        
                        # Then apply formula for remaining amount
                        before_formula = self.risk_pool
                        formula_calculation = excess_loss * before_formula / (before_formula + self.K_VALUE)
                        
                        self.risk_pool = self.calculate_reduced_risk_pool(self.risk_pool, excess_loss)
                        reduced_by_formula = before_formula - self.risk_pool
                        
                        self.log(f"Risk Pool Update: {reduced_by_direct:.4f} subtracted directly to reach threshold")
                        self.log(f"Formula calculation: {excess_loss:.2f} * {before_formula:.2f} / ({before_formula:.2f} + {self.K_VALUE}) = {formula_calculation:.4f}")
                        self.log(f"Formula used for remaining {excess_loss:.2f}, reducing by {reduced_by_formula:.4f}")
                else:
                    # Already below threshold, use formula for full loss
                    before_formula = self.risk_pool
                    self.risk_pool = self.calculate_reduced_risk_pool(self.risk_pool, loss_amount)
                    reduced_by = before_formula - self.risk_pool
                    
                    formula_calculation = loss_amount * before_formula / (before_formula + self.K_VALUE)
                    
                    self.log(f"Risk Pool Update: Formula used for entire {loss_amount:.2f} loss")
                    self.log(f"Formula calculation: {loss_amount:.2f} * {before_formula:.2f} / ({before_formula:.2f} + {self.K_VALUE}) = {formula_calculation:.4f}")
                    self.log(f"Formula reduced pool by {reduced_by:.4f}")
                
                self.log(f"Risk Pool Change: ${old_risk_pool:.2f} → ${self.risk_pool:.2f}")
            
            self.log(f"Current Risk Pool: ${self.risk_pool:.2f} ({(self.risk_pool/self.account_size*100):.2f}%)")
            
            # Record history
            self.balance_history.append(self.account_size)
            self.risk_pool_history.append(self.risk_pool)
            self.win_rate_history.append(self.win_rate)
        
        # Final summary
        self.log("\n===== Simulation Results =====")
        self.log(f"Initial Balance: ${self.initial_balance:.2f}")
        self.log(f"Final Balance: ${self.account_size:.2f}")
        self.log(f"Profit/Loss: ${(self.account_size - self.initial_balance):.2f} ({((self.account_size - self.initial_balance) / self.initial_balance * 100):.2f}%)")
        self.log(f"Final Risk Pool: ${self.risk_pool:.2f} ({(self.risk_pool/self.account_size*100):.2f}%)")
        self.log(f"Final Win Rate: {(self.win_rate * 100):.2f}%")
        
        return {
            'final_balance': self.account_size,
            'final_risk_pool': self.risk_pool,
            'balance_history': self.balance_history,
            'risk_pool_history': self.risk_pool_history,
            'win_rate_history': self.win_rate_history
        }
    
    def plot_results(self):
        """Plot the simulation results"""
        plt.figure(figsize=(15, 10))
        
        # Plot balance
        plt.subplot(3, 1, 1)
        plt.plot(self.balance_history, 'b-', linewidth=2)
        plt.title('Account Balance')
        plt.ylabel('USD')
        plt.grid(True)
        
        # Plot risk pool
        plt.subplot(3, 1, 2)
        plt.plot(self.risk_pool_history, 'r-', linewidth=2)
        plt.title('Risk Pool')
        plt.ylabel('USD')
        plt.grid(True)
        
        # Plot risk pool as percentage of account
        risk_pool_pct = [r/b*100 for r, b in zip(self.risk_pool_history, self.balance_history)]
        plt.subplot(3, 1, 3)
        plt.plot(risk_pool_pct, 'g-', linewidth=2)
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='0.5% Threshold')
        plt.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% Cap')
        plt.title('Risk Pool as % of Account')
        plt.ylabel('Percentage')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def run_manual_simulation():
    print("===== Risk Management Simulator =====")
    print("This simulator uses the risk management technique from the provided code.")
    print("Initial balance is set to $1000.\n")
    
    # Get user inputs
    num_trades = int(input("Enter number of trades to simulate: "))
    avg_gain = float(input("Enter average gain per winning trade (% of risk): "))
    avg_loss = float(input("Enter average loss per losing trade (% of risk): "))
    win_rate = float(input("Enter win rate (0.0 to 1.0): "))
    
    # Create simulator
    simulator = RiskManagementSimulator()
    
    # Run simulation
    result = simulator.run_simulation(num_trades, avg_gain, avg_loss, win_rate)
    
    # Plot results if requested
    plot_choice = input("\nDo you want to see a plot of the results? (y/n): ")
    if plot_choice.lower() == 'y':
        simulator.plot_results()

def run_multiple_scenarios():
    """Run multiple scenarios and compare results"""
    print("===== Multiple Scenario Analysis =====")
    print("This will run several scenarios with different parameters and compare results.")
    
    # Define scenarios
    scenarios = [
        {"name": "High Win Rate, Small Gains", "trades": 100, "avg_gain": 50, "avg_loss": 100, "win_rate": 0.7},
        {"name": "Low Win Rate, Big Gains", "trades": 100, "avg_gain": 200, "avg_loss": 100, "win_rate": 0.3},
        {"name": "Balanced Approach", "trades": 100, "avg_gain": 100, "avg_loss": 100, "win_rate": 0.5},
        {"name": "Risky Approach", "trades": 100, "avg_gain": 300, "avg_loss": 150, "win_rate": 0.35},
        {"name": "Conservative Approach", "trades": 100, "avg_gain": 80, "avg_loss": 50, "win_rate": 0.6}
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario['name']}")
        simulator = RiskManagementSimulator()
        result = simulator.run_simulation(
            scenario["trades"], 
            scenario["avg_gain"], 
            scenario["avg_loss"], 
            scenario["win_rate"]
        )
        results.append({
            "name": scenario["name"],
            "final_balance": result["final_balance"],
            "profit_pct": (result["final_balance"] - 1000) / 10,  # Percentage
            "final_risk_pool": result["final_risk_pool"]
        })
    
    # Print comparison table
    print("\n===== Scenario Comparison =====")
    print(f"{'Scenario':<25} {'Final Balance':<15} {'Profit %':<10} {'Final Risk Pool':<15}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<25} ${r['final_balance']:<14.2f} {r['profit_pct']:<9.2f}% ${r['final_risk_pool']:<14.2f}")

if __name__ == "__main__":
    print("===== Risk Management Simulation =====")
    choice = input("Choose simulation type:\n1. Manual parameters\n2. Multiple scenario analysis\nChoice (1/2): ")
    
    if choice == "1":
        run_manual_simulation()
    elif choice == "2":
        run_multiple_scenarios()
    else:
        print("Invalid choice. Exiting.")