import random
import argparse
import matplotlib.pyplot as plt
import sys

class RiskManagementSimulator:
    def __init__(self, initial_balance=1000, threshold_pct=0.005, max_risk_pool_pct=0.05):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.threshold_pct = threshold_pct  # 0.5%
        self.max_risk_pool_pct = max_risk_pool_pct  # 5%
        self.risk_pool = initial_balance * threshold_pct  # Start at 0.5% of account
        self.last_8_trades = []  # true for win, false for loss
        self.win_rate = 0
        self.trading_well = False
        self.logs = []
        self.trade_data = []  # Store data for plotting
        
    def log(self, message):
        print(message)
        self.logs.append(message)
        
    def calculate_increased_risk_pool(self, current_risk_pool, win_amount):
        # Symmetric formula to the loss formula
        increase_factor = 0.2  # Same factor for consistency
        
        # Safe division to avoid errors
        if current_risk_pool <= 0:
            return current_risk_pool + win_amount  # Just add the whole win if risk pool is 0
        
        # Calculate the proportion of the win compared to the risk pool
        win_proportion = min(win_amount / current_risk_pool, 1.0)
        
        # Calculate increase amount (scales with win size, capped at increase_factor%)
        increase = current_risk_pool * increase_factor * win_proportion
        
        return current_risk_pool + increase


    def calculate_reduced_risk_pool(self, current_risk_pool, loss_amount):
        # This creates a decay that will reduce by approximately half after 5 full losses
        reduction_factor = 0.2  # Approximately 20% reduction per full risk pool loss
        
        # Safe division to avoid errors
        if current_risk_pool <= 0:
            return current_risk_pool
        
        # Calculate the proportion of the risk pool that was lost
        loss_proportion = min(loss_amount / current_risk_pool, 1.0)
        
        # Calculate reduction amount (maximum reduction of reduction_factor% of current risk pool)
        reduction = current_risk_pool * reduction_factor * loss_proportion
        
        return current_risk_pool - reduction
    
    def process_trade(self, is_win, amount, trade_num):
        """Process a single trade and update all values"""
        # Update last 8 trades history
        if len(self.last_8_trades) >= 8:
            self.last_8_trades.pop(0)  # Remove oldest trade
        self.last_8_trades.append(is_win)
        
        # Calculate current win rate
        wins = self.last_8_trades.count(True)
        self.win_rate = wins / len(self.last_8_trades) if self.last_8_trades else 0
        
        # Determine if we can take new positions
        previous_trading_well = self.trading_well
        self.trading_well = self.win_rate >= 0.375  # At least 3/8 wins (37.5%)
        
        # Check if win rate just crossed the threshold
        if not previous_trading_well and self.trading_well:
            self.log(f"WIN RATE THRESHOLD CROSSED: Win rate is now {self.win_rate * 100:.2f}%, trading improved!")
            
            # Set risk pool to 0.5% when crossing the threshold only if the risk pool is below 0.5% of account
            if self.risk_pool < self.current_balance * self.threshold_pct:
                old_risk_pool = self.risk_pool
                self.risk_pool = self.current_balance * self.threshold_pct  # Set to 0.5% of account
                self.log(f"RISK POOL ADJUSTED: Setting risk pool to 0.5% of account: {old_risk_pool:.4f} → {self.risk_pool:.4f}")
        elif previous_trading_well and not self.trading_well:
            self.log(f"WIN RATE DROPPED BELOW THRESHOLD: Win rate is now {self.win_rate * 100:.2f}%, trading worse!")

        self.log(f"Win Rate: {self.win_rate * 100:.2f}% ({wins}/{len(self.last_8_trades)})")
        
        if is_win:
            # Handle winning trade
            win_amount = amount
            old_balance = self.current_balance
            self.current_balance += win_amount
            self.log(f"Win Amount: ${win_amount:.2f}")
            self.log(f"Balance Before: ${old_balance:.2f}")
            self.log(f"Balance After: ${self.current_balance:.2f}")
            
            # Update threshold based on new account size
            new_threshold_amount = self.current_balance * self.threshold_pct
            old_risk_pool = self.risk_pool
            
            # Update risk pool based on threshold
            if self.risk_pool < new_threshold_amount:
                # If below threshold, we need to handle it differently
                # Calculate how much the risk pool would increase if we applied the formula to the entire win amount
                potential_new_risk_pool = self.calculate_increased_risk_pool(old_risk_pool, win_amount)
                
                if potential_new_risk_pool < new_threshold_amount:
                    # Even applying the formula to the entire win wouldn't reach the threshold
                    # So apply formula to the entire win amount
                    self.risk_pool = potential_new_risk_pool
                    increased_amount = self.risk_pool - old_risk_pool
                    
                    k = 20  # K value used in formula
                    formula_calculation = win_amount * old_risk_pool / (old_risk_pool + k)
                    
                    self.log(f"Risk Pool Update: Formula used for entire win amount (${win_amount:.2f})")
                    self.log(f"Formula calculation: ${win_amount:.2f} * ${old_risk_pool:.2f} / (${old_risk_pool:.2f} + {k}) = ${formula_calculation:.4f}")
                    self.log(f"Formula added: ${increased_amount:.4f} to pool")
                else:
                    # Applying the formula to the entire win would exceed the threshold
                    # Find the amount that would reach the threshold exactly when using the formula
                    k = 20
                    amount_needed_for_threshold = (new_threshold_amount - old_risk_pool) * (old_risk_pool + k) / old_risk_pool
                    
                    # Use formula for the threshold portion
                    risk_pool_at_threshold = self.calculate_increased_risk_pool(old_risk_pool, amount_needed_for_threshold)
                    increased_by_formula = risk_pool_at_threshold - old_risk_pool
                    
                    # Add the rest directly
                    remaining_win = win_amount - amount_needed_for_threshold
                    
                    self.log(f"Risk Pool Update: Formula used for ${amount_needed_for_threshold:.4f}, adding ${increased_by_formula:.4f}")
                    self.log(f"Formula calculation: ${amount_needed_for_threshold:.4f} * ${old_risk_pool:.2f} / (${old_risk_pool:.2f} + {k}) = ${increased_by_formula:.4f}")
                    self.log(f"Full addition for remaining ${remaining_win:.2f}")
                    
                    # Apply both parts
                    self.risk_pool = risk_pool_at_threshold + remaining_win
            else:
                # If already above threshold, add full amount
                self.risk_pool += win_amount
                self.log(f"Risk Pool Update: Full win amount ${win_amount:.2f} added")
            
            # Cap risk pool at maximum percentage
            max_risk_pool = self.current_balance * self.max_risk_pool_pct
            if self.risk_pool > max_risk_pool:
                self.log(f"Risk Pool Capped: ${self.risk_pool:.2f} → ${max_risk_pool:.2f} (5% limit)")
                self.risk_pool = max_risk_pool
            
            self.log(f"Risk Pool Change: ${old_risk_pool:.2f} → ${self.risk_pool:.2f}")
        else:
            # Handle losing trade
            loss_amount = amount
            old_balance = self.current_balance
            self.current_balance -= loss_amount
            self.log(f"Loss Amount: ${loss_amount:.2f}")
            self.log(f"Balance Before: ${old_balance:.2f}")
            self.log(f"Balance After: ${self.current_balance:.2f}")
            
            # Update threshold based on new account size
            new_threshold_amount = self.current_balance * self.threshold_pct
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
                    k = 20  # K value used in formula
                    formula_calculation = excess_loss * before_formula / (before_formula + k)
                    
                    self.risk_pool = self.calculate_reduced_risk_pool(self.risk_pool, excess_loss)
                    reduced_by_formula = before_formula - self.risk_pool
                    
                    self.log(f"Risk Pool Update: {reduced_by_direct:.4f} subtracted directly to reach threshold")
                    self.log(f"Formula calculation: {excess_loss:.2f} * {before_formula:.2f} / ({before_formula:.2f} + {k}) = {formula_calculation:.4f}")
                    self.log(f"Formula used for remaining {excess_loss:.2f}, reducing by {reduced_by_formula:.4f}")
            else:
                # Already below threshold, use formula for full loss
                before_formula = self.risk_pool
                self.risk_pool = self.calculate_reduced_risk_pool(self.risk_pool, loss_amount)
                reduced_by = before_formula - self.risk_pool
                
                k = 20  # K value used in formula
                formula_calculation = loss_amount * before_formula / (before_formula + k)
                
                self.log(f"Risk Pool Update: Formula used for entire {loss_amount:.2f} loss")
                self.log(f"Formula calculation: {loss_amount:.2f} * {before_formula:.2f} / ({before_formula:.2f} + {k}) = {formula_calculation:.4f}")
                self.log(f"Formula reduced pool by {reduced_by:.4f}")
            
            self.log(f"Risk Pool Change: ${old_risk_pool:.2f} → ${self.risk_pool:.2f}")
        
        self.log(f"Current Risk Pool: ${self.risk_pool:.2f} ({self.risk_pool/self.current_balance*100:.2f}%)")
        self.log("---")
        
        # Store data for plotting
        self.trade_data.append({
            'trade_num': trade_num,
            'is_win': is_win,
            'amount': amount,
            'balance': self.current_balance,
            'risk_pool': self.risk_pool,
            'risk_percent': self.risk_pool/self.current_balance*100,
            'win_rate': self.win_rate * 100
        })
        
        return self.risk_pool, self.current_balance
    
    def run_simulation(self, num_trades, win_rate, avg_gain, avg_loss, gain_stdev=0, loss_stdev=0):
        """Run a full simulation with the given parameters"""
        self.log(f"\n=== SIMULATION PARAMETERS ===")
        self.log(f"Number of trades: {num_trades}")
        self.log(f"Win rate: {win_rate*100:.1f}%")
        self.log(f"Average gain: ${avg_gain:.2f}x risk (stdev: ${gain_stdev:.2f})")
        self.log(f"Average loss: ${avg_loss:.2f}x risk (stdev: ${loss_stdev:.2f})")
        self.log(f"Initial balance: ${self.initial_balance:.2f}")
        self.log(f"Initial risk pool: ${self.risk_pool:.2f} ({self.risk_pool/self.initial_balance*100:.2f}%)")
        self.log(f"===========================\n")
        
        # Clear previous data
        self.trade_data = []
        
        wins = 0
        losses = 0
        
        for i in range(num_trades):
            # Determine if this trade is a win based on win rate
            is_win = random.random() < win_rate
            
            if is_win:
                wins += 1
            else:
                losses += 1
            
            # Determine the amount based on the risk pool
            # The trade risk is the full risk pool
            risk_amount = self.risk_pool
            
            # Calculate the actual gain/loss with some randomness
            if is_win:
                # For a win, the gain is the risk_amount * avg_gain factor with some randomness
                if gain_stdev > 0:
                    gain_factor = max(0.1, random.normalvariate(avg_gain, gain_stdev))
                else:
                    gain_factor = avg_gain
                amount = risk_amount * gain_factor
            else:
                # For a loss, the loss is the risk_amount itself (sometimes less)
                if loss_stdev > 0:
                    loss_factor = max(0.1, random.normalvariate(avg_loss, loss_stdev))
                else:
                    loss_factor = avg_loss
                amount = risk_amount * loss_factor
            
            self.log(f"\n--- Trade {i+1} ---")
            self.log(f"Trade {'WIN' if is_win else 'LOSS'} (Risk: ${risk_amount:.2f})")
            if is_win:
                self.log(f"Gain factor: {gain_factor:.2f}x")
            else:
                self.log(f"Loss factor: {loss_factor:.2f}x")
            
            self.process_trade(is_win, amount, i+1)
        
        # Print summary statistics
        self.log("\n=== SIMULATION SUMMARY ===")
        self.log(f"Starting balance: ${self.initial_balance:.2f}")
        self.log(f"Final balance: ${self.current_balance:.2f}")
        self.log(f"Profit/Loss: ${self.current_balance - self.initial_balance:.2f} ({(self.current_balance/self.initial_balance - 1)*100:.2f}%)")
        self.log(f"Final risk pool: ${self.risk_pool:.2f} ({self.risk_pool/self.current_balance*100:.2f}%)")
        self.log(f"Final win rate (last 8): {self.win_rate*100:.2f}%")
        
        self.log(f"Total wins: {wins} ({wins/num_trades*100:.2f}%)")
        self.log(f"Total losses: {losses} ({losses/num_trades*100:.2f}%)")
        
        if wins > 0:
            avg_win_amount = sum(t['amount'] for t in self.trade_data if t['is_win']) / wins
            self.log(f"Average win amount: ${avg_win_amount:.2f}")
        
        if losses > 0:
            avg_loss_amount = sum(t['amount'] for t in self.trade_data if not t['is_win']) / losses
            self.log(f"Average loss amount: ${avg_loss_amount:.2f}")
        
        return self.trade_data

    def create_plots(self):
        """Create plots from simulation data"""
        if not self.trade_data:
            print("No data to plot. Run a simulation first.")
            return
        
        # Extract data for plotting
        trade_nums = [t['trade_num'] for t in self.trade_data]
        balances = [t['balance'] for t in self.trade_data]
        risk_pools = [t['risk_pool'] for t in self.trade_data]
        risk_percents = [t['risk_percent'] for t in self.trade_data]
        win_rates = [t['win_rate'] for t in self.trade_data]
        
        # Create a figure with 4 subplots
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        fig.suptitle('Risk Management Simulation Results', fontsize=16)
        
        # Add baseline
        axs[0].axhline(y=self.initial_balance, color='gray', linestyle='--', alpha=0.5, label='Initial Balance')
        
        # Plot account balance
        axs[0].plot(trade_nums, balances, linestyle='-', color='blue')
        axs[0].set_ylabel('Account Balance ($)')
        axs[0].set_title('Account Balance Over Time')
        axs[0].grid(True, alpha=0.3)
        
        # Plot risk pool
        axs[1].plot(trade_nums, risk_pools, linestyle='-', color='purple')
        axs[1].set_ylabel('Risk Pool Size ($)')
        axs[1].set_title('Risk Pool Size Over Time')
        axs[1].grid(True, alpha=0.3)
        
        # Plot risk percent
        axs[2].plot(trade_nums, risk_percents, linestyle='-', color='green')
        axs[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5% Threshold')
        axs[2].axhline(y=5.0, color='gray', linestyle='--', alpha=0.5, label='5% Max')
        axs[2].set_ylabel('Risk Percentage (%)')
        axs[2].set_title('Risk Percentage Over Time')
        axs[2].grid(True, alpha=0.3)
        axs[2].legend()
        
        # Plot win rate
        axs[3].plot(trade_nums, win_rates, linestyle='-', color='orange')
        axs[3].axhline(y=37.5, color='red', linestyle='--', alpha=0.5, label='37.5% Threshold')
        axs[3].set_xlabel('Trade Number')
        axs[3].set_ylabel('Win Rate (%)')
        axs[3].set_title('Win Rate Over Time (8-Trade Window)')
        axs[3].grid(True, alpha=0.3)
        axs[3].legend()
        
        # Add win/loss markers
        # for i, t in enumerate(self.trade_data):
        #     if t['is_win']:
        #         axs[0].plot(t['trade_num'], t['balance'], 'go', markersize=8, alpha=0.5)
        #     else:
        #         axs[0].plot(t['trade_num'], t['balance'], 'ro', markersize=8, alpha=0.5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig('risk_management_results.png')
        plt.show()

def get_user_input():
    """Get simulation parameters from user input"""
    print("\n=== Risk Management Simulation ===")
    
    # Get number of trades
    while True:
        try:
            num_trades = int(input("Enter number of trades: "))
            if 1 <= num_trades <= 9999999:
                break
            else:
                print("Please enter a valid number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get win rate
    while True:
        try:
            win_rate = float(input("Enter win rate (0.1-0.9): "))
            if 0.1 <= win_rate <= 0.9:
                break
            else:
                print("Please enter a value between 0.1 and 0.9.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get average gain
    while True:
        try:
            avg_gain = float(input("Enter average gain factor (e.g., 2.0 means 2x risk): "))
            if avg_gain > 0:
                break
            else:
                print("Please enter a positive value.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get average loss
    while True:
        try:
            avg_loss = float(input("Enter average loss factor (e.g., 1.0 means full risk amount): "))
            if avg_loss > 0:
                break
            else:
                print("Please enter a positive value.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get gain standard deviation (optional)
    try:
        gain_stdev = float(input("Enter gain standard deviation (0 for no randomness): "))
        if gain_stdev < 0:
            gain_stdev = 0
    except ValueError:
        gain_stdev = 0
    
    # Get loss standard deviation (optional)
    try:
        loss_stdev = float(input("Enter loss standard deviation (0 for no randomness): "))
        if loss_stdev < 0:
            loss_stdev = 0
    except ValueError:
        loss_stdev = 0
    
    # Get initial balance (optional)
    try:
        initial_balance = float(input("Enter initial balance (default=1000): "))
        if initial_balance <= 0:
            initial_balance = 1000
    except ValueError:
        initial_balance = 1000
    
    return {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_gain': avg_gain,
        'avg_loss': avg_loss,
        'gain_stdev': gain_stdev,
        'loss_stdev': loss_stdev,
        'initial_balance': initial_balance
    }

def main():
    parser = argparse.ArgumentParser(description='Risk Management Simulation')
    parser.add_argument('--trades', type=int, help='Number of trades to simulate')
    parser.add_argument('--win-rate', type=float, help='Win rate as a decimal (e.g., 0.6 for 60%%)')
    parser.add_argument('--avg-gain', type=float, help='Average gain factor (e.g., 2.0 means risk*2)')
    parser.add_argument('--avg-loss', type=float, help='Average loss factor (e.g., 1.0 means full risk amount)')
    parser.add_argument('--gain-stdev', type=float, default=0.2, help='Standard deviation for gain randomness')
    parser.add_argument('--loss-stdev', type=float, default=0.1, help='Standard deviation for loss randomness')
    parser.add_argument('--initial-balance', type=float, default=1000, help='Initial account balance')
    parser.add_argument('--no-plots', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    # Check if required args are provided, if not, ask for input
    if None in [args.trades, args.win_rate, args.avg_gain, args.avg_loss]:
        user_params = get_user_input()
        num_trades = user_params['num_trades']
        win_rate = user_params['win_rate']
        avg_gain = user_params['avg_gain']
        avg_loss = user_params['avg_loss']
        gain_stdev = user_params['gain_stdev']
        loss_stdev = user_params['loss_stdev']
        initial_balance = user_params['initial_balance']
    else:
        num_trades = args.trades
        win_rate = args.win_rate
        avg_gain = args.avg_gain
        avg_loss = args.avg_loss
        gain_stdev = args.gain_stdev
        loss_stdev = args.loss_stdev
        initial_balance = args.initial_balance
    
    # Create simulator
    simulator = RiskManagementSimulator(initial_balance=initial_balance)
    
    # Run simulation
    simulator.run_simulation(
        num_trades=num_trades,
        win_rate=win_rate,
        avg_gain=avg_gain,
        avg_loss=avg_loss,
        gain_stdev=gain_stdev,
        loss_stdev=loss_stdev
    )
    
    # Create plots if not disabled
    if not args.no_plots:
        try:
            simulator.create_plots()
        except Exception as e:
            print(f"Error creating plots: {e}")
            print("Make sure matplotlib is installed: pip install matplotlib")

if __name__ == "__main__":
    main()