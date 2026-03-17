"""
Coin Flip Simulation - Finding the longest streak
Uses secrets module for cryptographically secure randomness
"""
import secrets

def simulate_coin_flips(num_flips=100000):
    longest_streak = 0
    current_streak = 1
    longest_streak_value = None
    
    # First flip
    previous = secrets.randbelow(2)  # 0 = Tails, 1 = Heads
    
    for _ in range(1, num_flips):
        current = secrets.randbelow(2)
        
        if current == previous:
            current_streak += 1
            if current_streak > longest_streak:
                longest_streak = current_streak
                longest_streak_value = current
        else:
            current_streak = 1
        
        previous = current
    
    return longest_streak, "Heads" if longest_streak_value == 1 else "Tails"

if __name__ == "__main__":
    print("🪙 Simulating 100,000 coin flips with true randomness...\n")
    
    # Run multiple simulations to see variation
    num_simulations = 10
    results = []
    
    for i in range(num_simulations):
        streak, side = simulate_coin_flips(100000)
        results.append((streak, side))
        print(f"Simulation {i+1}: Longest streak = {streak} {side} in a row")
    
    print("\n" + "="*50)
    max_streak = max(results, key=lambda x: x[0])
    avg_streak = sum(r[0] for r in results) / len(results)
    
    print(f"📊 Results from {num_simulations} simulations of 100,000 flips each:")
    print(f"   Best streak:    {max_streak[0]} {max_streak[1]} in a row")
    print(f"   Average streak: {avg_streak:.1f}")
    print(f"\n💡 Fun fact: With 100,000 flips, you'd statistically expect")
    print(f"   a streak of around 16-17 in a row (log₂(100000) ≈ 16.6)")
