import React, { useState, useEffect } from 'react';
import { tradeAPI } from '../services/tradeAPI';
import { balanceAPI } from '../services/balanceAPI';
import { Trade } from '@/TradeHistoryPage/types/Trade';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { DollarSign, Percent, Wallet, Edit, BarChart, ChevronDown, ChevronUp, Info } from "lucide-react";

interface RiskPoolStatsProps {
  summaryStats?: {
    winningPercentage: number;
    averageGain: number;
    averageLoss: number;
  };
}

const StatCard = ({
  label,
  value,
  icon: Icon,
  valueColor
}: {
  label: string;
  value: string;
  icon: React.ElementType;
  valueColor: string;
}) => (
  <Card className="bg-card border border-border">
    <CardContent className="py-1 px-3 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">{label}:</span>
        <span className={`font-semibold ${valueColor}`}>{value}</span>
      </div>
      <Icon className={`w-4 h-4 ${valueColor}`} />
    </CardContent>
  </Card>
);

export const RiskPoolStats: React.FC<RiskPoolStatsProps> = ({ summaryStats }) => {
  const [initialBalance, setInitialBalance] = useState<number>(1000);
  const [currentBalance, setCurrentBalance] = useState<number>(1000);
  const [riskPool, setRiskPool] = useState<number>(5);
  const [winRate, setWinRate] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [editingBalance, setEditingBalance] = useState<boolean>(false);
  const [tempBalance, setTempBalance] = useState<string>('1000');
  const [showSimulationDetails, setShowSimulationDetails] = useState<boolean>(false);

  const [showAlgoHelp, setShowAlgoHelp] = useState<boolean>(false);

  // New simulation parameters - defaults will be set from summaryStats if available
  const [accountRiskPercent, setAccountRiskPercent] = useState<number>(1.25); // % of account risk per trade
  const [gainPercent, setGainPercent] = useState<number>(6.44); // % gain on winning trades
  const [lossPercent, setLossPercent] = useState<number>(4.30); // % loss on losing trades (positive number)
  const [winningPercent, setWinningPercent] = useState<number>(26.92); // % of trades that are winners
  const [targetReturnPercent, setTargetReturnPercent] = useState<number>(100); // Target return %

  // Update simulation parameters when summaryStats changes
  useEffect(() => {
    if (summaryStats) {
      setWinningPercent(Number(summaryStats.winningPercentage.toFixed(2)));
      setGainPercent(Number(summaryStats.averageGain.toFixed(2)));
      setLossPercent(Number(Math.abs(summaryStats.averageLoss).toFixed(2))); // Make sure it's positive
    }
  }, [summaryStats]);

  // We intentionally run this once on mount to bootstrap stats from the backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [tradesResponse, balance] = await Promise.all([
          tradeAPI.getTrades(),
          balanceAPI.getBalance()
        ]);

        // Filter trades that have status "Exited"
        const fetchedTrades = tradesResponse.data;
        setInitialBalance(balance);
        setCurrentBalance(balance);
        setTempBalance('69');

        // Sort trades by entry date to ensure chronological processing
        const sortedTrades = fetchedTrades
          .filter(trade => trade.Entry_Date) // Only include trades with exit dates
          .sort((a, b) => new Date(a.Entry_Date!).getTime() - new Date(b.Entry_Date!).getTime());

        let accountSize = balance;
        let currentRiskPool = accountSize * 0.005; // Start at 0.5% of account
        const logs: string[] = [];

        // Initialize last 8 trades array for tracking win rate
        const last8Trades: boolean[] = []; // true for win, false for loss
        let currentWinRate = 0;
        let tradingWell = false;

        logs.push(`Initial Risk Pool: $${currentRiskPool.toFixed(2)} (${(currentRiskPool / accountSize * 100).toFixed(2)}%)`);

        // Change this line from fetchedTrades to sortedTrades:
        sortedTrades.forEach((trade: Trade, index: number) => {
          // Constants
          const thresholdPct = 0.005; // 0.5%
          const maxRiskPoolPct = 0.05; // 5%
          const reductionFactor = 0.25; // 25% reduction factor
          const increaseFactor = 0.25; // 25% increase factor

          const isWinningTrade = trade.Return !== null && trade.Return > 0;

          // Update last 8 trades history
          if (last8Trades.length >= 8) {
            last8Trades.shift(); // Remove oldest trade
          }
          last8Trades.push(isWinningTrade);

          // Calculate current win rate
          const wins = last8Trades.filter(win => win).length;
          currentWinRate = last8Trades.length > 0 ? wins / last8Trades.length : 0;



          logs.push(`\n--- Trade ${index + 1} (${trade.Ticker}) ---`);

          // Determine if we can take new positions
          const previousTradingWell = tradingWell;
          tradingWell = currentWinRate >= 0.375; // At least 3/8 wins (37.5%)

          // Check if win rate just crossed the threshold to allow trading
          if (!previousTradingWell && tradingWell) {
            logs.push(`WIN RATE THRESHOLD CROSSED: Win rate is now ${(currentWinRate * 100).toFixed(2)}%, trading improved!`);

            // Set risk pool to 0.5% when crossing the threshold only if the risk pool is below 0.5% of account
            if (currentRiskPool < accountSize * 0.005) {
              const oldRiskPool = currentRiskPool;
              currentRiskPool = accountSize * 0.005; // Set to 0.5% of account
              logs.push(`RISK POOL ADJUSTED: Setting risk pool to 0.5% of account: ${oldRiskPool.toFixed(4)} → ${currentRiskPool.toFixed(4)}`);
            }
          } else if (previousTradingWell && !tradingWell) {
            logs.push(`WIN RATE DROPPED BELOW THRESHOLD: Win rate is now ${(currentWinRate * 100).toFixed(2)}%, trading worse!`);
          }

          logs.push(`Win Rate: ${(currentWinRate * 100).toFixed(2)}% (${wins}/${last8Trades.length})`);

          // Formula for reducing risk pool when below threshold
          const calculateReducedRiskPool = (
            currentRiskPool: number,
            lossAmount: number
          ): number => {
            const lossProportion = Math.min(lossAmount / currentRiskPool, 1.0);
            const reduction = currentRiskPool * reductionFactor * lossProportion;
            return currentRiskPool - reduction;
          };


          // Formula for increasing risk pool when below threshold
          const calculateIncreasedRiskPool = (
            currentRiskPool: number,
            winAmount: number
          ): number => {
            const winProportion = Math.min(winAmount / currentRiskPool, 1.0);
            const increase = currentRiskPool * increaseFactor * winProportion;
            return currentRiskPool + increase;
          };

          // The corrected code portion for handling winning trades
          // Replace the existing win-handling logic with this code
          if (isWinningTrade && trade.Return !== null) {
            const winAmount = trade.Return;
            accountSize += winAmount;
            logs.push(`Win Amount: $${winAmount.toFixed(2)}`);

            // Update threshold based on new account size
            const newThresholdAmount = accountSize * thresholdPct;

            // Update risk pool based on threshold
            if (currentRiskPool < newThresholdAmount) {
              // If below threshold, we need to handle it differently
              const oldRiskPool = currentRiskPool;

              // Calculate how much the risk pool would increase if we applied the formula to the entire win amount
              const potentialNewRiskPool = calculateIncreasedRiskPool(oldRiskPool, winAmount);

              if (potentialNewRiskPool < newThresholdAmount) {
                // Even applying the formula to the entire win wouldn't reach the threshold
                // So apply formula to the entire win amount
                currentRiskPool = potentialNewRiskPool;
                const increasedAmount = currentRiskPool - oldRiskPool;

                const winProportion = Math.min(winAmount / oldRiskPool, 1.0);
                const formulaCalculation = oldRiskPool * increaseFactor * winProportion;

                logs.push(`Risk Pool Update: Formula used for entire win amount ($${winAmount.toFixed(2)})`);
                logs.push(`Formula calculation: $${oldRiskPool.toFixed(2)} * ${increaseFactor} * ${winProportion.toFixed(4)} = $${formulaCalculation.toFixed(4)}`);
                logs.push(`Formula added: $${increasedAmount.toFixed(4)} to pool`);
              } else {
                // Applying the formula to the entire win would exceed the threshold
                // Find the amount that would reach the threshold exactly when using the formula

                // We need to solve for x in: oldRiskPool + (oldRiskPool * increaseFactor * min(x/oldRiskPool, 1)) = threshold
                // If x >= oldRiskPool: oldRiskPool + (oldRiskPool * increaseFactor) = threshold
                // If x < oldRiskPool: oldRiskPool + (increaseFactor * x) = threshold

                const maxIncrease = oldRiskPool * increaseFactor;
                const neededIncrease = newThresholdAmount - oldRiskPool;

                let amountNeededForThreshold: number;
                if (neededIncrease <= maxIncrease) {
                  // We can reach threshold with partial application
                  amountNeededForThreshold = neededIncrease / increaseFactor;
                } else {
                  // Need full application plus direct addition
                  amountNeededForThreshold = oldRiskPool; // This will give us max increase
                }

                // Use formula for the threshold portion
                const riskPoolAtThreshold = calculateIncreasedRiskPool(oldRiskPool, amountNeededForThreshold);
                const increasedByFormula = riskPoolAtThreshold - oldRiskPool;

                // Add the rest directly
                const remainingWin = winAmount - amountNeededForThreshold;

                const winProportion = Math.min(amountNeededForThreshold / oldRiskPool, 1.0);

                logs.push(`Risk Pool Update: Formula used for $${amountNeededForThreshold.toFixed(4)}, adding $${increasedByFormula.toFixed(4)}`);
                logs.push(`Formula calculation: $${oldRiskPool.toFixed(2)} * ${increaseFactor} * ${winProportion.toFixed(4)} = $${increasedByFormula.toFixed(4)}`);
                logs.push(`Full addition for remaining $${remainingWin.toFixed(2)}`);

                // Apply both parts
                currentRiskPool = riskPoolAtThreshold + remainingWin;
              }
            } else {
              // If already above threshold, add full amount
              currentRiskPool += winAmount;
              logs.push(`Risk Pool Update: Full win amount $${winAmount.toFixed(2)} added`);
            }

            // Cap risk pool at maximum percentage
            const maxRiskPool = accountSize * maxRiskPoolPct;
            if (currentRiskPool > maxRiskPool) {
              logs.push(`Risk Pool Capped: $${currentRiskPool.toFixed(2)} → $${maxRiskPool.toFixed(2)} (5% limit)`);
              currentRiskPool = maxRiskPool;
            }
          } else if (!isWinningTrade && trade.Return !== null) {
            const lossAmount = Math.abs(trade.Return);
            accountSize -= lossAmount;
            logs.push(`Loss Amount: $${lossAmount.toFixed(2)}`);

            // Update threshold based on new account size
            const newThresholdAmount = accountSize * thresholdPct;

            // Store the old risk pool for logging
            const oldRiskPool = currentRiskPool;

            // Update risk pool based on threshold
            if (currentRiskPool > newThresholdAmount) {
              // If above threshold, subtract full amount down to threshold
              const amountAboveThreshold = currentRiskPool - newThresholdAmount;

              if (lossAmount <= amountAboveThreshold) {
                // Can subtract full loss without going below threshold
                currentRiskPool -= lossAmount;
                logs.push(`Risk Pool Update: Full loss of $${lossAmount.toFixed(2)} subtracted`);
              } else {
                // Need to reduce to threshold and then apply formula
                const reducedByDirect = amountAboveThreshold;
                const excessLoss = lossAmount - amountAboveThreshold;

                // First reduce to threshold
                currentRiskPool = newThresholdAmount;

                // Then apply formula for remaining amount
                const beforeFormula = currentRiskPool;
                const lossProportion = Math.min(excessLoss / beforeFormula, 1.0);
                const formulaCalculation = beforeFormula * reductionFactor * lossProportion;

                currentRiskPool = calculateReducedRiskPool(currentRiskPool, excessLoss);
                const reducedByFormula = beforeFormula - currentRiskPool;

                logs.push(`Risk Pool Update: ${reducedByDirect.toFixed(4)} subtracted directly to reach threshold`);
                logs.push(`Formula calculation: ${beforeFormula.toFixed(2)} * ${reductionFactor} * ${lossProportion.toFixed(4)} = ${formulaCalculation.toFixed(4)}`);
                logs.push(`Formula used for remaining ${excessLoss.toFixed(2)}, reducing by ${reducedByFormula.toFixed(4)}`);
              }
            } else {
              // Already below threshold, use formula for full loss
              const beforeFormula = currentRiskPool;
              currentRiskPool = calculateReducedRiskPool(currentRiskPool, lossAmount);
              const reducedBy = beforeFormula - currentRiskPool;

              const lossProportion = Math.min(lossAmount / beforeFormula, 1.0);
              const formulaCalculation = beforeFormula * reductionFactor * lossProportion;

              logs.push(`Risk Pool Update: Formula used for entire ${lossAmount.toFixed(2)} loss`);
              logs.push(`Formula calculation: ${beforeFormula.toFixed(2)} * ${reductionFactor} * ${lossProportion.toFixed(4)} = ${formulaCalculation.toFixed(4)}`);
              logs.push(`Formula reduced pool by ${reducedBy.toFixed(4)}`);
            }

            logs.push(`Risk Pool Change: $${oldRiskPool.toFixed(2)} → $${currentRiskPool.toFixed(2)}`);
          }

          logs.push(`Current Risk Pool: $${currentRiskPool.toFixed(2)}`);
        });

        setCurrentBalance(accountSize);
        setRiskPool(currentRiskPool);
        setWinRate(currentWinRate);

        // Logs removed as requested

      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const handleBalanceUpdate = async () => {
    const newBalance = parseFloat(tempBalance);
    if (!isNaN(newBalance) && newBalance >= 0) {
      try {
        await balanceAPI.updateBalance(newBalance);
        setInitialBalance(newBalance);
        setCurrentBalance(newBalance);
        setRiskPool(newBalance * 0.005); // Set to 0.5% of new balance
        setEditingBalance(false);
      } catch (error) {
        console.error('Error updating balance:', error);
      }
    }
  };

  const percentageChange = ((currentBalance - initialBalance) / initialBalance) * 100;

  if (loading) return <div>Loading...</div>;

  const stats = [
    {
      label: "Initial Balance",
      value: `Hidden`,
      icon: Wallet,
      valueColor: "text-blue-500"
    },
    {
      label: "Percentage Change",
      value: `${percentageChange.toFixed(2)}%`,
      icon: Percent,
      valueColor: percentageChange >= 0 ? "text-green-500" : "text-red-0"
    },
    {
      label: "Current Balance",
      value: `Hidden`,
      icon: DollarSign,
      valueColor: "text-green-500"
    },
    {
      label: "Current Recommended Risk",
      value: `$${riskPool.toFixed(2)}`,
      icon: DollarSign,
      valueColor: "text-purple-500"
    },
    {
      label: "Win Rate (Last 8)",
      value: `${(winRate * 100).toFixed(2)}%`,
      icon: BarChart,
      valueColor: winRate >= 0.375 ? "text-green-500" : "text-amber-500"
    }
  ];

  // Calculate trading simulation to target return
  const calculateSimulation = () => {
    const safeInitialBalance = Number.isFinite(initialBalance) ? initialBalance : 0;
    const safeTargetReturnPercent = Number.isFinite(targetReturnPercent) ? targetReturnPercent : 0;
    const safeAccountRiskPercent = Number.isFinite(accountRiskPercent) ? accountRiskPercent : 0;
    const safeGainPercent = Number.isFinite(gainPercent) ? gainPercent : 0;
    const safeLossPercent = Number.isFinite(lossPercent) ? lossPercent : 0;
    const boundedWinRate = Number.isFinite(winningPercent) ? Math.max(0, Math.min(100, winningPercent)) : 0;

    if (safeInitialBalance <= 0) {
      return { trades: [], tooManyTrades: false, blownUp: false, blowupAtTrade: 0, invalidReason: 'Initial balance must be greater than 0.' };
    }
    if (safeAccountRiskPercent <= 0) {
      return { trades: [], tooManyTrades: false, blownUp: false, blowupAtTrade: 0, invalidReason: 'Account Risk/Trade must be greater than 0.' };
    }
    if (safeLossPercent <= 0) {
      return { trades: [], tooManyTrades: false, blownUp: false, blowupAtTrade: 0, invalidReason: 'Loss/Trade must be greater than 0 to avoid undefined position size.' };
    }

    const targetReturn = safeInitialBalance * (safeTargetReturnPercent / 100);
    const targetBalance = safeInitialBalance + targetReturn;
    const blowupThreshold = safeInitialBalance * 0.5; // -50% = only 50% of initial balance left

    // Calculate number of wins and losses based on winning %
    const winRatio = boundedWinRate / 100;

    const seededRandom = (seed: number) => {
      const x = Math.sin(seed) * 10000;
      return x - Math.floor(x);
    };

    const runSimulation = (totalTrades: number, markTooManyTrades: boolean) => {
      const numWins = Math.round(totalTrades * winRatio);
      const numLosses = totalTrades - numWins;

      const tradeOutcomes: boolean[] = [];
      for (let i = 0; i < numWins; i++) tradeOutcomes.push(true);
      for (let i = 0; i < numLosses; i++) tradeOutcomes.push(false);

      for (let i = tradeOutcomes.length - 1; i > 0; i--) {
        const j = Math.floor(seededRandom(i + totalTrades) * (i + 1));
        [tradeOutcomes[i], tradeOutcomes[j]] = [tradeOutcomes[j], tradeOutcomes[i]];
      }

      let simAccountSize = safeInitialBalance;
      const trades: {
        tradeNumber: number;
        positionSize: number;
        riskAmount: number;
        returnAmount: number;
        newBalance: number;
        percentageGain: number;
        isWin: boolean;
      }[] = [];

      for (let i = 0; i < tradeOutcomes.length; i++) {
        const isWin = tradeOutcomes[i];
        const riskAmount = simAccountSize * (safeAccountRiskPercent / 100);
        const positionSize = (simAccountSize * safeAccountRiskPercent) / safeLossPercent;

        let returnAmount: number;
        if (isWin) {
          returnAmount = positionSize * (safeGainPercent / 100);
          simAccountSize += returnAmount;
        } else {
          returnAmount = -riskAmount;
          simAccountSize -= riskAmount;
        }

        trades.push({
          tradeNumber: i + 1,
          positionSize,
          riskAmount,
          returnAmount,
          newBalance: simAccountSize,
          percentageGain: ((simAccountSize - safeInitialBalance) / safeInitialBalance) * 100,
          isWin
        });

        if (simAccountSize <= blowupThreshold) {
          return { trades, tooManyTrades: markTooManyTrades, blownUp: true, blowupAtTrade: trades.length, invalidReason: null };
        }
      }

      const reachedTarget = simAccountSize >= targetBalance;
      return {
        trades,
        tooManyTrades: markTooManyTrades,
        blownUp: false,
        blowupAtTrade: 0,
        invalidReason: null,
        reachedTarget,
      };
    };

    // Simulate with increasing number of trades until we reach target or hit 1000
    for (let totalTrades = 1; totalTrades <= 1000; totalTrades++) {
      const result = runSimulation(totalTrades, false);
      if (result.blownUp) return result;
      if (result.reachedTarget) {
        return {
          trades: result.trades,
          tooManyTrades: false,
          blownUp: false,
          blowupAtTrade: 0,
          invalidReason: null,
        };
      }
    }

    // If we get here, we hit 1000 trades without reaching target
    const finalResult = runSimulation(1000, true);
    return {
      trades: finalResult.trades,
      tooManyTrades: true,
      blownUp: finalResult.blownUp,
      blowupAtTrade: finalResult.blowupAtTrade,
      invalidReason: null,
    };
  };

  const simulationResult = calculateSimulation();
  const simulatedTrades = simulationResult.trades;
  const tooManyTrades = simulationResult.tooManyTrades;
  const blownUp = simulationResult.blownUp;
  const blowupAtTrade = simulationResult.blowupAtTrade;
  const invalidSimulationReason = simulationResult.invalidReason;
  const finalGain = simulatedTrades.length > 0 ? simulatedTrades[simulatedTrades.length - 1].percentageGain : 0;

  const winsCount = simulatedTrades.filter(t => t.isWin).length;
  const lossesCount = simulatedTrades.length - winsCount;


  return (
    <>
      <Card className="mb-4">
        <CardHeader>
          <CardTitle>Current Recommended Risk</CardTitle>
          <p className="text-sm text-muted-foreground">
            The recommended risk algorithm reflects how much capital the system suggests putting on the line next.
            Strong recent results push the recommendation higher, while drawdowns dial it back to help
            protect the account. Good for beginners learning to use progressive exposure and keep the risk very low.
          </p>
          <div className="mt-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowAlgoHelp((v) => !v)}
              className="flex items-center gap-2"
            >
              <Info className="w-4 h-4" />
              <span>{showAlgoHelp ? "Hide how this is calculated" : "How this is calculated"}</span>
              {showAlgoHelp ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </Button>
            {showAlgoHelp && (
              <div className="mt-3 rounded-md border border-border bg-card p-3 text-sm leading-6">
                <div className="mb-2 font-medium">Simple explanation</div>
                <ul className="list-disc pl-5 space-y-1">
                  <li>We start from a safety baseline: 0.5% of your account per trade.</li>
                  <li>We look at your last 8 trades. If win rate goes above 37.5%, you’re "trading better."</li>
                  <li>When trading improves({'>'}37.5%), we raise the recommended risk up to that 0.5% baseline if it’s below it.</li>
                  <li>After each trade:
                    <ul className="list-disc pl-5 mt-1 space-y-1">
                      <li>If you win and the recommended risk is below the baseline, we grow it slowly if win rate is below 37.5% so you don't bump up your risk just after 1 good trade with low win rate, until it reaches the baseline. Above the baseline, wins add in full.</li>
                      <li>If you lose and the recommended risk is above the baseline, we first subtract losses fully until the baseline is hit; any extra loss reduces it slowly, so you don't end up with 0$ risk.</li>
                    </ul>
                  </li>
                  <li>The recommended risk never exceeds 5% of your account to avoid high risk of ruin.</li>
                </ul>
                <div className="mt-3 text-xs text-muted-foreground">
                  In practice: baseline ≈ 0.5% of account, max cap = 5%, below baseline subtraction = 25% of actual win/loss size.
                </div>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-1">
            <div className="grid grid-cols-2 gap-1">
              {stats.map((stat) => (
                <div key={stat.label}>
                  {stat.label === "Initial Balance" && editingBalance ? (
                    <Card className="bg-card">
                      <CardContent className="py-1 px-3 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-muted-foreground">{stat.label}:</span>
                          <Input
                            type="number"
                            value={tempBalance}
                            onChange={(e) => setTempBalance(e.target.value)}
                            className="w-32 h-8 text-sm"
                          />
                        </div>
                        <div className="flex gap-1">
                          <Button size="sm" onClick={handleBalanceUpdate}>Save</Button>
                          <Button size="sm" variant="outline" onClick={() => setEditingBalance(false)}>Cancel</Button>
                        </div>
                      </CardContent>
                    </Card>
                  ) : (
                    <div className="relative">
                      <StatCard {...stat} />
                      {stat.label === "Initial Balance" && !editingBalance && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="absolute right-8 top-1 h-6 w-6 p-0"
                          onClick={() => setEditingBalance(true)}
                        >
                          <Edit className="w-4 h-4 text-gray-500" />
                        </Button>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="mb-4">
        <CardHeader>
          <div className="text-center">
            <CardTitle>Trading Simulation to {targetReturnPercent}% Return</CardTitle>
            <div className="text-sm text-muted-foreground">
              Simulating trades with {winningPercent.toFixed(1)}% win rate, {gainPercent.toFixed(1)}% gain on wins, {lossPercent.toFixed(1)}% loss on losses and {accountRiskPercent.toFixed(2)}% account risk per trade.
              {simulatedTrades.length > 0 && (
                <div className="mt-1 text-xs font-medium">
                  <span className="text-green-600">Wins: {winsCount}</span> | <span className="text-red-500">Losses: {lossesCount}</span>
                </div>
              )}
              {invalidSimulationReason ? (
                <>
                  <br />
                  <span className="font-semibold text-amber-500">
                    ⚠️ {invalidSimulationReason}
                  </span>
                </>
              ) : blownUp ? (
                <>
                  <br />
                  <span className="font-semibold text-amber-500">
                    With these stats, you blow up your account in {blowupAtTrade} trades 💥 (-50% reached).
                  </span>
                </>
              ) : tooManyTrades ? (
                <>
                  <br />
                  <span className="font-semibold text-amber-500">
                    ⚠️ Too many trades needed! After 1000 trades, only reached {finalGain.toFixed(1)}% return.
                  </span>
                </>
              ) : simulatedTrades.length > 0 && (
                <>
                  <br />
                  <span className="font-semibold text-green-600">
                    ✅ {simulatedTrades.length} trades needed to reach {finalGain.toFixed(1)}% return
                  </span>
                </>
              )}
            </div>
          </div>
          <div className="flex flex-wrap items-center justify-start gap-4 mt-4">
            <div className="flex items-center gap-1">
              <span className="text-sm text-muted-foreground">Target Return:</span>
              <Input
                type="number"
                value={targetReturnPercent}
                onChange={(e) => setTargetReturnPercent(parseFloat(e.target.value) || 100)}
                className="w-24 h-8 text-sm"
                step="10"
                min="10"
                max="1000"
              />
              <span className="text-sm text-muted-foreground">%</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-sm text-muted-foreground">Account Risk/Trade:</span>
              <Input
                type="number"
                value={accountRiskPercent}
                onChange={(e) => setAccountRiskPercent(parseFloat(e.target.value) || 1)}
                className="w-24 h-8 text-sm"
                step="0.1"
                min="0.1"
                max="100"
              />
              <span className="text-sm text-muted-foreground">%</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-sm text-muted-foreground">Gain/Trade:</span>
              <Input
                type="number"
                value={gainPercent}
                onChange={(e) => setGainPercent(parseFloat(e.target.value) || 3)}
                className="w-24 h-8 text-sm"
                step="0.1"
                min="0.1"
                max="100"
              />
              <span className="text-sm text-muted-foreground">%</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-sm text-muted-foreground">Loss/Trade:</span>
              <Input
                type="number"
                value={lossPercent}
                onChange={(e) => {
                  const parsed = parseFloat(e.target.value);
                  setLossPercent(Number.isNaN(parsed) ? 0 : parsed);
                }}
                className="w-24 h-8 text-sm"
                step="0.1"
                min="0"
                max="100"
              />
              <span className="text-sm text-muted-foreground">%</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-sm text-muted-foreground">Win Rate:</span>
              <Input
                type="number"
                value={winningPercent}
                onChange={(e) => setWinningPercent(parseFloat(e.target.value) || 50)}
                className="w-24 h-8 text-sm"
                step="1"
                min="0"
                max="100"
              />
              <span className="text-sm text-muted-foreground">%</span>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowSimulationDetails(!showSimulationDetails)}
            >
              {showSimulationDetails ? "Hide Details" : "Show Details"}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {/* Table display for trades */}
          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-card border-b border-border">
                <tr>
                  <th className="px-2 py-2 text-left font-medium text-muted-foreground">#</th>
                  <th className="px-2 py-2 text-left font-medium text-muted-foreground">Result</th>
                  {showSimulationDetails && (
                    <>
                      <th className="px-2 py-2 text-right font-medium text-muted-foreground">Position Size</th>
                      <th className="px-2 py-2 text-right font-medium text-muted-foreground">Risk $</th>
                      <th className="px-2 py-2 text-right font-medium text-muted-foreground">Return $</th>
                      <th className="px-2 py-2 text-right font-medium text-muted-foreground">Balance</th>
                    </>
                  )}
                  <th className="px-2 py-2 text-right font-medium text-muted-foreground">Total Gain</th>
                </tr>
              </thead>
              <tbody>
                {invalidSimulationReason && (
                  <tr>
                    <td className="px-2 py-3 text-amber-500" colSpan={showSimulationDetails ? 7 : 3}>
                      {invalidSimulationReason}
                    </td>
                  </tr>
                )}
                {simulatedTrades.map((trade) => (
                  <tr
                    key={trade.tradeNumber}
                    className={`border-b border-border/50 ${trade.isWin
                      ? "bg-green-50/10 dark:bg-green-950/10"
                      : "bg-red-50/10 dark:bg-red-950/10"
                      }`}
                  >
                    <td className="px-2 py-1.5 font-medium">{trade.tradeNumber}</td>
                    <td className={`px-2 py-1.5 font-semibold ${trade.isWin ? "text-green-600" : "text-red-500"}`}>
                      {trade.isWin ? "WIN" : "LOSS"}
                    </td>
                    {showSimulationDetails && (
                      <>
                        <td className="px-2 py-1.5 text-right">${trade.positionSize.toFixed(2)}</td>
                        <td className="px-2 py-1.5 text-right">${trade.riskAmount.toFixed(2)}</td>
                        <td className={`px-2 py-1.5 text-right font-medium ${trade.isWin ? "text-green-600" : "text-red-500"}`}>
                          {trade.returnAmount >= 0 ? "+" : ""}${trade.returnAmount.toFixed(2)}
                        </td>
                        <td className="px-2 py-1.5 text-right">${trade.newBalance.toFixed(2)}</td>
                      </>
                    )}
                    <td className={`px-2 py-1.5 text-right font-medium ${trade.percentageGain >= 0 ? "text-green-600" : "text-red-500"}`}>
                      {trade.percentageGain >= 0 ? "+" : ""}{trade.percentageGain.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </>
  );
};