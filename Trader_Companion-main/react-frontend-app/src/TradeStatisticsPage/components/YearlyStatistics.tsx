// components/YearlyStatistics.tsx
import React from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { TrendingUp, TrendingDown, Percent, Scale } from "lucide-react";
// import { DollarSign } from "lucide-react";
interface YearlyStatisticsProps {
  yearlyStats: {
    winningPercentage: number;
    averageGain: number;
    averageLoss: number;
    winLossRatio: number;
    expectedValuePerTrade: number;
    expectedReturnOn10Trades_125?: number;
    expectedReturnOn50Trades_125?: number;
    expectedReturnOn10Trades_25?: number;
    expectedReturnOn50Trades_25?: number;
    avgLargestGain: number;
    avgLargestLoss: number;
    avgLargestGainLossRatio: number;
    avgDaysGains: number;
    avgDaysLoss: number;
    avgDaysRatio: number;
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
  <Card className="bg-card">
    <CardContent className="py-1 px-3 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">{label}:</span>
        <span className={`font-semibold ${valueColor}`}>{value}</span>
      </div>
      <Icon className={`w-4 h-4 ${valueColor}`} />
    </CardContent>
  </Card>
);

export const YearlyStatistics: React.FC<YearlyStatisticsProps> = ({ yearlyStats }) => {
  const mainStats = [
    {
      label: "Win Rate",
      value: `${yearlyStats.winningPercentage.toFixed(2)}%`,
      icon: Percent,
      valueColor: "text-blue-500"
    },
    {
      label: "Avg Gain",
      value: `${yearlyStats.averageGain.toFixed(2)}%`,
      icon: TrendingUp,
      valueColor: "text-green-500"
    },
    {
      label: "Avg Loss",
      value: `${yearlyStats.averageLoss.toFixed(2)}%`,
      icon: TrendingDown,
      valueColor: "text-purple-500"
    },
    {
      label: "Avg Gain / Avg Loss",
      value: yearlyStats.winLossRatio.toFixed(2),
      icon: Scale,
      valueColor: "text-yellow-500"
    }
    ,
    {
      label: "Avg Lg Gain / Avg Lg Loss", // New stat
      value: `${yearlyStats.avgLargestGainLossRatio.toFixed(2)}`,
      icon: Scale,
      valueColor: "text-purple-500"
    },
    {
      label: "Avg Days Gains / Avg Days Loss",
      value: yearlyStats.avgDaysRatio.toFixed(1),
      icon: TrendingUp, // You can choose a different icon if you prefer
      valueColor: "text-orange-400"
    }
  ];
  
  // const returnStats = [
  //   {
  //     label: "Expected Return (12.5% Position sizing, 10 Trades)",
  //     value: `${(yearlyStats.expectedReturnOn10Trades_125 || 0).toFixed(2)}%`,
  //     icon: DollarSign,
  //     valueColor: "text-purple-500"
  //   },
  //   {
  //     label: "Expected Return (12.5% Position sizing, 50 Trades)",
  //     value: `${(yearlyStats.expectedReturnOn50Trades_125 || 0).toFixed(2)}%`,
  //     icon: DollarSign,
  //     valueColor: "text-purple-500"
  //   },
  //   {
  //     label: "Expected Return (25% Position sizing, 10 Trades)",
  //     value: `${(yearlyStats.expectedReturnOn10Trades_25 || 0).toFixed(2)}%`,
  //     icon: DollarSign,
  //     valueColor: "text-indigo-500"
  //   },
  //   {
  //     label: "Expected Return (25% Position sizing, 50 Trades)",
  //     value: `${(yearlyStats.expectedReturnOn50Trades_25 || 0).toFixed(2)}%`,
  //     icon: DollarSign,
  //     valueColor: "text-indigo-500"
  //   }
  // ];
  
  // const expectedValueStat = {
  //   label: "Expected Growth Per Trade",
  //   value: `${yearlyStats.expectedValuePerTrade.toFixed(2)}%`,
  //   icon: DollarSign,
  //   valueColor: "text-purple-500"
  // };

  return (
    <div className="space-y-1">
      <div className="grid grid-cols-2 gap-1">
        {mainStats.map((stat) => (
          <div key={stat.label}>
            <StatCard {...stat} />
          </div>
        ))}
      </div>
      {/* <div className="grid grid-cols-2 gap-1">
        {returnStats.map((stat) => (
          <div key={stat.label}>
            <StatCard {...stat} />
          </div>
        ))}
      </div>
      <div>
        <StatCard {...expectedValueStat} />
      </div> */}
    </div>
  );
};

export default YearlyStatistics;