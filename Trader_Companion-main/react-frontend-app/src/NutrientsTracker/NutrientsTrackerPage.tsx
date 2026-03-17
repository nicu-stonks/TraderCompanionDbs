import { useState } from 'react';
import { ConfigPanel } from './components/ConfigPanel';
import { DailyTracker } from './components/DailyTracker';
import { TrendChart, RecordsTable } from './components/AnalyticsDashboard';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export function NutrientsTrackerPage() {
  const [activeTab, setActiveTab] = useState('tracker');

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">AI Nutrients Tracker</h1>
      </div>
      <div className="text-muted-foreground w-full max-w-3xl">
        Track your macronutrients, vitamins, and diet. The AI nutritionist will analyze your daily
        intake and recommend specific whole foods or supplements from your inventory to hit your goals.
      </div>

      <TrendChart />

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3 max-w-[500px]">
          <TabsTrigger value="tracker">Daily Log Chat</TabsTrigger>
          <TabsTrigger value="analytics">Records List</TabsTrigger>
          <TabsTrigger value="config">Settings & Inventory</TabsTrigger>
        </TabsList>

        <TabsContent value="tracker" className="mt-6">
          <DailyTracker />
        </TabsContent>

        <TabsContent value="analytics" className="mt-6">
          <RecordsTable />
        </TabsContent>

        <TabsContent value="config" className="mt-6">
          <ConfigPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
