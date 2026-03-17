import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { fetchDailyRecords, DailyRecord } from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { nutrientCategories } from './NutrientStats';

export function TrendChart() {
  const [records, setRecords] = useState<DailyRecord[]>([]);

  const allNutrients = Object.values(nutrientCategories).flat();
  const totalNutrients = allNutrients.length;

  useEffect(() => {
    fetchDailyRecords().then(data => {
      const sorted = [...data].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
      setRecords(sorted);
    }).catch(console.error);
  }, []);

  const chartData = records.map(r => {
    const discreteCompleted = allNutrients.filter(n => (r.nutrient_completion?.[n] || 0) >= 100).length;
    const discreteCompletion = Math.round((discreteCompleted / totalNutrients) * 100);

    const nonDiscreteSum = allNutrients.reduce((acc, n) => acc + Math.min(r.nutrient_completion?.[n] || 0, 100), 0);
    const nonDiscreteCompletion = Math.round(nonDiscreteSum / totalNutrients);

    return {
      date: r.date.split('-').slice(1).join('/'),
      discreteCompletion,
      nonDiscreteCompletion
    };
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Nutrition History Trend</CardTitle>
        <CardDescription>Discrete completion (fully met nutrients) vs non-discrete average completion.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[250px] w-full">
          {chartData.length === 0 ? (
            <div className="h-full flex items-center justify-center text-muted-foreground italic">
              No data available yet.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="date" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `${val}%`} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'hsl(var(--background))', borderColor: 'hsl(var(--border))' }}
                  labelStyle={{ color: 'hsl(var(--foreground))' }}
                />
                <Line type="monotone" dataKey="nonDiscreteCompletion" name="Non-Discrete" stroke="hsl(var(--ring))" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                <Line type="monotone" dataKey="discreteCompletion" name="Discrete" stroke="hsl(var(--primary))" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export function RecordsTable() {
  const [records, setRecords] = useState<DailyRecord[]>([]);
  const [expandedDate, setExpandedDate] = useState<string | null>(null);

  useEffect(() => {
    fetchDailyRecords().then(data => {
      setRecords([...data].reverse());
    }).catch(console.error);
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Daily Records Details</CardTitle>
        <CardDescription>Expand a day to see exactly what you ate and what the AI recommended.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {records.length === 0 && (
            <div className="text-base text-muted-foreground italic text-center py-4">
              No records found.
            </div>
          )}
          {records.map(record => {
            const isExpanded = expandedDate === record.date;
            const nKeys = Object.keys(record.nutrient_completion || {});
            const sum = nKeys.reduce((acc, k) => acc + Math.min(record.nutrient_completion[k] || 0, 100), 0);
            const avg = nKeys.length > 0 ? Math.round(sum / nKeys.length) : 0;

            return (
              <div key={record.date} className="border rounded-md overflow-hidden">
                <div
                  className="bg-muted/30 p-4 flex items-center justify-between cursor-pointer hover:bg-muted/50 transition-colors"
                  onClick={() => setExpandedDate(isExpanded ? null : record.date)}
                >
                  <div className="font-semibold">{record.date}</div>
                  <div className="flex items-center space-x-4">
                    <div className="hidden sm:block text-base text-muted-foreground">Avg Completion:</div>
                    <div className="w-24 flex items-center space-x-2">
                      <Progress value={avg} className="h-2" />
                      <span className="text-sm font-medium w-8 text-right">{avg}%</span>
                    </div>
                    <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                      {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>

                {isExpanded && (
                  <div className="p-4 bg-background border-t space-y-6 text-base">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 border-b pb-4">
                      <div>
                        <h4 className="font-semibold mb-2 text-muted-foreground">Foods Eaten</h4>
                        {record.foods_eaten ? (
                          <ul className="list-disc pl-4 space-y-1">
                            {record.foods_eaten.split('\n').map((food: string, i: number) => (
                              <li key={i}>{food}</li>
                            ))}
                          </ul>
                        ) : (
                          <div>None specified.</div>
                        )}
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2 text-muted-foreground">Supplements Taken</h4>
                        <ul className="list-disc list-inside">
                          {record.supplements_taken && record.supplements_taken.length > 0 ? (
                            record.supplements_taken.map((s, i) => (
                              <li key={i}>{typeof s === 'string' ? s : s.name}</li>
                            ))
                          ) : "None specified."}
                        </ul>
                      </div>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2 text-muted-foreground">Nutrient Fills</h4>
                      {nKeys.length === 0 ? "No nutrient data." : (
                        <div className="grid grid-cols-2 gap-x-8 gap-y-2">
                          {Object.entries(record.nutrient_completion).map(([k, v]) => (
                            <div key={k} className="flex justify-between items-center text-sm">
                              <span>{k}</span>
                              <span className="font-medium">{v}%</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                    <div className="border-t pt-4">
                      <h4 className="font-semibold mb-2 text-muted-foreground">Final AI Recommendations</h4>
                      <div className="bg-secondary/20 p-3 rounded text-base whitespace-pre-wrap">
                        {record.recommendations || "No recommendations generated."}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  );
}
