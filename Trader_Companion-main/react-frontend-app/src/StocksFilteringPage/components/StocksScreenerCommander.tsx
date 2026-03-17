import React, { useState } from 'react';
import { ScreeningOptions } from '../types/screenerCommander';
import { useStockScreener } from '../hooks/useStockScreener';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Moon } from "lucide-react";

const OBLIGATORY_SCREEN_OPTIONS = [
  { id: 'above_52week_low', label: 'Above 52 Week Low' },
  { id: 'trending_up', label: 'Trending Up' },
  { id: 'close_to_52week_high', label: 'Close to 52 Week High' },
  { id: 'minimum_volume_100k', label: 'Minimum Volume 100K' },
  { id: 'minimum_price_increase', label: 'Minimum Price Increase' },
];

const RANKING_SCREEN_OPTIONS = [
  { id: 'annual_EPS_acceleration', label: 'Annual EPS Acceleration' },
  { id: 'annual_margin_acceleration', label: 'Annual Margin Acceleration' },
  { id: 'annual_sales_acceleration', label: 'Annual Sales Acceleration' },
  { id: 'quarterly_EPS_acceleration', label: 'Quarterly EPS Acceleration' },
  { id: 'quarterly_eps_breakout', label: 'Quarterly EPS Breakout' },
  { id: 'quarterly_margin_acceleration', label: 'Quarterly Margin Acceleration' },
  { id: 'quarterly_sales_acceleration', label: 'Quarterly Sales Acceleration' },
  { id: 'rs_over_70', label: 'RS Over 70' },
  { id: 'rsi_trending_up', label: 'RSI Trending Up' },
  { id: 'volume_acceleration', label: 'Volume Acceleration' },
  { id: 'price_spikes', label: 'Price Spikes' },
  { id: 'top_price_increases_1y', label: 'Top Price Increases (1Y)' }
];

export const StocksScreenerCommander: React.FC = () => {
  const { response, loading, error, sendCommand } = useStockScreener();

  const IS_DEPRECATED_AND_DISABLED = true;

  const [options, setOptions] = useState<ScreeningOptions>({
    min_price_increase: 20,
    ranking_method: 'price',
    fetch_data: true,
    top_n: 1000,
    obligatory_screens: ['above_52week_low', 'trending_up', 'close_to_52week_high', 'minimum_price_increase', 'minimum_volume_100k'],
    ranking_screens: ['annual_EPS_acceleration', 'annual_margin_acceleration', 'annual_sales_acceleration', 'quarterly_EPS_acceleration', 'quarterly_eps_breakout', 'quarterly_margin_acceleration', 'quarterly_sales_acceleration', 'rs_over_70', 'rsi_trending_up', 'volume_acceleration', 'price_spikes', 'top_price_increases_1y'],
    skip_obligatory: false,
    skip_sentiment: false,
    sleep_after: true
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (IS_DEPRECATED_AND_DISABLED) {
      return;
    }

    sendCommand(options);
  };

  const handleObligatoryScreenChange = (screenId: string) => {
    setOptions(prev => ({
      ...prev,
      obligatory_screens: prev.obligatory_screens.includes(screenId)
        ? prev.obligatory_screens.filter(id => id !== screenId)
        : [...prev.obligatory_screens, screenId]
    }));
  };

  const handleRankingScreenChange = (screenId: string) => {
    setOptions(prev => ({
      ...prev,
      ranking_screens: prev.ranking_screens.includes(screenId)
        ? prev.ranking_screens.filter(id => id !== screenId)
        : [...prev.ranking_screens, screenId]
    }));
  };

  return (
    <div className="p-4 md:p-6">
      <Card className="w-full max-w-4xl mx-auto">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>
            Stock Screener Commander
            {IS_DEPRECATED_AND_DISABLED ? (
              <span className="ml-2 text-sm font-normal text-muted-foreground">
                (Deprecated / Disabled)
              </span>
            ) : null}
          </CardTitle>
        </CardHeader>

        <form onSubmit={handleSubmit}>
          <CardContent className="space-y-8">
            {IS_DEPRECATED_AND_DISABLED ? (
              <Alert>
                <AlertDescription>
                  This feature is currently disabled and deprecated (not maintained) since it's not used/needed anymore. It’s kept here for reference.
                </AlertDescription>
              </Alert>
            ) : null}

            {/* Global Settings */}
            <div className={IS_DEPRECATED_AND_DISABLED ? "space-y-4 opacity-50 pointer-events-none" : "space-y-4"}>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex items-center space-x-2">
                  <Switch
                    disabled={IS_DEPRECATED_AND_DISABLED}
                    checked={!options.skip_obligatory}
                    onCheckedChange={(checked) => setOptions(prev => ({
                      ...prev,
                      skip_obligatory: !checked
                    }))}
                  />
                  <Label>Enable Obligatory Screens</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    disabled={IS_DEPRECATED_AND_DISABLED}
                    checked={!options.skip_sentiment}
                    onCheckedChange={(checked) => setOptions(prev => ({
                      ...prev,
                      skip_sentiment: !checked
                    }))}
                  />
                  <Label>Enable Sentiment Analysis</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    disabled={IS_DEPRECATED_AND_DISABLED}
                    checked={options.fetch_data}
                    onCheckedChange={(checked) => setOptions(prev => ({
                      ...prev,
                      fetch_data: checked
                    }))}
                  />
                  <Label>Fetch Latest Stock Data</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    disabled={IS_DEPRECATED_AND_DISABLED}
                    checked={options.sleep_after}
                    onCheckedChange={(checked) => setOptions(prev => ({
                      ...prev,
                      sleep_after: checked
                    }))}
                  />
                  <div className="flex items-center space-x-1">
                    <Moon className="h-4 w-4" />
                    <Label>Sleep After Completion</Label>
                  </div>
                </div>
              </div>
            </div>

            {/* Screening Parameters */}
            <div className={IS_DEPRECATED_AND_DISABLED ? "grid grid-cols-1 md:grid-cols-2 gap-6 opacity-50 pointer-events-none" : "grid grid-cols-1 md:grid-cols-2 gap-6"}>
              <div className="space-y-2">
                <Label>Ranking Criteria</Label>
                <Select
                  value={options.ranking_method}
                  onValueChange={(value) => setOptions(prev => ({
                    ...prev,
                    ranking_method: value as ScreeningOptions['ranking_method']
                  }))}
                >
                  <SelectTrigger disabled={IS_DEPRECATED_AND_DISABLED}>
                    <SelectValue placeholder="Select ranking method" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="price">Price Increase</SelectItem>
                    <SelectItem value="screeners">Number of Screeners</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Minimum Price Increase (%)</Label>
                <Input
                  type="number"
                  value={options.min_price_increase}
                  disabled={IS_DEPRECATED_AND_DISABLED || options.skip_obligatory}
                  onChange={(e) => setOptions(prev => ({
                    ...prev,
                    min_price_increase: parseFloat(e.target.value)
                  }))}
                />
              </div>

              <div className="space-y-2">
                <Label>Top N Results</Label>
                <Input
                  type="number"
                  value={options.top_n}
                  disabled={IS_DEPRECATED_AND_DISABLED}
                  onChange={(e) => setOptions(prev => ({
                    ...prev,
                    top_n: parseInt(e.target.value)
                  }))}
                />
              </div>
            </div>

            {/* Screening Options */}
            <div className={IS_DEPRECATED_AND_DISABLED ? "space-y-6 opacity-50 pointer-events-none" : "space-y-6"}>
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Obligatory Screens</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {OBLIGATORY_SCREEN_OPTIONS.map(screen => (
                    <div key={screen.id} className="flex items-center space-x-2">
                      <Switch
                        disabled={IS_DEPRECATED_AND_DISABLED || options.skip_obligatory}
                        checked={options.obligatory_screens.includes(screen.id)}
                        onCheckedChange={() => handleObligatoryScreenChange(screen.id)}
                      />
                      <Label>{screen.label}</Label>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Ranking Screens</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {RANKING_SCREEN_OPTIONS.map(screen => (
                    <div key={screen.id} className="flex items-center space-x-2">
                      <Switch
                        disabled={IS_DEPRECATED_AND_DISABLED}
                        checked={options.ranking_screens.includes(screen.id)}
                        onCheckedChange={() => handleRankingScreenChange(screen.id)}
                      />
                      <Label>{screen.label}</Label>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>

          <CardFooter className="flex flex-col space-y-4">
            <Button
              type="submit"
              className="w-full"
              variant="default"
              disabled={IS_DEPRECATED_AND_DISABLED || loading}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Screening...
                </>
              ) : (
                IS_DEPRECATED_AND_DISABLED ? 'Disabled' : 'Start Screening'
              )}
            </Button>

            <p className="text-sm text-muted-foreground">
              TODO: Redo the entire logic. Save the data per ticker instead of having lists for each metric... MUCH more efficient. Have checkboxes for each stock where each checkbox is a criteria like above 200MA, RS, etc. That will make the filtering instand like in marketsurge. Let user add tickers to be screened. Let user do his own screens like in marketsurge. Show user chart with RS ranking and RS line for a stock. Let him go through his list with space like in marketsmurge(keep fetching in the background the next tickers one by one until all list done to improve performance, show user how many loaded already, but do this if normal fetching is slow). You have that ibd rs script done already which looked good.
            </p>
            <p className="text-sm text-muted-foreground">
              TODO: You could ALSO do a request for that single ticker to stockanalysis or some other page for the earnings data with scraping. genius
            </p>
            <p className="text-sm text-muted-foreground">
              TODO: Also you can do it like mark screening where he just presses a button or just double taps left arrow and its added to a list. You can make it so that when you press 1 its added to small caps, 2 for big caps, 3 for medium caps etc. Or you can let user customize it also. Many other hotkeys you can add to maximize efficiency like this.
            </p>
            <p className="text-sm text-muted-foreground">
              TODO: Low volume vs 50 day and tight price screen
            </p>
            <p className="text-sm text-muted-foreground">
              IDEA: Broke out excluder: 52wk high 2 weeks ago(Or N weeks/days ago, could be configurable again) more than 15% away from current price(or 10%, configurable preferably)
            </p>
            {response && (
              <Alert>
                <AlertDescription>{response}</AlertDescription>
              </Alert>
            )}

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardFooter>
        </form>
      </Card>
    </div>
  )
    ;
};

export default StocksScreenerCommander;