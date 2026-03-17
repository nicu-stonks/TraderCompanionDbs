import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { fetchConfig, saveConfig, fetchSupplements, addSupplement, deleteSupplement, fetchFoods, addFood, deleteFood, LLMConfig, Supplement, FoodItem } from '../services/api';

export function ConfigPanel() {
  const [config, setConfig] = useState<LLMConfig>({ model_name: 'gemini-3-flash-preview', api_key: '', user_profile: '', available_models: 'gemini-3-flash-preview,gemini-1.5-pro' });
  const [supplements, setSupplements] = useState<Supplement[]>([]);
  const [foods, setFoods] = useState<FoodItem[]>([]);

  const [newSuppName, setNewSuppName] = useState('');
  const [newSuppDetails, setNewSuppDetails] = useState('');

  const [newFoodName, setNewFoodName] = useState('');
  const [newFoodDetails, setNewFoodDetails] = useState('');

  const [isSavingConfig, setIsSavingConfig] = useState(false);

  useEffect(() => {
    fetchConfig().then(c => setConfig(c || { model_name: 'gemini-3-flash-preview', api_key: '', user_profile: '', available_models: 'gemini-3-flash-preview,gemini-1.5-pro' })).catch(console.error);
    fetchSupplements().then(setSupplements).catch(console.error);
    fetchFoods().then(setFoods).catch(console.error);
  }, []);

  const handleSaveConfig = async () => {
    setIsSavingConfig(true);
    try {
      const updated = await saveConfig(config);
      setConfig(updated);
    } catch (e) {
      console.error(e);
    } finally {
      setIsSavingConfig(false);
    }
  };

  const handleAddSupplement = async () => {
    if (!newSuppName.trim()) return;
    try {
      const added = await addSupplement({ name: newSuppName, details: newSuppDetails });
      setSupplements([...supplements, added]);
      setNewSuppName('');
      setNewSuppDetails('');
    } catch (e) {
      console.error(e);
    }
  };

  const handleDeleteSupp = async (id: number) => {
    try {
      await deleteSupplement(id);
      setSupplements(supplements.filter(s => s.id !== id));
    } catch (e) {
      console.error(e);
    }
  };

  const handleAddFood = async () => {
    if (!newFoodName.trim()) return;
    try {
      const added = await addFood({ name: newFoodName, details: newFoodDetails });
      setFoods([...foods, added]);
      setNewFoodName('');
      setNewFoodDetails('');
    } catch (e) {
      console.error(e);
    }
  };

  const handleDeleteFood = async (id: number) => {
    try {
      await deleteFood(id);
      setFoods(foods.filter(f => f.id !== id));
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
      {/* LLM Config */}
      <Card>
        <CardHeader>
          <CardTitle>LLM Configuration</CardTitle>
          <CardDescription>Configure the Gemini model used for analysis.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Default Model Name</Label>
            <Input
              value={config.model_name}
              onChange={e => setConfig({ ...config, model_name: e.target.value })}
            />
          </div>
          <div className="space-y-2">
            <Label>Available Models</Label>
            <Input
              value={config.available_models || ''}
              onChange={e => setConfig({ ...config, available_models: e.target.value })}
              placeholder="gemini-3-flash-preview,gemini-1.5-pro-preview-0409"
            />
            <p className="text-sm text-muted-foreground">Comma-separated list of models to choose from in the chat.</p>
          </div>
          <div className="space-y-2">
            <Label>User Profile Details</Label>
            <Textarea
              value={config.user_profile || ''}
              onChange={e => setConfig({ ...config, user_profile: e.target.value })}
              placeholder="E.g., Age 30, Male, 180 lbs, 6'0&quot;, Activity Level: 4 intense workouts/week, Goals: Muscle gain & fat loss, Allergies: Peanuts, Medical Conditions: Mild hypertension, Diet Type: Vegetarian, Calorie Target: 2500, sit at computer all most of day(D3 supplements needeed), etc..."
              rows={3}
            />
          </div>
          <div className="space-y-2">
            <Label>API Key</Label>
            <Input
              type="password"
              value={config.api_key}
              onChange={e => setConfig({ ...config, api_key: e.target.value })}
              placeholder="AIzaSy..."
            />
          </div>
          <Button onClick={handleSaveConfig} disabled={isSavingConfig}>
            {isSavingConfig ? 'Saving...' : 'Save Config'}
          </Button>
        </CardContent>
      </Card>

      {/* Supplements Manager */}
      <Card>
        <CardHeader>
          <CardTitle>My Supplements Inventory</CardTitle>
          <CardDescription>Tell the AI what supplements you have available to take.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-4 max-h-[300px] overflow-y-auto pr-2">
            {supplements.map(s => (
              <div key={s.id} className="p-3 border rounded-md flex justify-between items-start">
                <div>
                  <div className="font-medium">{s.name}</div>
                  <div className="text-base text-muted-foreground whitespace-pre-wrap">{s.details}</div>
                </div>
                <Button variant="destructive" size="sm" onClick={() => s.id && handleDeleteSupp(s.id)}>
                  Remove
                </Button>
              </div>
            ))}
            {supplements.length === 0 && (
              <div className="text-base text-muted-foreground italic">No supplements configured yet.</div>
            )}
          </div>

          <div className="pt-4 border-t space-y-3">
            <h4 className="font-semibold text-sm">Add New Supplement</h4>
            <div className="space-y-2">
              <Input
                placeholder="Name (e.g. Omega 3)"
                value={newSuppName}
                onChange={e => setNewSuppName(e.target.value)}
              />
              <Textarea
                placeholder="Details (e.g. 1 pill contains 500mg EPA...)"
                value={newSuppDetails}
                onChange={e => setNewSuppDetails(e.target.value)}
                rows={2}
              />
              <Button onClick={handleAddSupplement} className="w-full">Add Supplement</Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Foods Manager */}
      <Card>
        <CardHeader>
          <CardTitle>Food Inventory</CardTitle>
          <CardDescription>Tell the AI what foods you have at your disposal in your kitchen/pantry.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-4 max-h-[300px] overflow-y-auto pr-2">
            {foods.map(f => (
              <div key={f.id} className="p-3 border rounded-md flex justify-between items-start">
                <div>
                  <div className="font-medium">{f.name}</div>
                  <div className="text-base text-muted-foreground whitespace-pre-wrap">{f.details}</div>
                </div>
                <Button variant="destructive" size="sm" onClick={() => f.id && handleDeleteFood(f.id)}>
                  Remove
                </Button>
              </div>
            ))}
            {foods.length === 0 && (
              <div className="text-base text-muted-foreground italic">No foods configured yet.</div>
            )}
          </div>

          <div className="pt-4 border-t space-y-3">
            <h4 className="font-semibold text-sm">Add New Food</h4>
            <div className="space-y-2">
              <Input
                placeholder="Name (e.g. Protein Shake)"
                value={newFoodName}
                onChange={e => setNewFoodName(e.target.value)}
              />
              <Textarea
                placeholder="Details (e.g. 50g protein, 200 calories...)"
                value={newFoodDetails}
                onChange={e => setNewFoodDetails(e.target.value)}
                rows={2}
              />
              <Button onClick={handleAddFood} className="w-full">Add Food</Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
