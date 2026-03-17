import React, { useState, useEffect } from 'react';
import type { GlobalCharacteristic } from '../types';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Plus, Edit, Check, X, Save, Trash2 } from 'lucide-react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { globalCharacteristicsApi } from '../services/globalCharacteristics';

interface Props {
  onClose?: () => void;
}

const GlobalCharacteristicsManager: React.FC<Props> = ({ onClose }) => {
  const [characteristics, setCharacteristics] = useState<GlobalCharacteristic[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isAddingNew, setIsAddingNew] = useState(false);
  const [newCharacteristic, setNewCharacteristic] = useState({
    name: '',
    default_score: '0'
  });
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editedName, setEditedName] = useState('');
  const [editedScore, setEditedScore] = useState('0');

  // Fetch characteristics on component mount
  useEffect(() => {
    fetchCharacteristics();
  }, []);

  // Fetch global characteristics from API
  const fetchCharacteristics = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await globalCharacteristicsApi.getAllGlobalCharacteristics();
      setCharacteristics(response.data);
    } catch (err) {
      console.error('Error fetching global characteristics:', err);
      setError('Failed to load global characteristics');
    } finally {
      setIsLoading(false);
    }
  };

  // Add a new characteristic
  const handleAddCharacteristic = async () => {
    if (!newCharacteristic.name.trim()) {
      setError('Characteristic name is required');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      await globalCharacteristicsApi.createGlobalCharacteristic({
        name: newCharacteristic.name.trim(),
        default_score: Number(newCharacteristic.default_score) || 0
      });

      // Refresh the list
      await fetchCharacteristics();
      setIsAddingNew(false);
      setNewCharacteristic({ name: '', default_score: '0' });
    } catch (err) {
      console.error('Error adding global characteristic:', err);
      setError('Failed to add characteristic');
    } finally {
      setIsLoading(false);
    }
  };

  // Start editing a characteristic
  const handleStartEdit = (characteristic: GlobalCharacteristic) => {
    setEditingId(characteristic.id);
    setEditedName(characteristic.name);
    setEditedScore(String(characteristic.default_score));
  };

  // Cancel editing
  const handleCancelEdit = () => {
    setEditingId(null);
  };

  // Save edited characteristic
  const handleSaveEdit = async (id: number) => {
    if (!editedName.trim()) {
      setError('Characteristic name is required');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      await globalCharacteristicsApi.updateGlobalCharacteristic(id, {
        name: editedName.trim(),
        default_score: Number(editedScore) || 0
      });

      // Refresh the list
      await fetchCharacteristics();
      setEditingId(null);
    } catch (err) {
      console.error('Error updating global characteristic:', err);
      setError('Failed to update characteristic');
    } finally {
      setIsLoading(false);
    }
  };

  // Delete a characteristic
  const handleDeleteCharacteristic = async (id: number) => {
    try {
      setIsLoading(true);
      setError(null);
      await globalCharacteristicsApi.deleteGlobalCharacteristic(id);

      // Refresh the list
      await fetchCharacteristics();
    } catch (err) {
      console.error('Error deleting global characteristic:', err);
      setError('Failed to delete characteristic');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="w-full h-full flex flex-col">
      <CardHeader className="flex flex-row items-center justify-between py-2 px-2">
        <CardTitle className="text-base">Global Characteristics</CardTitle>
        <div className="flex items-center gap-1">
          <Button
            size="sm"
            onClick={() => setIsAddingNew(!isAddingNew)}
            disabled={isLoading}
            className="h-7 px-2"
          >
            {isAddingNew ? (
              <>
                <X className="mr-1 h-3 w-3" /> Cancel
              </>
            ) : (
              <>
                <Plus className="mr-1 h-3 w-3" /> Add
              </>
            )}
          </Button>
          {onClose && (
            <Button size="sm" variant="outline" onClick={onClose} className="h-7 px-2">
              Close
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0 flex-1 overflow-auto">
        <div className="p-2">
          {error && (
            <Alert variant="destructive" className="mb-2 py-1 text-sm">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {isAddingNew && (
            <div className="mb-2 p-2 border rounded-md bg-muted/50">
              <div className="flex gap-1 items-center">
                <Input
                  placeholder="Name"
                  value={newCharacteristic.name}
                  onChange={(e) => setNewCharacteristic(prev => ({
                    ...prev,
                    name: e.target.value
                  }))}
                  disabled={isLoading}
                  className="flex-1 h-7 text-sm"
                />
                <Input
                  type="text"
                  inputMode="numeric"
                  placeholder="Score"
                  value={newCharacteristic.default_score}
                  onChange={(e) => setNewCharacteristic(prev => ({
                    ...prev,
                    default_score: e.target.value
                  }))}
                  disabled={isLoading}
                  className="w-16 h-7 text-sm"
                />
                <Button
                  onClick={handleAddCharacteristic}
                  disabled={isLoading || !newCharacteristic.name.trim()}
                  size="sm"
                  className="h-7 px-2"
                >
                  <Save className="h-3 w-3" />
                </Button>
              </div>
            </div>
          )}

          {isLoading && characteristics.length === 0 ? (
            <p className="text-center text-muted-foreground py-2 text-sm">Loading...</p>
          ) : characteristics.length === 0 ? (
            <p className="text-center text-muted-foreground py-2 text-sm">No characteristics defined.</p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
              {characteristics.map(characteristic => (
                <div key={characteristic.id} className="border rounded-md p-2">
                  {editingId === characteristic.id ? (
                    <div className="flex items-center gap-1">
                      <Input
                        value={editedName}
                        onChange={(e) => setEditedName(e.target.value)}
                        disabled={isLoading}
                        className="flex-1 h-7 text-sm"
                      />
                      <Input
                        type="text"
                        inputMode="numeric"
                        value={editedScore}
                        onChange={(e) => setEditedScore(e.target.value)}
                        disabled={isLoading}
                        className="w-16 h-7 text-sm"
                      />
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={() => handleSaveEdit(characteristic.id)}
                        disabled={isLoading || !editedName.trim()}
                        className="h-6 w-6"
                      >
                        <Check className="h-3 w-3" />
                      </Button>
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={handleCancelEdit}
                        disabled={isLoading}
                        className="h-6 w-6"
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </div>
                  ) : (
                    <div className="flex items-center justify-between">
                      <div className="flex-1 text-sm">
                        <span className="font-medium">{characteristic.name}</span>
                        <span className="text-muted-foreground ml-2">Score: {characteristic.default_score}</span>
                      </div>
                      <div className="flex items-center">
                        <Button
                          size="icon"
                          variant="ghost"
                          onClick={() => handleStartEdit(characteristic)}
                          disabled={isLoading}
                          className="h-6 w-6"
                        >
                          <Edit className="h-3 w-3" />
                        </Button>
                        <AlertDialog>
                          <AlertDialogTrigger asChild>
                            <Button
                              size="icon"
                              variant="ghost"
                              disabled={isLoading}
                              className="h-6 w-6 text-destructive"
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </AlertDialogTrigger>
                          <AlertDialogContent className="max-w-xs">
                            <AlertDialogHeader className="space-y-1">
                              <AlertDialogTitle className="text-base">Delete Characteristic</AlertDialogTitle>
                              <AlertDialogDescription className="text-sm">
                                Delete "{characteristic.name}"? This will remove it from all stocks and cannot be undone.
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter className="flex-row gap-1 justify-end">
                              <AlertDialogCancel className="h-7 px-2 text-sm">Cancel</AlertDialogCancel>
                              <AlertDialogAction
                                onClick={() => handleDeleteCharacteristic(characteristic.id)}
                                className="bg-destructive text-destructive-foreground hover:bg-destructive/90 h-7 px-2 text-sm"
                              >
                                Delete
                              </AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default GlobalCharacteristicsManager;