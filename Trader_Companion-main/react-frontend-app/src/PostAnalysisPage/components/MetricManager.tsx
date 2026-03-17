import React, { useState } from 'react';
import { Edit2, Loader, Plus, X } from 'lucide-react';
import { Metric } from '../types/types';
import { metricService } from '../services/postAnalysis';
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
} from '@/components/ui/alert-dialog';


const MetricManager: React.FC<{
  metrics: Metric[];
  onRefetch: () => void;
}> = ({ metrics, onRefetch }) => {
  const [newMetricName, setNewMetricName] = useState('');
  const [newMetricDescription, setNewMetricDescription] = useState('');
  const [editingMetric, setEditingMetric] = useState<number | null>(null);
  const [newOptionName, setNewOptionName] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAddMetric = async () => {
    if (!newMetricName.trim()) return;

    setLoading(true);
    try {
      await metricService.createMetric({
        name: newMetricName.trim(),
        description: newMetricDescription.trim(),
        options: []
      });

      setNewMetricName('');
      setNewMetricDescription('');
      onRefetch();
    } catch (error) {
      console.error('Failed to create metric:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveMetric = async (metricId: number) => {
    setLoading(true);
    try {
      await metricService.deleteMetric(metricId);
      onRefetch();
    } catch (error) {
      console.error('Failed to delete metric:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddOption = async (metricId: number) => {
    if (!newOptionName.trim()) return;

    setLoading(true);
    try {
      await metricService.addOption(metricId, newOptionName.trim());
      setNewOptionName('');
      setEditingMetric(null);
      onRefetch();
    } catch (error) {
      console.error('Failed to add option:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveOption = async (metricId: number, optionId: number) => {
    setLoading(true);
    try {
      await metricService.removeOption(metricId, optionId);
      onRefetch();
    } catch (error) {
      console.error('Failed to remove option:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-lg shadow-md p-6 mb-6 bg-background text-foreground">
      <h2 className="text-2xl font-bold mb-4 flex items-center">
        <Edit2 className="mr-2" />
        Metric Manager
      </h2>

      {/* Add new metric */}
      <div className="mb-6 p-4 bg-muted rounded-lg">
        <h3 className="text-lg font-semibold mb-3">Create New Metric</h3>
        <div className="space-y-3">
          <input
            type="text"
            value={newMetricName}
            onChange={(e) => setNewMetricName(e.target.value)}
            placeholder="Enter metric name (e.g., Entry Point, Fundamentals)"
            className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary bg-background text-foreground border-input"
            disabled={loading}
          />

          <button
            onClick={handleAddMetric}
            disabled={loading || !newMetricName.trim()}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {loading ? <Loader className="w-4 h-4 mr-2 animate-spin" /> : <Plus className="w-4 h-4 mr-1" />}
            Add Metric
          </button>
        </div>
      </div>

      {/* Existing metrics */}
      <div className="space-y-4">
        {metrics.map(metric => (
          <div key={metric.id} className="border rounded-lg p-4 border-border bg-card text-card-foreground">
            <div className="flex justify-between items-start mb-3">
              <div className="flex-1">
                <h4 className="text-lg font-medium">{metric.name}</h4>
                {metric.description && (
                  <p className="text-sm text-muted-foreground mt-1">{metric.description}</p>
                )}
              </div>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <button
                    disabled={loading}
                    className="text-destructive hover:text-destructive/80 disabled:opacity-50"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Delete Metric?</AlertDialogTitle>
                    <AlertDialogDescription>
                      Are you sure you want to delete <strong>{metric.name}</strong>?
                      This will also delete all associated grades. This action cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      onClick={() => handleRemoveMetric(metric.id)}
                      className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                    >
                      Delete
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>

            {/* Options */}
            <div className="mb-3">
              <div className="flex flex-wrap gap-2 mb-2">
                {metric.options.map(option => (
                  <div key={option.id} className="flex items-center bg-accent text-accent-foreground rounded-full px-3 py-1">
                    <span className="text-sm">{option.name}</span>
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <button
                          disabled={loading}
                          className="ml-2 text-destructive hover:text-destructive/80 disabled:opacity-50"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Delete Option?</AlertDialogTitle>
                          <AlertDialogDescription>
                            Are you sure you want to delete the option <strong>{option.name}</strong> from <strong>{metric.name}</strong>?
                            This action cannot be undone.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogAction
                            onClick={() => handleRemoveOption(metric.id, option.id)}
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                          >
                            Delete
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  </div>
                ))}
              </div>
            </div>

            {/* Add option */}
            {editingMetric === metric.id ? (
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newOptionName}
                  onChange={(e) => setNewOptionName(e.target.value)}
                  placeholder="Enter option name"
                  className="flex-1 px-3 py-1 text-sm border rounded-md focus:outline-none focus:ring-1 focus:ring-primary bg-background text-foreground border-input"
                  disabled={loading}
                />
                <button
                  onClick={() => handleAddOption(metric.id)}
                  disabled={loading || !newOptionName.trim()}
                  className="px-3 py-1 bg-primary text-primary-foreground text-sm rounded-md hover:bg-primary/90 disabled:opacity-50"
                >
                  {loading ? <Loader className="w-3 h-3 animate-spin" /> : 'Add'}
                </button>
                <button
                  onClick={() => {
                    setEditingMetric(null);
                    setNewOptionName('');
                  }}
                  disabled={loading}
                  className="px-3 py-1 bg-muted text-muted-foreground text-sm rounded-md hover:bg-muted/80 disabled:opacity-50"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setEditingMetric(metric.id)}
                disabled={loading}
                className="text-primary hover:text-primary/80 text-sm flex items-center disabled:opacity-50"
              >
                <Plus className="w-3 h-3 mr-1" />
                Add Option
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};


export default MetricManager;