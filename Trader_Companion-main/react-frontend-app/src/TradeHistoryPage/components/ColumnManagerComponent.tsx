// components/ColumnManagerComponent.tsx
import React, { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
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
import { Settings, Plus, Trash2, GripVertical, Pencil, Check, X, TextCursorInput } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import type { UseCustomTradeDataReturn, ColumnDef } from '../hooks/useCustomTradeData';

interface ColumnManagerComponentProps {
  customTradeData: UseCustomTradeDataReturn;
}

export const ColumnManagerComponent: React.FC<ColumnManagerComponentProps> = ({ customTradeData }) => {
  const {
    customColumns,
    orderedColumns,
    addCustomColumn,
    renameCustomColumn,
    deleteCustomColumn,
    saveColumnOrder,
  } = customTradeData;

  const [newColumnName, setNewColumnName] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editingName, setEditingName] = useState('');
  const [draggedKey, setDraggedKey] = useState<string | null>(null);
  const [dragOverKey, setDragOverKey] = useState<string | null>(null);
  const draggedKeyRef = useRef<string | null>(null);
  const [localColumns, setLocalColumns] = useState<ColumnDef[] | null>(null);

  // Use localColumns when dragging, otherwise orderedColumns
  const displayColumns = localColumns || orderedColumns;

  const handleAddColumn = async () => {
    const name = newColumnName.trim();
    if (!name) {
      setError('Column name cannot be empty.');
      return;
    }
    if (customColumns.some(c => c.name.toLowerCase() === name.toLowerCase())) {
      setError('A column with this name already exists.');
      return;
    }
    setIsAdding(true);
    setError(null);
    try {
      await addCustomColumn(name);
      setNewColumnName('');
    } catch {
      setError('Failed to create column.');
    } finally {
      setIsAdding(false);
    }
  };

  const handleRename = async (id: number) => {
    const name = editingName.trim();
    if (!name) return;
    try {
      await renameCustomColumn(id, name);
      setEditingId(null);
      setEditingName('');
    } catch {
      setError('Failed to rename column.');
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await deleteCustomColumn(id);
    } catch {
      setError('Failed to delete column.');
    }
  };

  // Drag and drop handlers — track by stable key, not mutable index
  const handleDragStart = (e: React.DragEvent, key: string) => {
    draggedKeyRef.current = key;
    setDraggedKey(key);
    setLocalColumns([...displayColumns]);
    // Required for Firefox
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent, overKey: string) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    const currentDraggedKey = draggedKeyRef.current;
    if (!currentDraggedKey || currentDraggedKey === overKey) {
      setDragOverKey(null);
      return;
    }
    setDragOverKey(overKey);

    // Live reorder while dragging
    setLocalColumns(prev => {
      if (!prev) return prev;
      const dragIdx = prev.findIndex(c => c.key === currentDraggedKey);
      const overIdx = prev.findIndex(c => c.key === overKey);
      if (dragIdx === -1 || overIdx === -1 || dragIdx === overIdx) return prev;
      const newCols = [...prev];
      const [dragged] = newCols.splice(dragIdx, 1);
      newCols.splice(overIdx, 0, dragged);
      return newCols;
    });
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOverKey(null);
    setDraggedKey(null);
    draggedKeyRef.current = null;

    if (localColumns) {
      await saveColumnOrder(localColumns);
      setLocalColumns(null);
    }
  };

  const handleDragEnd = () => {
    setDraggedKey(null);
    setDragOverKey(null);
    draggedKeyRef.current = null;
    if (localColumns) {
      saveColumnOrder(localColumns);
      setLocalColumns(null);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader className="py-3">
        <CardTitle className="text-lg font-semibold flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Column Manager
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3 space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Add New Custom Column */}
        <div className="flex items-end gap-2">
          <div className="flex-grow space-y-1">
            <Label htmlFor="new-col-name" className="text-sm">Add Custom Column</Label>
            <Input
              id="new-col-name"
              placeholder="Column name (text only)"
              value={newColumnName}
              onChange={(e) => setNewColumnName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAddColumn()}
              className="h-8"
              disabled={isAdding}
            />
          </div>
          <Button
            onClick={handleAddColumn}
            disabled={isAdding || !newColumnName.trim()}
            size="sm"
            className="h-8 flex items-center gap-1"
          >
            <Plus className="h-4 w-4" />
            Add
          </Button>
        </div>

        {/* Custom Columns List (CRUD) */}
        {customColumns.length > 0 && (
          <div className="space-y-1">
            <Label className="text-sm font-medium">Custom Columns</Label>
            <div className="border rounded-md divide-y">
              {customColumns.map(col => (
                <div key={col.id} className="flex items-center justify-between px-3 py-1.5">
                  {editingId === col.id ? (
                    <div className="flex items-center gap-1 flex-grow">
                      <Input
                        value={editingName}
                        onChange={(e) => setEditingName(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleRename(col.id)}
                        className="h-7 text-sm"
                        autoFocus
                      />
                      <Button size="sm" variant="ghost" className="h-7 w-7 p-0" onClick={() => handleRename(col.id)}>
                        <Check className="h-3.5 w-3.5 text-emerald-500" />
                      </Button>
                      <Button size="sm" variant="ghost" className="h-7 w-7 p-0" onClick={() => { setEditingId(null); setEditingName(''); }}>
                        <X className="h-3.5 w-3.5 text-red-500" />
                      </Button>
                    </div>
                  ) : (
                    <>
                      <span className="text-sm">{col.name}</span>
                      <div className="flex items-center gap-1">
                        <Button
                          size="sm" variant="ghost" className="h-7 w-7 p-0"
                          onClick={() => { setEditingId(col.id); setEditingName(col.name); }}
                        >
                          <Pencil className="h-3.5 w-3.5" />
                        </Button>
                        <AlertDialog>
                          <AlertDialogTrigger asChild>
                            <Button size="sm" variant="ghost" className="h-7 w-7 p-0 text-red-500 hover:text-red-600">
                              <Trash2 className="h-3.5 w-3.5" />
                            </Button>
                          </AlertDialogTrigger>
                          <AlertDialogContent>
                            <AlertDialogHeader>
                              <AlertDialogTitle>Delete Column "{col.name}"</AlertDialogTitle>
                              <AlertDialogDescription>
                                This will permanently delete this custom column and all its values across all trades. This action cannot be undone.
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                              <AlertDialogCancel>Cancel</AlertDialogCancel>
                              <AlertDialogAction onClick={() => handleDelete(col.id)}>
                                Delete
                              </AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Column Ordering */}
        <div className="space-y-1">
          <Label className="text-sm font-medium">Column Order (drag to reorder) · Textarea toggle · Width in px (0 = default)</Label>
          <div className="border rounded-md divide-y max-h-[400px] overflow-y-auto">
            {displayColumns.map((col) => (
              <div
                key={col.key}
                draggable
                onDragStart={(e) => handleDragStart(e, col.key)}
                onDragOver={(e) => handleDragOver(e, col.key)}
                onDrop={handleDrop}
                onDragEnd={handleDragEnd}
                className={`flex items-center gap-2 px-3 py-1.5 cursor-move hover:bg-muted/50 transition-colors
                  ${draggedKey === col.key ? 'opacity-50 bg-muted' : ''}
                  ${dragOverKey === col.key ? 'border-t-2 border-primary' : ''}
                `}
              >
                <GripVertical className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                <span className="text-sm flex-grow">{col.label}</span>
                {col.isCustom && (
                  <span className="text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded">custom</span>
                )}
                <div
                  className="flex items-center gap-1 flex-shrink-0"
                  title="Expandable textarea"
                  onClick={(e) => e.stopPropagation()}
                  onMouseDown={(e) => e.stopPropagation()}
                  onDragStart={(e) => { e.stopPropagation(); e.preventDefault(); }}
                  draggable={false}
                >
                  <Checkbox
                    checked={col.isTextarea}
                    onCheckedChange={(checked) => {
                      const updated = displayColumns.map(c =>
                        c.key === col.key ? { ...c, isTextarea: !!checked } : c
                      );
                      saveColumnOrder(updated);
                      setLocalColumns(null);
                    }}
                    className="h-3.5 w-3.5"
                  />
                  <TextCursorInput className="h-3 w-3 text-muted-foreground" />
                </div>
                <Input
                  type="number"
                  min={0}
                  value={col.width || ''}
                  placeholder="auto"
                  onClick={(e) => e.stopPropagation()}
                  onMouseDown={(e) => e.stopPropagation()}
                  onDragStart={(e) => { e.stopPropagation(); e.preventDefault(); }}
                  draggable={false}
                  onChange={(e) => {
                    const w = parseInt(e.target.value) || 0;
                    const updated = displayColumns.map(c =>
                      c.key === col.key ? { ...c, width: w } : c
                    );
                    setLocalColumns(updated);
                  }}
                  onBlur={() => {
                    if (localColumns) {
                      saveColumnOrder(localColumns);
                      setLocalColumns(null);
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && localColumns) {
                      saveColumnOrder(localColumns);
                      setLocalColumns(null);
                    }
                  }}
                  className="w-16 h-6 text-xs text-center px-1 flex-shrink-0"
                />
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
