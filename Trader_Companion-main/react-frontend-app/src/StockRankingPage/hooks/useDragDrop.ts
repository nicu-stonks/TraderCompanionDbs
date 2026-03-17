// hooks/useDragDrop.ts
import { useState } from 'react';
import { RankingBox } from '../types';
import { cn } from '@/lib/utils';

export const useDragDrop = (
  rankingBoxes: RankingBox[],
  onReorder: (newOrder: RankingBox[]) => void
) => {
  const [draggedBox, setDraggedBox] = useState<RankingBox | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragStart = (e: React.DragEvent<HTMLDivElement>, box: RankingBox) => {
    setDraggedBox(box);
    setIsDragging(true);
    e.currentTarget.classList.add('opacity-50');
  };

  const handleDragEnd = (e: React.DragEvent<HTMLDivElement>) => {
    e.currentTarget.classList.remove('opacity-50');
    setDraggedBox(null);
    setIsDragging(false);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!isDragging) return;
    
    e.currentTarget.classList.add(
      'ring-2',
      'ring-primary',
      'ring-offset-2',
      'ring-offset-background'
    );
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.currentTarget.classList.remove(
      'ring-2',
      'ring-primary',
      'ring-offset-2',
      'ring-offset-background'
    );
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>, targetBox: RankingBox) => {
    e.preventDefault();
    e.currentTarget.classList.remove(
      'ring-2',
      'ring-primary',
      'ring-offset-2',
      'ring-offset-background'
    );
    
    if (!draggedBox || draggedBox.id === targetBox.id) return;

    const newBoxes = [...rankingBoxes];
    const draggedIdx = newBoxes.findIndex(box => box.id === draggedBox.id);
    const targetIdx = newBoxes.findIndex(box => box.id === targetBox.id);

    newBoxes.splice(draggedIdx, 1);
    newBoxes.splice(targetIdx, 0, draggedBox);

    onReorder(newBoxes);
  };

  const getDragProps = (box: RankingBox) => ({
    draggable: true,
    onDragStart: (e: React.DragEvent<HTMLDivElement>) => handleDragStart(e, box),
    onDragEnd: handleDragEnd,
    onDragOver: handleDragOver,
    onDragLeave: handleDragLeave,
    onDrop: (e: React.DragEvent<HTMLDivElement>) => handleDrop(e, box),
    className: cn(
      'break-inside-avoid mb-1',
      'transition-all duration-200',
      'cursor-move',
      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
      isDragging && 'opacity-75'
    )
  });

  return {
    isDragging,
    getDragProps
  };
};