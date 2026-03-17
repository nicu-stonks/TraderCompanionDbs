import { useState } from 'react'
import { rankingBoxesApi } from '../services/rankingBoxes'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Plus, Minus, Columns, Trash2 } from 'lucide-react'
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
} from "@/components/ui/alert-dialog"
import { Card, CardContent } from '@/components/ui/card'

type ConfigurationBoxProps = {
  columnCount: number
  onColumnCountChange: (count: number) => void
  onRankingBoxCreated?: () => void
  onDeleteAllBoxes?: () => void
}

export function ConfigurationBox({ 
  columnCount, 
  onColumnCountChange,
  onRankingBoxCreated,
  onDeleteAllBoxes
}: ConfigurationBoxProps) {
  const [title, setTitle] = useState('')
  const [isCreating, setIsCreating] = useState(false)
  const [error, setError] = useState<string>()

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim()) return

    try {
      setIsCreating(true)
      setError(undefined)
      await rankingBoxesApi.createRankingBox(title.trim())
      setTitle('')
      onRankingBoxCreated?.()
    } catch (err) {
      setError('Failed to create ranking box')
      console.error(err)
    } finally {
      setIsCreating(false)
    }
  }

  const incrementColumns = () => {
    if (columnCount < 6) {
      onColumnCountChange(columnCount + 1)
    }
  }

  const decrementColumns = () => {
    if (columnCount > 1) {
      onColumnCountChange(columnCount - 1)
    }
  }

  return (
    <Card className="h-full bg-card">
      <CardContent className="flex items-center gap-4 p-4">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5 text-sm font-medium text-muted-foreground">
            <Columns className="h-4 w-4" />
            {columnCount}
          </span>
          
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={decrementColumns}
              disabled={columnCount <= 1}
              className="h-8 w-8 p-0"
            >
              <Minus className="h-4 w-4" />
            </Button>
            
            <div className="w-24">
              <Slider 
                value={[columnCount]}
                min={1}
                max={6}
                step={1}
                onValueChange={([value]) => onColumnCountChange(value)}
                className="cursor-pointer"
              />
            </div>
            
            <Button
              variant="outline"
              size="sm"
              onClick={incrementColumns}
              disabled={columnCount >= 6}
              className="h-8 w-8 p-0"
            >
              <Plus className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <form onSubmit={handleCreate} className="flex flex-1 items-center gap-2">
          <Input
            type="text"
            value={title}
            onChange={e => setTitle(e.target.value)}
            placeholder="Enter industry name"
            className="h-9"
            disabled={isCreating}
          />
          <Button 
            type="submit"
            size="sm"
            disabled={isCreating || !title.trim()}
            className="h-9"
          >
            <Plus className="mr-1 h-4 w-4" />
            Add Box
          </Button>
        </form>

        {onDeleteAllBoxes && (
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="h-9 text-muted-foreground hover:text-destructive hover:border-destructive"
              >
                <Trash2 className="mr-1 h-4 w-4" />
                Delete All Boxes
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete All Ranking Boxes</AlertDialogTitle>
                <AlertDialogDescription>
                  Are you sure you want to delete all ranking boxes and their associated stock picks? This action cannot be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  className="bg-destructive hover:bg-destructive/90"
                  onClick={onDeleteAllBoxes}
                >
                  Delete All Boxes
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}

        {error && (
          <Alert variant="destructive" className="mt-0">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  )
}