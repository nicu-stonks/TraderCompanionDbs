import React, { useState } from 'react';
import { usePipelineStatus } from '../hooks/usePipelineStatus';
import { XCircle, CheckCircle2, AlertCircle, Timer } from 'lucide-react';
import { API_CONFIG } from '../../config';
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Alert,
  AlertDescription,
} from "@/components/ui/alert";

interface PipelineStatusProps {
    pollingInterval?: number;
}

export const PipelineStatus: React.FC<PipelineStatusProps> = ({ 
    pollingInterval = 1000
}) => {
    const { status, error, isLoading } = usePipelineStatus({ pollingInterval });
    const [isStoppingPipeline, setIsStoppingPipeline] = useState(false);
    const [stopError, setStopError] = useState<string | null>(null);

    const stopPipeline = async () => {
        try {
            setIsStoppingPipeline(true);
            setStopError(null);
            const response = await fetch(`${API_CONFIG.baseURL}/stock_filtering_app/pipeline/stop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Failed to stop pipeline');
            }
        } catch (err) {
            setStopError(err instanceof Error ? err.message : 'Failed to stop pipeline');
        } finally {
            setIsStoppingPipeline(false);
        }
    };

    if (isLoading) {
        return <Alert><AlertDescription className="flex items-center"><Timer className="mr-2 h-4 w-4" />Loading pipeline status...</AlertDescription></Alert>;
    }

    if (error) {
        if (/No pipeline run yet/i.test(error.message)) {
            return (
                <Alert>
                    <AlertDescription className="text-sm">
                        <span className="font-semibold">No pipeline activity.</span> Start a screening from the Stock Screener Commander. When it begins, live status will appear here.
                    </AlertDescription>
                </Alert>
            );
        }
        return <Alert variant="destructive"><AlertDescription className="flex items-center"><AlertCircle className="mr-2 h-4 w-4" />{error.message}</AlertDescription></Alert>;
    }

    if (!status) return null;

    const getStatusBadgeVariant = (status: string): "default" | "destructive" | "secondary" | "outline" => {
        switch (status) {
            case 'completed':
                return 'secondary';
            case 'running':
                return 'default';
            case 'failed':
                return 'destructive';
            default:
                return 'outline';
        }
    };

    const formatTime = (timestamp: number) => {
        return new Date(timestamp * 1000).toLocaleTimeString();
    };

    return (
        <Card className="w-full">
            <CardContent className="pt-4">
                <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                        <h2 className="text-sm font-semibold">Pipeline Status</h2>
                        <Badge variant={getStatusBadgeVariant(status.status)}>
                            {status.status.toUpperCase()}
                        </Badge>
                    </div>
                    {status.status === 'running' && (
                        <Button
                            variant="destructive"
                            size="sm"
                            onClick={stopPipeline}
                            disabled={isStoppingPipeline}
                            className="h-7"
                        >
                            <XCircle className="h-4 w-4 mr-1" />
                            {isStoppingPipeline ? 'Stopping...' : 'Stop'}
                        </Button>
                    )}
                </div>

                {stopError && (
                    <Alert variant="destructive" className="mb-2 py-1">
                        <AlertDescription className="text-xs">{stopError}</AlertDescription>
                    </Alert>
                )}

                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-muted-foreground mb-2">
                    <span>Step: <span className="font-medium">{status.current_step}</span></span>
                    {status.current_batch && (
                        <span>Batch: {status.current_batch}/{status.total_batches}</span>
                    )}
                    <span>Started: {formatTime(status.start_time)}</span>
                    <span>Updated: {formatTime(status.last_updated)}</span>
                </div>

                <div className="flex flex-wrap gap-1">
                    {status.steps_completed.map((step) => (
                        <Badge 
                            key={step} 
                            variant="outline"
                            className="text-xs flex items-center gap-1"
                        >
                            <CheckCircle2 className="h-3 w-3 text-green-500" />
                            {step}
                        </Badge>
                    ))}
                </div>
            </CardContent>
        </Card>
    );
};