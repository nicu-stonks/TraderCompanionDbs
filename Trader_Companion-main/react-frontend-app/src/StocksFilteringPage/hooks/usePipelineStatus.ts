// hooks/usePipelineStatus.ts

import { useState, useEffect } from 'react';
import { PipelineStatus } from '../types/pipelineStatus';
import { getPipelineStatus } from '../services/pipelineService';

interface UsePipelineStatusOptions {
    pollingInterval?: number;
}

export function usePipelineStatus({ pollingInterval = 1000 }: UsePipelineStatusOptions = {}) {
    const [status, setStatus] = useState<PipelineStatus | null>(null);
    const [error, setError] = useState<Error | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        let isSubscribed = true;

        const fetchStatus = async () => {
            try {
                const data = await getPipelineStatus();
                if (isSubscribed) {
                    setStatus(data);
                    setError(null);
                }
            } catch (err) {
                if (isSubscribed) {
                    setError(err instanceof Error ? err : new Error('Failed to fetch pipeline status'));
                }
            } finally {
                if (isSubscribed) {
                    setIsLoading(false);
                }
            }
        };

        // Initial fetch
        fetchStatus();

        // Poll at the specified interval
        const intervalId = setInterval(fetchStatus, pollingInterval);

        // Cleanup function
        return () => {
            isSubscribed = false;
            clearInterval(intervalId);
        };
    }, [pollingInterval]);

    return { status, error, isLoading };
}