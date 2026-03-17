import { PipelineStatus } from '../types/pipelineStatus';
import { API_CONFIG } from '../../config';

export async function getPipelineStatus(): Promise<PipelineStatus> {
    const response = await fetch(`${API_CONFIG.baseURL}/stock_filtering_app/pipeline/status`);
    if (!response.ok) {
        if (response.status === 404) {
            throw new Error('No pipeline run yet. Start a screening to see live progress.');
        }
        throw new Error(`Failed to fetch pipeline status: ${response.statusText}`);
    }
    return response.json();
}