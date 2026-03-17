const API_BASE = '/api/nutrients_tracker';

export interface LLMConfig {
  model_name: string;
  api_key: string;
  user_profile?: string;
  available_models?: string;
}

export interface Supplement {
  id?: number;
  name: string;
  details: string;
}

export interface FoodItem {
  id?: number;
  name: string;
  details: string;
}

export interface ChatMessage {
  role: 'user' | 'model';
  parts: { text: string }[];
}

interface RollbackRecordResponse {
  date: string;
  foods_eaten: string;
  supplements_taken: any[];
  nutrient_completion: Record<string, number>;
  nutrient_sources: Record<string, string[]>;
  recommendations: string;
  chat_history: ChatMessage[];
}

export interface RollbackResponse {
  discarded_messages: number;
  history_length: number;
  record: RollbackRecordResponse;
}

export interface DailyRecord {
  date: string;
  foods_eaten: string;
  supplements_taken: any[];
  nutrient_completion: Record<string, number>;
  nutrient_sources: Record<string, string[]>;
  recommendations: string;
  chat_history: ChatMessage[];
}

export const fetchConfig = async (): Promise<LLMConfig> => {
  const res = await fetch(`${API_BASE}/config/`);
  return res.json();
};

export const saveConfig = async (config: LLMConfig): Promise<LLMConfig> => {
  const res = await fetch(`${API_BASE}/config/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  return res.json();
};

export const fetchSupplements = async (): Promise<Supplement[]> => {
  const res = await fetch(`${API_BASE}/supplements/`);
  return res.json();
};

export const addSupplement = async (supp: Supplement): Promise<Supplement> => {
  const res = await fetch(`${API_BASE}/supplements/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(supp),
  });
  return res.json();
};

export const deleteSupplement = async (id: number): Promise<void> => {
  await fetch(`${API_BASE}/supplements/${id}/`, { method: 'DELETE' });
};

export const fetchFoods = async (): Promise<FoodItem[]> => {
  const res = await fetch(`${API_BASE}/foods/`);
  return res.json();
};

export const addFood = async (food: FoodItem): Promise<FoodItem> => {
  const res = await fetch(`${API_BASE}/foods/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(food),
  });
  return res.json();
};

export const deleteFood = async (id: number): Promise<void> => {
  await fetch(`${API_BASE}/foods/${id}/`, { method: 'DELETE' });
};

export const fetchDailyRecords = async (): Promise<DailyRecord[]> => {
  const res = await fetch(`${API_BASE}/daily_records/`);
  return res.json();
};

export const fetchDailyRecord = async (date: string): Promise<DailyRecord> => {
  const res = await fetch(`${API_BASE}/daily_records/${date}/`);
  if (!res.ok) throw new Error("Record not found");
  return res.json();
};

export const deleteDailyRecord = async (date: string): Promise<void> => {
  const res = await fetch(`${API_BASE}/daily_records/${date}/`, { method: 'DELETE' });
  if (!res.ok) throw new Error("Failed to delete record");
};

export const sendChat = async (date: string, prompt: string, model_name?: string): Promise<DailyRecord> => {
  const bodyData: any = { prompt };
  if (model_name) bodyData.model_name = model_name;

  const res = await fetch(`${API_BASE}/chat/${date}/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(bodyData),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || "Failed to send chat");
  }
  return res.json();
};

export const getChatPrompt = async (date: string, prompt: string): Promise<{ full_prompt: string }> => {
  const res = await fetch(`${API_BASE}/chat/${date}/prompt/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || "Failed to get prompt");
  }
  return res.json();
};

export const submitManualChat = async (date: string, prompt: string, model_response: string): Promise<DailyRecord> => {
  const res = await fetch(`${API_BASE}/chat/${date}/manual/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, model_response }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || "Failed to submit manual chat");
  }
  return res.json();
};

export const rollbackChatBeforeMessage = async (date: string, before_message_index: number): Promise<RollbackResponse> => {
  const res = await fetch(`${API_BASE}/chat/${date}/rollback/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ before_message_index }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || "Failed to rollback chat");
  }
  return res.json();
};
