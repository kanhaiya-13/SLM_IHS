import type {
  StatesListResponse,
  HistoryResponse,
  ForecastResponse,
  CustomForecastRequest,
  ModelInfoResponse,
} from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
  });

  if (!response.ok) {
    let errorMsg = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errData = await response.json();
      if (errData.detail) {
        errorMsg = typeof errData.detail === "string" ? errData.detail : JSON.stringify(errData.detail);
      }
    } catch {
      // Ignore JSON parse error and use statusText
    }
    throw new Error(errorMsg);
  }

  return response.json();
}

export const api = {
  async getStates(): Promise<StatesListResponse> {
    return fetchJson<StatesListResponse>(`${API_BASE_URL}/states`);
  },

  async getHistory(state: string, weeks: number = 24): Promise<HistoryResponse> {
    return fetchJson<HistoryResponse>(
      `${API_BASE_URL}/history/${encodeURIComponent(state)}?weeks=${weeks}`
    );
  },

  async getForecast(state: string): Promise<ForecastResponse> {
    return fetchJson<ForecastResponse>(
      `${API_BASE_URL}/forecast/${encodeURIComponent(state)}`
    );
  },

  async postCustomForecast(request: CustomForecastRequest): Promise<ForecastResponse> {
    return fetchJson<ForecastResponse>(`${API_BASE_URL}/forecast/custom`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  },

  async getModelInfo(state: string): Promise<ModelInfoResponse> {
    return fetchJson<ModelInfoResponse>(
      `${API_BASE_URL}/model-info/${encodeURIComponent(state)}`
    );
  },

  async checkHealth(): Promise<{ status: string; models_loaded: boolean }> {
    return fetchJson<{ status: string; models_loaded: boolean }>(`${API_BASE_URL}/health`);
  },
};
