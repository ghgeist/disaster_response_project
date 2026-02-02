/**
 * API client for Model Information dashboard.
 * Fetches single payload from GET /api/model-info/dashboard.
 */

export interface DashboardModel {
  id: string;
  version: string;
  lastUpdated: string | null;
  status: string;
  generatedAt: string;
}

export interface DashboardMetrics {
  f1: number;
  precision: number;
  recall: number;
}

export interface DashboardCategory {
  key: string;
  label: string;
  f1: number;
  precision: number;
  recall: number;
  support: number;
}

export interface DashboardCriticalThreshold {
  key: string;
  label: string;
  threshold: number;
}

export interface DashboardRegistryItem {
  name: string;
  size: number;
  type: string;
}

export interface DashboardPayload {
  model: DashboardModel;
  metrics: DashboardMetrics;
  categories: DashboardCategory[];
  criticalThresholds: DashboardCriticalThreshold[];
  registry: DashboardRegistryItem[];
}

const DASHBOARD_API_URL = "/api/model-info/dashboard";

export async function fetchDashboard(): Promise<DashboardPayload> {
  const response = await fetch(DASHBOARD_API_URL);
  if (!response.ok) {
    throw new Error(`Dashboard API error: ${response.status}`);
  }
  const data = await response.json();
  if (!data || typeof data !== "object") {
    throw new Error("Invalid dashboard payload");
  }
  return data as DashboardPayload;
}
