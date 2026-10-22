import axios, { AxiosError, type AxiosRequestConfig } from 'axios'
import type {
  DatasetInfo,
  DatasetPreview,
  ForestTreesResponse,
  GenerateModelBody,
  ModelResult,
  PredictionResult,
  ProblemTypeHint,
  SampleDataset,
  TreeStructure,
} from '../types'

// Deployment: set VITE_API_BASE_URL (build-time) to the deployed backend origin,
// e.g. https://treeviz-api.onrender.com
// Local development: leave unset — requests go to the same origin on :3000 and
// Vite's dev proxy (vite.config.ts, server.proxy) forwards them to :8000.
const rawBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? ''
export const apiBaseUrl = rawBase.replace(/\/+$/, '')

const client = axios.create({
  baseURL: `${apiBaseUrl}/api`,
  timeout: 120_000,
})

export const api = {
  health: (): Promise<{ status: string }> =>
    client.get('/health').then((r) => r.data),

  listSamples: (): Promise<SampleDataset[]> =>
    client.get('/datasets').then((r) => r.data.samples),

  uploadDataset: (file: File): Promise<DatasetInfo> => {
    const form = new FormData()
    form.append('file', file)
    return client.post('/datasets/upload', form).then((r) => r.data)
  },

  loadSample: (sampleId: string): Promise<DatasetInfo> =>
    client.post(`/datasets/sample/${sampleId}`).then((r) => r.data),

  previewDataset: (datasetId: string, limit = 10): Promise<DatasetPreview> =>
    client
      .get(`/datasets/${datasetId}/preview`, { params: { limit } })
      .then((r) => r.data),

  getProblemType: (datasetId: string, targetColumn: string): Promise<ProblemTypeHint> =>
    client
      .get(`/datasets/${datasetId}/problem-type`, { params: { target_column: targetColumn } })
      .then((r) => r.data),

  generateModel: (body: GenerateModelBody, config?: AxiosRequestConfig): Promise<ModelResult> =>
    client.post('/models/train', body, config).then((r) => r.data),

  getTree: (modelId: string): Promise<TreeStructure> =>
    client.get(`/models/${modelId}/tree`).then((r) => r.data),

  getForestTrees: (modelId: string): Promise<ForestTreesResponse> =>
    client.get(`/models/${modelId}/forest/trees`).then((r) => r.data),

  getForestTree: (modelId: string, treeIndex: number): Promise<TreeStructure> =>
    client.get(`/models/${modelId}/forest/trees/${treeIndex}`).then((r) => r.data),

  predict: (modelId: string, features: Record<string, unknown>): Promise<PredictionResult> =>
    client.post(`/models/${modelId}/predict`, { features }).then((r) => r.data),

  decisionPath: (
    modelId: string,
    features: Record<string, unknown>,
    treeIndex?: number,
  ): Promise<PredictionResult> =>
    client
      .post(`/models/${modelId}/decision-path`, { features, tree_index: treeIndex ?? null })
      .then((r) => r.data),
}

export function errorMessage(error: unknown): string {
  if (axios.isCancel(error)) return 'Request cancelled.'
  const axiosError = error as AxiosError<{ detail?: string }>
  const detail = axiosError?.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail
  if (axiosError?.message === 'Network Error') {
    const target = apiBaseUrl || window.location.origin
    return `Cannot reach the backend at ${target}. Make sure it is running and VITE_API_BASE_URL is set (see README).`
  }
  if (axiosError?.message?.includes('timeout')) return 'The request timed out. Please try again.'
  return axiosError?.message || 'Something went wrong. Please try again.'
}

export function isCancelled(error: unknown): boolean {
  return axios.isCancel(error)
}
