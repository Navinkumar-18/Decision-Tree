export type ModelType = 'decision_tree' | 'random_forest'
export type ProblemType = 'classification' | 'regression'
export type ProblemTypeChoice = ProblemType | 'auto'

export interface SampleDataset {
  id: string
  name: string
  task: string
  description: string
  default_target: string
}

export interface DatasetInfo {
  dataset_id: string
  name: string
  rows: number
  columns: number
  column_names: string[]
  column_types: Record<string, string>
  column_kinds: Record<string, 'numeric' | 'categorical'>
  missing_values: Record<string, number>
  categories: Record<string, string[]>
  preview: Record<string, string | number | null>[]
}

export interface DatasetPreview {
  dataset_id: string
  name: string
  rows: number
  columns: number
  column_names: string[]
  column_kinds: Record<string, 'numeric' | 'categorical'>
  missing_values: Record<string, number>
  preview: Record<string, string | number | null>[]
}

export interface TreeNode {
  id: number
  type?: 'decision' | 'leaf'
  is_leaf: boolean
  impurity: number
  samples: number
  children_left: number
  children_right: number
  parent: number
  depth: number
  feature?: string
  feature_display?: string
  feature_kind?: 'numeric' | 'categorical'
  category?: string
  threshold?: number
  threshold_display?: number
  condition?: string
  left_rule?: string
  right_rule?: string
  left_label?: string
  right_label?: string
  prediction?: string | number | null
  prediction_value?: number
  class_distribution?: Record<string, number> | null
}

export interface TreeStructure {
  nodes: TreeNode[]
  n_nodes: number
  n_leaves: number
  max_depth: number
}

export interface Metrics {
  accuracy?: number | null
  precision?: number | null
  recall?: number | null
  f1_score?: number | null
  confusion_matrix?: number[][]
  mae?: number | null
  mse?: number | null
  rmse?: number | null
  r2_score?: number | null
}

export interface GenerateModelBody {
  dataset_id: string
  target_column: string
  algorithm: ModelType
  problem_type: ProblemTypeChoice
  max_depth?: number | null
  min_samples_leaf?: number
  n_estimators?: number
  criterion?: string | null
  max_features?: string | null
}

export interface ModelResult {
  model_id: string
  type: ModelType
  problem_type: ProblemType
  problem_type_detected: ProblemType
  target_column: string
  metrics: Metrics
  tree?: TreeStructure
  trees?: TreeStructure[]
  n_estimators?: number
  feature_importances: number[]
  feature_names: string[]
  target_names?: string[] | null
  feature_columns: string[]
  n_train: number
  n_test: number
  n_samples: number
  eval_on: 'train' | 'test'
  eval_note?: string | null
  config: Record<string, unknown>
}

export interface TreeOverview {
  index: number
  depth: number
  nodes: number
  leaves: number
}

export interface ForestTreesResponse {
  n_estimators: number
  trees: number[]
  overview: TreeOverview[]
}

export interface DecisionPathStep {
  node_id: number
  feature: string | null
  condition: string | null
  branch: string | null
  went_left: boolean
}

export interface PredictionResult {
  model_type: ModelType
  problem_type: ProblemType
  target_column: string
  prediction: string
  probabilities?: Record<string, number> | null
  tree_predictions?: (string | number)[]
  tree_votes?: Record<string, number>
  note?: string
  path?: number[]
  rules?: DecisionPathStep[]
  edges?: [number, number][]
  leaf_node_id?: number
  tree_index?: number | null
  tree_prediction?: string
}

export interface AdvancedSettings {
  problem_type: ProblemTypeChoice
  max_depth: number
  min_samples_leaf: number
  criterion: string
  n_estimators: number
  max_features: string
}

export type AppStatus = 'empty' | 'awaiting_target' | 'generating' | 'ready' | 'error'
