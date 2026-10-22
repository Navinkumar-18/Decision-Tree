import type { AdvancedSettings } from './types'

export const DEFAULT_SETTINGS: AdvancedSettings = {
  problem_type: 'auto',
  max_depth: 4,
  min_samples_leaf: 1,
  criterion: 'gini',
  n_estimators: 10,
  max_features: 'sqrt',
}
