import type { ReactNode } from 'react'
import { ChevronDown, RotateCcw, SlidersHorizontal } from 'lucide-react'
import { DEFAULT_SETTINGS } from '../constants'
import type { AdvancedSettings, ModelType, ProblemTypeChoice } from '../types'

interface Props {
  open: boolean
  modelType: ModelType
  settings: AdvancedSettings
  onToggle: () => void
  onChange: (settings: AdvancedSettings) => void
}

const fieldClass =
  'w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-800 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100'

const Label = ({ children }: { children: ReactNode }) => (
  <span className="mb-1 block text-[11px] font-medium text-slate-500">{children}</span>
)

const AdvancedSettingsPanel = ({ open, modelType, settings, onToggle, onChange }: Props) => {
  const isClassification = settings.problem_type !== 'regression'
  const criteria = isClassification
    ? [
        { value: 'gini', label: 'Gini impurity' },
        { value: 'entropy', label: 'Entropy' },
        { value: 'log_loss', label: 'Log loss' },
      ]
    : [
        { value: 'squared_error', label: 'Squared error' },
        { value: 'absolute_error', label: 'Absolute error' },
        { value: 'friedman_mse', label: 'Friedman MSE' },
      ]
  const criterionValue = criteria.some((c) => c.value === settings.criterion)
    ? settings.criterion
    : criteria[0].value

  const update = (patch: Partial<AdvancedSettings>) => onChange({ ...settings, ...patch })

  return (
    <div className="rounded-xl border border-slate-200 bg-white">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between px-4 py-3"
      >
        <span className="flex items-center gap-2">
          <SlidersHorizontal className="h-4 w-4 text-slate-400" />
          <span className="text-sm font-semibold text-slate-700">Advanced Settings</span>
          <span className="text-[11px] font-normal text-slate-400">optional</span>
        </span>
        <ChevronDown
          className={`h-4 w-4 text-slate-400 transition ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {open && (
        <div className="border-t border-slate-100 px-4 py-4">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
            <div>
              <Label>Problem type</Label>
              <select
                className={fieldClass}
                value={settings.problem_type}
                onChange={(event) => {
                  const next = event.target.value as ProblemTypeChoice
                  update({
                    problem_type: next,
                    criterion:
                      next === 'regression' ? 'squared_error' : 'gini',
                  })
                }}
              >
                <option value="auto">Auto (detect)</option>
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>
            </div>

            <div>
              <Label>Maximum depth</Label>
              <input
                type="number"
                min={1}
                max={64}
                className={fieldClass}
                value={settings.max_depth}
                onChange={(event) =>
                  update({ max_depth: Math.max(1, Math.min(64, Number(event.target.value) || 1)) })
                }
              />
            </div>

            <div>
              <Label>Minimum samples per leaf</Label>
              <input
                type="number"
                min={1}
                max={100}
                className={fieldClass}
                value={settings.min_samples_leaf}
                onChange={(event) =>
                  update({
                    min_samples_leaf: Math.max(
                      1,
                      Math.min(100, Number(event.target.value) || 1),
                    ),
                  })
                }
              />
            </div>

            {modelType === 'decision_tree' && (
              <div>
                <Label>Criterion</Label>
                <select
                  className={fieldClass}
                  value={criterionValue}
                  onChange={(event) => update({ criterion: event.target.value })}
                >
                  {criteria.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {modelType === 'random_forest' && (
              <>
                <div>
                  <Label>Number of trees</Label>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    className={fieldClass}
                    value={settings.n_estimators}
                    onChange={(event) =>
                      update({
                        n_estimators: Math.max(
                          1,
                          Math.min(100, Number(event.target.value) || 1),
                        ),
                      })
                    }
                  />
                </div>
                <div>
                  <Label>Maximum features</Label>
                  <select
                    className={fieldClass}
                    value={settings.max_features}
                    onChange={(event) => update({ max_features: event.target.value })}
                  >
                    <option value="sqrt">Square root (√p)</option>
                    <option value="log2">Log2 (log₂p)</option>
                    <option value="all">All features</option>
                  </select>
                </div>
              </>
            )}
          </div>

          <div className="mt-3 flex items-center justify-between">
            <p className="text-[11px] text-slate-400">
              Changes regenerate the tree automatically after you finish editing.
            </p>
            <button
              type="button"
              onClick={() =>
                onChange({
                  ...DEFAULT_SETTINGS,
                  problem_type: settings.problem_type,
                  criterion:
                    settings.problem_type === 'regression' ? 'squared_error' : 'gini',
                })
              }
              className="flex items-center gap-1 text-[11px] font-medium text-slate-500 hover:text-indigo-600"
            >
              <RotateCcw className="h-3 w-3" /> Reset defaults
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default AdvancedSettingsPanel
