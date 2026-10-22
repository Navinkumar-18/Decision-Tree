import { useMemo, useState } from 'react'
import { ChevronRight, FlaskConical, Loader2, Route, X } from 'lucide-react'
import { api, errorMessage } from '../services/api'
import type { DatasetInfo, ModelResult, PredictionResult } from '../types'
import { formatNumber, formatPercent } from '../utils/format'

interface Props {
  open: boolean
  dataset: DatasetInfo
  model: ModelResult
  selectedTree: number
  onClose: () => void
  onPath: (path: number[], edges: [number, number][]) => void
  onClearPath: () => void
}

const inputClass =
  'w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-800 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100'

const PredictionExplorer = ({
  open,
  dataset,
  model,
  selectedTree,
  onClose,
  onPath,
  onClearPath,
}: Props) => {
  const [values, setValues] = useState<Record<string, string>>({})
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const fields = useMemo(
    () =>
      dataset.column_names.filter((column) => column !== model.target_column),
    [dataset.column_names, model.target_column],
  )

  if (!open) return null

  const submit = async () => {
    setSubmitting(true)
    setError(null)
    try {
      const features: Record<string, unknown> = {}
      for (const field of fields) {
        const raw = values[field] ?? ''
        features[field] =
          dataset.column_kinds[field] === 'numeric'
            ? raw === ''
              ? null
              : Number(raw)
            : raw === ''
              ? null
              : raw
      }
      const pathResult = await api.decisionPath(
        model.model_id,
        features,
        model.type === 'random_forest' ? selectedTree : undefined,
      )
      setResult(pathResult)
      onPath(pathResult.path ?? [], (pathResult.edges as [number, number][]) ?? [])
    } catch (err) {
      setResult(null)
      onClearPath()
      setError(errorMessage(err))
    } finally {
      setSubmitting(false)
    }
  }

  const isForest = model.type === 'random_forest'
  const probabilities = result?.probabilities
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : []

  return (
    <div className="rounded-xl border border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <div className="flex items-center gap-2">
          <FlaskConical className="h-4 w-4 text-indigo-500" />
          <h3 className="text-sm font-semibold text-slate-800">Try a Prediction</h3>
          <span className="text-[11px] text-slate-400">
            enter values and follow the decision path
          </span>
        </div>
        <button
          type="button"
          onClick={() => {
            onClearPath()
            setResult(null)
            setError(null)
            onClose()
          }}
          className="rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="px-4 py-4">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {fields.map((field) => {
            const kind = dataset.column_kinds[field]
            const options = dataset.categories[field]
            return (
              <label key={field} className="block">
                <span className="mb-1 flex items-center justify-between text-[11px] font-medium text-slate-500">
                  {field}
                  <span className="font-normal text-slate-300">
                    {kind === 'numeric' ? 'number' : 'category'}
                  </span>
                </span>
                {kind === 'numeric' ? (
                  <input
                    type="number"
                    step="any"
                    className={inputClass}
                    value={values[field] ?? ''}
                    placeholder={dataset.missing_values[field] ? 'empty = missing' : 'e.g. 42'}
                    onChange={(event) =>
                      setValues((prev) => ({ ...prev, [field]: event.target.value }))
                    }
                  />
                ) : (
                  <select
                    className={inputClass}
                    value={values[field] ?? ''}
                    onChange={(event) =>
                      setValues((prev) => ({ ...prev, [field]: event.target.value }))
                    }
                  >
                    <option value="">— leave empty (missing) —</option>
                    {(options ?? []).map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                )}
              </label>
            )
          })}
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <button
            type="button"
            disabled={submitting}
            onClick={submit}
            className="flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:opacity-60"
          >
            {submitting ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Route className="h-4 w-4" />
            )}
            Predict &amp; trace path
          </button>
          {result && (
            <button
              type="button"
              onClick={() => {
                onClearPath()
                setResult(null)
              }}
              className="text-xs font-medium text-slate-400 hover:text-indigo-600"
            >
              Clear highlight
            </button>
          )}
          {isForest && (
            <span className="text-[11px] text-slate-400">
              Path shown through <strong className="text-slate-600">Tree {selectedTree + 1}</strong>
            </span>
          )}
        </div>

        {error && (
          <p className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-600">
            {error}
          </p>
        )}

        {result && (
          <div className="mt-4 space-y-4">
            <div className="flex flex-wrap items-center gap-4 rounded-xl border border-indigo-100 bg-indigo-50 px-4 py-3">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-wide text-indigo-500">
                  {isForest ? 'Ensemble prediction' : 'Prediction'}
                </p>
                <p className="text-xl font-bold text-indigo-900">{result.prediction}</p>
              </div>
              {isForest && result.tree_prediction !== undefined && (
                <>
                  <span className="h-8 w-px bg-indigo-200" />
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-wide text-indigo-400">
                      Tree {selectedTree + 1} alone says
                    </p>
                    <p className="text-lg font-semibold text-indigo-800">
                      {result.tree_prediction}
                    </p>
                  </div>
                </>
              )}
              <div className="ml-auto text-right text-[11px] leading-relaxed text-indigo-500">
                predicting <strong>{model.target_column}</strong>
              </div>
            </div>

            {probabilities.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                  Class probabilities
                </p>
                {probabilities.map(([label, probability]) => (
                  <div key={label} className="flex items-center gap-2 text-xs">
                    <span className="w-28 truncate font-medium text-slate-600">{label}</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
                      <div
                        className="h-full rounded-full bg-indigo-400"
                        style={{ width: `${probability * 100}%` }}
                      />
                    </div>
                    <span className="w-12 text-right text-slate-400">
                      {formatPercent(probability)}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {isForest && result.tree_votes && (
              <div className="space-y-1.5">
                <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                  Individual tree votes
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(result.tree_votes).map(([label, votes]) => (
                    <span
                      key={label}
                      className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${
                        label === String(result.prediction)
                          ? 'bg-emerald-100 text-emerald-700'
                          : 'bg-slate-100 text-slate-500'
                      }`}
                    >
                      {label}: {votes} {votes === 1 ? 'vote' : 'votes'}
                    </span>
                  ))}
                </div>
                {result.note && (
                  <p className="text-[11px] leading-relaxed text-slate-400">{result.note}</p>
                )}
              </div>
            )}

            {result.rules && result.rules.length > 0 && (
              <div>
                <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                  Decision path
                </p>
                <ol className="space-y-1.5">
                  {result.rules.map((rule, index) => (
                    <li
                      key={`${rule.node_id}-${index}`}
                      className="flex items-start gap-2 rounded-lg border border-slate-200 px-3 py-2 text-xs"
                    >
                      <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-[10px] font-bold text-indigo-600">
                        {index + 1}
                      </span>
                      <span className="text-slate-600">
                        <ChevronRight className="mr-0.5 inline h-3 w-3 text-slate-300" />
                        {rule.condition}{' '}
                        <span className="font-semibold text-slate-800">→ {rule.branch}</span>
                      </span>
                    </li>
                  ))}
                  <li className="flex items-start gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs">
                    <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-[10px] font-bold text-emerald-700">
                      ✓
                    </span>
                    <span className="text-emerald-800">
                      Ends in leaf node #{result.leaf_node_id} — model predicts{' '}
                      <strong>{result.prediction}</strong>
                    </span>
                  </li>
                </ol>
                {isForest && (
                  <p className="mt-2 text-[11px] text-slate-400">
                    This path explains a single tree. The ensemble prediction (
                    <strong className="text-slate-600">{result.prediction}</strong>) combines all{' '}
                    {model.n_estimators} trees — individual trees may disagree.
                  </p>
                )}
              </div>
            )}

            {result.rules && result.rules.length === 0 && (
              <p className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500">
                The root node is already a leaf — every record receives the same prediction{' '}
                <strong>{result.prediction}</strong> ({formatNumber(result.prediction as never)}).
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default PredictionExplorer
