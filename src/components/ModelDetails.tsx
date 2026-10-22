import { ChevronDown } from 'lucide-react'
import type { ModelResult } from '../types'
import { formatNumber, formatPercent } from '../utils/format'

interface Props {
  model: ModelResult
  open: boolean
  onToggle: () => void
}

const Metric = ({ label, value }: { label: string; value: string }) => (
  <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
    <span className="text-xs text-slate-500">{label}</span>
    <span className="text-sm font-semibold text-slate-800">{value}</span>
  </div>
)

const ModelDetails = ({ model, open, onToggle }: Props) => {
  const isClassification = model.problem_type === 'classification'
  const importances = model.feature_importances
    .map((value, index) => ({ name: model.feature_names[index] ?? `feature ${index}`, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 6)
  const maxImportance = importances[0]?.value || 1

  return (
    <div className="rounded-xl border border-slate-200 bg-white">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between px-4 py-3"
      >
        <span className="text-sm font-semibold text-slate-700">Model Details</span>
        <ChevronDown className={`h-4 w-4 text-slate-400 transition ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="grid gap-6 border-t border-slate-100 px-4 py-4 md:grid-cols-2">
          <div className="space-y-2">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
              Evaluation ({model.eval_on === 'test' ? 'held-out test data' : 'training data'})
            </p>
            {isClassification ? (
              <>
                <Metric label="Accuracy" value={formatPercent(model.metrics.accuracy)} />
                <Metric label="Precision (weighted)" value={formatPercent(model.metrics.precision)} />
                <Metric label="Recall (weighted)" value={formatPercent(model.metrics.recall)} />
                <Metric label="F1 score (weighted)" value={formatPercent(model.metrics.f1_score)} />
              </>
            ) : (
              <>
                <Metric
                  label="Mean Absolute Error"
                  value={formatNumber(model.metrics.mae)}
                />
                <Metric
                  label="Root Mean Squared Error"
                  value={formatNumber(model.metrics.rmse)}
                />
                <Metric label="R² score" value={formatNumber(model.metrics.r2_score)} />
              </>
            )}
            <div className="flex items-center justify-between px-3 text-[11px] text-slate-400">
              <span>
                {model.n_train.toLocaleString()} train / {model.n_test.toLocaleString()} test
                samples
              </span>
            </div>
            {model.eval_note && (
              <p className="rounded-lg border border-amber-100 bg-amber-50 px-3 py-2 text-[11px] leading-relaxed text-amber-700">
                {model.eval_note}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
              Top feature importance
            </p>
            <div className="space-y-2">
              {importances.map((item) => (
                <div key={item.name}>
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="truncate font-medium text-slate-600" title={item.name}>
                      {item.name}
                    </span>
                    <span className="text-slate-400">{(item.value * 100).toFixed(1)}%</span>
                  </div>
                  <div className="mt-0.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                    <div
                      className="h-full rounded-full bg-indigo-400"
                      style={{ width: `${Math.max((item.value / maxImportance) * 100, 2)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ModelDetails
