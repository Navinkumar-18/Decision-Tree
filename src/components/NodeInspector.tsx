import { useState } from 'react'
import { ChevronRight, Info, X } from 'lucide-react'
import type { ProblemType, TreeNode } from '../types'
import { formatNumber } from '../utils/format'

interface Props {
  node: TreeNode | null
  problemType: ProblemType
  targetColumn: string
  onClose: () => void
}

function DistributionBars({ distribution }: { distribution: Record<string, number> }) {
  const entries = Object.entries(distribution)
  const total = entries.reduce((sum, [, count]) => sum + count, 0) || 1
  const sorted = [...entries].sort((a, b) => b[1] - a[1]).slice(0, 5)
  return (
    <div className="space-y-1.5">
      {sorted.map(([label, count]) => {
        const share = count / total
        return (
          <div key={label}>
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-medium text-slate-600">{label}</span>
              <span className="text-slate-400">
                {count} ({(share * 100).toFixed(0)}%)
              </span>
            </div>
            <div className="mt-0.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full bg-indigo-400"
                style={{ width: `${Math.max(share * 100, 2)}%` }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

const NodeInspector = ({ node, problemType, targetColumn, onClose }: Props) => {
  const [showTechnical, setShowTechnical] = useState(false)

  if (!node) return null

  const isLeaf = node.is_leaf
  const feature = node.feature_display ?? node.feature ?? 'feature'

  return (
    <aside className="flex w-full flex-col rounded-xl border border-slate-200 bg-white xl:w-[330px] xl:shrink-0">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <div className="flex items-center gap-2">
          <Info className="h-4 w-4 text-indigo-500" />
          <h3 className="text-sm font-semibold text-slate-800">
            {isLeaf ? 'Leaf node' : 'Decision node'}
          </h3>
          <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-500">
            #{node.id}
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
          aria-label="Close inspector"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="space-y-4 px-4 py-4">
        {isLeaf ? (
          <>
            <div>
              <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                Prediction
              </p>
              <p className="mt-1 text-lg font-bold text-slate-900">
                {String(node.prediction ?? '—')}
              </p>
            </div>
            <p className="text-xs leading-relaxed text-slate-500">
              This is the end of the decision path. The model predicts{' '}
              <span className="font-semibold text-slate-700">{String(node.prediction ?? '—')}</span>
              {' for '}
              records that reach this leaf
              {problemType === 'classification' ? ` of “${targetColumn}”.` : '.'}
            </p>
            <div>
              <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-slate-400">
                {problemType === 'classification' ? 'Class distribution' : 'Average value'}
              </p>
              {problemType === 'classification' && node.class_distribution ? (
                <DistributionBars distribution={node.class_distribution} />
              ) : (
                <p className="text-sm text-slate-600">
                  {formatNumber(node.prediction_value ?? undefined)}
                </p>
              )}
            </div>
          </>
        ) : (
          <>
            <div>
              <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                Split condition
              </p>
              <p className="mt-1 rounded-lg bg-indigo-50 px-2.5 py-2 font-mono text-[13px] font-medium text-indigo-800">
                {node.condition}
              </p>
            </div>
            <p className="text-xs leading-relaxed text-slate-500">
              The model uses{' '}
              <span className="font-semibold text-slate-700">{feature}</span> at this point to
              separate the records into two groups.
            </p>
            <div className="space-y-1.5">
              <div className="rounded-lg border border-slate-200 px-2.5 py-2">
                <p className="text-[11px] font-semibold text-slate-500">
                  Left branch · {node.left_label ?? 'Yes'}
                </p>
                <p className="mt-0.5 text-xs text-slate-700">{node.left_rule}</p>
              </div>
              <div className="rounded-lg border border-slate-200 px-2.5 py-2">
                <p className="text-[11px] font-semibold text-slate-500">
                  Right branch · {node.right_label ?? 'No'}
                </p>
                <p className="mt-0.5 text-xs text-slate-700">{node.right_rule}</p>
              </div>
            </div>
            {problemType === 'classification' && node.class_distribution && (
              <div>
                <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-slate-400">
                  Class distribution here
                </p>
                <DistributionBars distribution={node.class_distribution} />
              </div>
            )}
          </>
        )}

        <div>
          <p className="text-xs text-slate-500">
            <span className="font-semibold text-slate-700">{node.samples}</span> samples reach
            this node.
          </p>
        </div>

        <div>
          <button
            type="button"
            onClick={() => setShowTechnical((v) => !v)}
            className="flex items-center gap-1 text-[11px] font-semibold text-slate-400 hover:text-indigo-600"
          >
            <ChevronRight className={`h-3 w-3 transition ${showTechnical ? 'rotate-90' : ''}`} />
            Technical details
          </button>
          {showTechnical && (
            <dl className="mt-2 space-y-1 rounded-lg bg-slate-50 px-3 py-2 text-[11px]">
              <div className="flex justify-between">
                <dt className="text-slate-500">Impurity</dt>
                <dd className="font-mono text-slate-700">{formatNumber(node.impurity, 4)}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">Depth</dt>
                <dd className="font-mono text-slate-700">{node.depth}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">Node id</dt>
                <dd className="font-mono text-slate-700">{node.id}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">Parent node</dt>
                <dd className="font-mono text-slate-700">
                  {node.parent >= 0 ? `#${node.parent}` : 'root'}
                </dd>
              </div>
              {!isLeaf && (
                <>
                  <div className="flex justify-between">
                    <dt className="text-slate-500">Left child</dt>
                    <dd className="font-mono text-slate-700">#{node.children_left}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-500">Right child</dt>
                    <dd className="font-mono text-slate-700">#{node.children_right}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-500">Raw threshold</dt>
                    <dd className="font-mono text-slate-700">
                      {formatNumber(node.threshold, 4)}
                    </dd>
                  </div>
                </>
              )}
            </dl>
          )}
        </div>
      </div>
    </aside>
  )
}

export default NodeInspector
