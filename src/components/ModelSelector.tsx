import { Check, GitBranch, Network } from 'lucide-react'
import type { ModelType } from '../types'

interface Props {
  value: ModelType
  onChange: (modelType: ModelType) => void
}

const OPTIONS: {
  id: ModelType
  title: string
  description: string
  icon: typeof Network
}[] = [
  {
    id: 'decision_tree',
    title: 'Decision Tree',
    description: 'Explore one decision-making tree.',
    icon: Network,
  },
  {
    id: 'random_forest',
    title: 'Random Forest',
    description: 'Explore multiple trees working together.',
    icon: GitBranch,
  },
]

const ModelSelector = ({ value, onChange }: Props) => (
  <div className="rounded-xl border border-slate-200 bg-white p-4">
    <div className="mb-3 flex items-center gap-2">
      <Network className="h-4 w-4 text-indigo-500" />
      <h3 className="text-sm font-semibold text-slate-800">Model type</h3>
    </div>
    <div className="space-y-2">
      {OPTIONS.map((option) => {
        const active = value === option.id
        const Icon = option.icon
        return (
          <button
            key={option.id}
            type="button"
            onClick={() => onChange(option.id)}
            className={`flex w-full items-start gap-2.5 rounded-lg border px-3 py-2.5 text-left transition ${
              active
                ? 'border-indigo-400 bg-indigo-50 ring-1 ring-indigo-300'
                : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
            }`}
          >
            <Icon
              className={`mt-0.5 h-4 w-4 shrink-0 ${active ? 'text-indigo-600' : 'text-slate-400'}`}
            />
            <span className="min-w-0 flex-1">
              <span className="flex items-center justify-between gap-2">
                <span
                  className={`text-[13px] font-semibold ${active ? 'text-indigo-700' : 'text-slate-700'}`}
                >
                  {option.title}
                </span>
                {active && (
                  <span className="flex h-4 w-4 items-center justify-center rounded-full bg-indigo-600 text-white">
                    <Check className="h-3 w-3" />
                  </span>
                )}
              </span>
              <span className="mt-0.5 block text-[11px] text-slate-500">
                {option.description}
              </span>
            </span>
          </button>
        )
      })}
    </div>
  </div>
)

export default ModelSelector
