import { Target } from 'lucide-react'
import type { DatasetInfo } from '../types'

interface Props {
  dataset: DatasetInfo | null
  value: string
  onChange: (column: string) => void
}

const TargetSelector = ({ dataset, value, onChange }: Props) => (
  <div className="rounded-xl border border-slate-200 bg-white p-4">
    <div className="mb-3 flex items-center gap-2">
      <Target className="h-4 w-4 text-indigo-500" />
      <h3 className="text-sm font-semibold text-slate-800">What would you like to predict?</h3>
    </div>
    <select
      value={value}
      disabled={!dataset}
      onChange={(event) => onChange(event.target.value)}
      className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 disabled:cursor-not-allowed disabled:bg-slate-50"
    >
      <option value="">
        {dataset ? 'Select a target column...' : 'Upload a dataset first'}
      </option>
      {dataset?.column_names.map((column) => (
        <option key={column} value={column}>
          {column}
          {dataset.column_kinds[column] === 'numeric' ? '  (numeric)' : '  (category)'}
        </option>
      ))}
    </select>
    <p className="mt-2 text-[11px] text-slate-400">
      The model learns to predict this column from the others.
    </p>
  </div>
)

export default TargetSelector
