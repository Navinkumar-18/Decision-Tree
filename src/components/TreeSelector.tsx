import type { TreeOverview } from '../types'

interface Props {
  overview: TreeOverview[]
  value: number
  onChange: (index: number) => void
}

const TreeSelector = ({ overview, value, onChange }: Props) => {
  const current = overview[value]
  return (
    <div className="flex flex-wrap items-center gap-3">
      <label className="flex items-center gap-2 text-xs font-medium text-slate-500">
        Selected tree
        <select
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm font-semibold text-slate-800 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
        >
          {overview.map((tree) => (
            <option key={tree.index} value={tree.index}>
              Tree {tree.index + 1}
            </option>
          ))}
        </select>
      </label>
      {current && (
        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-medium text-slate-600">
          Depth {current.depth} · {current.nodes} nodes · {current.leaves} leaves
        </span>
      )}
    </div>
  )
}

export default TreeSelector
