import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import type { FlowNodeData } from '../utils/treeLayout'

const handleClass = '!w-2.5 !h-2.5 !bg-indigo-500 !border-2 !border-white'

type DecisionFlowNode = Node<FlowNodeData, 'decision'>

const DecisionNode = ({ data }: NodeProps<DecisionFlowNode>) => {
  const { node } = data
  const kindLabel = node.feature_kind === 'categorical' ? 'Category split' : 'Split on'

  return (
    <div
      className="h-[92px] rounded-xl border border-slate-200 bg-white px-3 py-2.5 shadow-sm transition-shadow hover:shadow-md"
      title={node.condition}
    >
      <Handle type="target" position={Position.Top} className={handleClass} />
      <div className="flex items-center gap-1.5">
        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-500" />
        <span className="truncate text-[10px] font-semibold uppercase tracking-wide text-indigo-600">
          {node.feature_display ?? kindLabel}
        </span>
      </div>
      <div className="mt-1 truncate font-mono text-[13px] font-medium text-slate-800">
        {node.condition}
      </div>
      <div className="mt-1 text-[11px] text-slate-400">{node.samples} samples</div>
      <Handle type="source" position={Position.Bottom} className={handleClass} />
    </div>
  )
}

export default DecisionNode
