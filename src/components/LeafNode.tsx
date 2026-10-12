import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import type { FlowNodeData } from '../utils/treeLayout'
import { formatValue } from '../utils/format'

const handleClass = '!w-2.5 !h-2.5 !bg-emerald-500 !border-2 !border-white'

function toneFor(node: FlowNodeData['node']): string {
  if (node.class_distribution == null) {
    // regression leaf: neutral slate
    return 'border-slate-300 bg-slate-50'
  }
  const classes = Object.keys(node.class_distribution)
  if (classes.length === 2) {
    const positive = classes[1]
    return String(node.prediction) === positive
      ? 'border-emerald-200 bg-emerald-50'
      : 'border-rose-200 bg-rose-50'
  }
  // multiclass: show the predicted class on a calm indigo tint
  return 'border-indigo-200 bg-indigo-50'
}

type LeafFlowNode = Node<FlowNodeData, 'leaf'>

const LeafNode = ({ data }: NodeProps<LeafFlowNode>) => {
  const { node } = data
  return (
    <div
      className={`h-[92px] rounded-xl border px-3 py-2.5 shadow-sm transition-shadow hover:shadow-md ${toneFor(node)}`}
      title={String(node.prediction ?? '')}
    >
      <Handle type="target" position={Position.Top} className={handleClass} />
      <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">
        Prediction
      </div>
      <div className="mt-1 truncate text-[15px] font-semibold text-slate-800">
        {formatValue(node.prediction)}
      </div>
      <div className="mt-1 text-[11px] text-slate-400">{node.samples} samples</div>
      <Handle type="source" position={Position.Bottom} className={handleClass} />
    </div>
  )
}

export default LeafNode
