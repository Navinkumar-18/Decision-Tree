import { useCallback, useEffect, useMemo } from 'react'
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Edge,
  type Node,
} from '@xyflow/react'
import type { TreeStructure } from '../types'
import { layoutTree, type FlowNodeData } from '../utils/treeLayout'
import DecisionNode from './DecisionNode'
import LeafNode from './LeafNode'
import { Loader2 } from 'lucide-react'

const nodeTypes = { decision: DecisionNode, leaf: LeafNode }

interface Props {
  tree: TreeStructure | null
  selectedNodeId: number | null
  highlightedPath: number[]
  highlightedEdges: [number, number][]
  generating: boolean
  message?: string
  onSelectNode: (nodeId: number | null) => void
}

function FlowCanvas({
  tree,
  selectedNodeId,
  highlightedPath,
  highlightedEdges,
  generating,
  message,
  onSelectNode,
}: Props) {
  const { fitView } = useReactFlow()
  const layout = useMemo(() => layoutTree(tree), [tree])
  const treeSignature = useMemo(
    () => (tree ? `${tree.n_nodes}:${tree.nodes.length}` : 'none'),
    [tree],
  )

  const pathSet = useMemo(() => new Set(highlightedPath), [highlightedPath])
  const edgeSet = useMemo(
    () => new Set(highlightedEdges.map(([a, b]) => `n${a}-n${b}`)),
    [highlightedEdges],
  )

  const nodes = useMemo<Node<FlowNodeData>[]>(() => {
    return layout.nodes.map((base) => {
      const nodeId = base.data.node.id
      const classes: string[] = []
      if (nodeId === selectedNodeId) classes.push('ring-2 ring-indigo-500')
      else if (pathSet.has(nodeId)) classes.push('ring-2 ring-indigo-300')
      return { ...base, className: classes.join(' ') }
    })
  }, [layout, selectedNodeId, pathSet])

  const edges = useMemo<Edge[]>(() => {
    return layout.edges.map((base) => {
      if (edgeSet.has(base.id)) {
        return {
          ...base,
          animated: true,
          style: { stroke: '#6366f1', strokeWidth: 2.5 },
          labelStyle: { fontSize: 11, fill: '#4338ca', fontWeight: 700 },
        }
      }
      return base
    })
  }, [layout, edgeSet])

  // Fit the freshly generated tree into view without shrinking it excessively.
  useEffect(() => {
    if (!tree || !layout.nodes.length) return
    const timer = window.setTimeout(() => {
      fitView({ padding: 0.15, duration: 300, maxZoom: 1 })
    }, 60)
    return () => window.clearTimeout(timer)
  }, [treeSignature, fitView, layout.nodes.length, tree])

  const onPaneClick = useCallback(() => onSelectNode(null), [onSelectNode])

  return (
    <div className="relative h-[560px] w-full rounded-2xl border border-slate-200 bg-white">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => {
          const flowData = node.data as FlowNodeData
          onSelectNode(flowData.node.id)
        }}
        onPaneClick={onPaneClick}
        fitView
        fitViewOptions={{ padding: 0.15, maxZoom: 1 }}
        minZoom={0.05}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
      >
        <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#e2e8f0" />
        <Controls showInteractive={false} position="bottom-left" />
        <MiniMap
          pannable
          zoomable
          position="bottom-right"
          nodeColor={(n) => (n.type === 'leaf' ? '#a7f3d0' : '#c7d2fe')}
          maskColor="rgba(15, 23, 42, 0.05)"
          style={{ width: 150, height: 100 }}
        />
      </ReactFlow>

      {!tree && message && (
        <div className="absolute inset-0 flex items-center justify-center rounded-2xl bg-slate-50/80">
          <div className="max-w-md px-6 text-center">
            <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-xl bg-indigo-50 text-indigo-500">
              <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M12 3v6m0 0-4 4m4-4 4 4M6 21v-4m12 4v-4M4 17h4m8 0h4" strokeLinecap="round" />
              </svg>
            </div>
            <p className="text-sm font-medium text-slate-600">{message}</p>
          </div>
        </div>
      )}

      {generating && (
        <div className="absolute inset-x-0 top-0 flex items-center justify-center rounded-t-2xl bg-white/85 py-3 backdrop-blur-sm">
          <div className="flex items-center gap-2 rounded-full border border-indigo-100 bg-indigo-50 px-4 py-1.5 text-sm font-medium text-indigo-700">
            <Loader2 className="h-4 w-4 animate-spin" />
            Generating your decision tree...
          </div>
        </div>
      )}
    </div>
  )
}

const DecisionTreeCanvas = (props: Props) => (
  <ReactFlowProvider>
    <FlowCanvas {...props} />
  </ReactFlowProvider>
)

export default DecisionTreeCanvas
