import type { Edge, Node } from '@xyflow/react'
import type { TreeNode, TreeStructure } from '../types'

export const NODE_WIDTH = 216
export const NODE_HEIGHT = 92
const H_GAP = 32
const V_GAP = 72

export interface DecisionNodeData extends Record<string, unknown> {
  node: TreeNode
}

export type FlowNodeData = DecisionNodeData

interface Positioned {
  x: number
  y: number
}

function childrenOf(node: TreeNode): number[] {
  if (node.is_leaf || node.children_left < 0 || node.children_right < 0) return []
  if (node.children_left === node.id || node.children_right === node.id) return []
  return [node.children_left, node.children_right]
}

/**
 * Classic tidy tree layout: root on top, children centred beneath their
 * parent, consistent horizontal/vertical spacing and no overlaps.
 */
export function layoutTree(tree: TreeStructure | null): {
  nodes: Node<FlowNodeData>[]
  edges: Edge[]
  width: number
  height: number
} {
  if (!tree || !tree.nodes.length) return { nodes: [], edges: [], width: 0, height: 0 }

  const byId = new Map<number, TreeNode>()
  for (const node of tree.nodes) byId.set(node.id, node)

  const root =
    tree.nodes.find((n) => n.parent === -1) ??
    tree.nodes.find((n) => n.id === 0) ??
    tree.nodes[0]

  const widths = new Map<number, number>()
  const measure = (id: number): number => {
    const node = byId.get(id)
    if (!node) return NODE_WIDTH
    const kids = childrenOf(node)
    if (!kids.length) {
      widths.set(id, NODE_WIDTH)
      return NODE_WIDTH
    }
    const total =
      kids.reduce((sum, kid) => sum + measure(kid), 0) + H_GAP * (kids.length - 1)
    const width = Math.max(NODE_WIDTH, total)
    widths.set(id, width)
    return width
  }
  measure(root.id)

  const positions = new Map<number, Positioned>()
  const place = (id: number, left: number, depth: number): void => {
    const node = byId.get(id)
    if (!node) return
    const width = widths.get(id) ?? NODE_WIDTH
    const kids = childrenOf(node)
    const y = depth * (NODE_HEIGHT + V_GAP)
    if (!kids.length) {
      positions.set(id, { x: left + (width - NODE_WIDTH) / 2, y })
      return
    }
    const childTotal =
      kids.reduce((sum, kid) => sum + (widths.get(kid) ?? NODE_WIDTH), 0) +
      H_GAP * (kids.length - 1)
    let cursor = left + (width - childTotal) / 2
    let childCenterSum = 0
    for (const kid of kids) {
      place(kid, cursor, depth + 1)
      const kidWidth = widths.get(kid) ?? NODE_WIDTH
      childCenterSum += cursor + kidWidth / 2
      cursor += kidWidth + H_GAP
    }
    const centre = childCenterSum / kids.length
    positions.set(id, { x: centre - NODE_WIDTH / 2, y })
  }
  place(root.id, 0, 0)

  const nodes: Node<FlowNodeData>[] = []
  for (const node of tree.nodes) {
    const pos = positions.get(node.id)
    if (!pos) continue
    nodes.push({
      id: `n${node.id}`,
      type: node.is_leaf ? 'leaf' : 'decision',
      position: { x: pos.x, y: pos.y },
      data: { node },
      draggable: false,
      deletable: false,
      style: { width: NODE_WIDTH },
    })
  }

  const edges: Edge[] = []
  for (const node of tree.nodes) {
    if (node.is_leaf) continue
    const kids = childrenOf(node)
    if (kids.length !== 2) continue
    const [left, right] = kids
    edges.push(
      {
        id: `e${node.id}-${left}`,
        source: `n${node.id}`,
        target: `n${left}`,
        type: 'smoothstep',
        label: node.left_label ?? 'Yes',
        labelStyle: { fontSize: 11, fill: '#475569', fontWeight: 600 },
        labelBgStyle: { fill: '#f8fafc' },
        labelBgPadding: [4, 2],
        style: { stroke: '#94a3b8', strokeWidth: 1.5 },
      },
      {
        id: `e${node.id}-${right}`,
        source: `n${node.id}`,
        target: `n${right}`,
        type: 'smoothstep',
        label: node.right_label ?? 'No',
        labelStyle: { fontSize: 11, fill: '#475569', fontWeight: 600 },
        labelBgStyle: { fill: '#f8fafc' },
        labelBgPadding: [4, 2],
        style: { stroke: '#94a3b8', strokeWidth: 1.5 },
      },
    )
  }

  let maxX = 0
  let maxY = 0
  for (const pos of positions.values()) {
    maxX = Math.max(maxX, pos.x + NODE_WIDTH)
    maxY = Math.max(maxY, pos.y + NODE_HEIGHT)
  }

  return { nodes, edges, width: maxX, height: maxY }
}
