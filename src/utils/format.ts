/** Small display helpers shared across components. */

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value == null || Number.isNaN(value)) return '—'
  if (!Number.isFinite(value)) return '—'
  const abs = Math.abs(value)
  if (abs !== 0 && (abs >= 1e6 || abs < 1e-4)) return value.toExponential(2)
  const rounded = Number(value.toFixed(digits))
  return String(rounded)
}

export function formatPercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value) || !Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(1)}%`
}

export function formatValue(value: unknown): string {
  if (value == null) return '—'
  if (typeof value === 'number') return formatNumber(value, 3)
  return String(value)
}
