import { Loader2, X } from 'lucide-react'
import type { DatasetPreview } from '../types'

interface Props {
  open: boolean
  loading: boolean
  preview: DatasetPreview | null
  error: string | null
  onClose: () => void
}

const DatasetPreviewModal = ({ open, loading, preview, error, onClose }: Props) => {
  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4"
      onClick={onClose}
    >
      <div
        className="max-h-[85vh] w-full max-w-4xl overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-3.5">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              {preview ? preview.name : 'Dataset preview'}
            </h3>
            {preview && (
              <p className="text-xs text-slate-500">
                Showing first {preview.preview.length} of {preview.rows.toLocaleString()} rows ·{' '}
                {preview.columns} columns
              </p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="max-h-[65vh] overflow-auto px-5 py-4">
          {loading && (
            <div className="flex items-center justify-center gap-2 py-10 text-sm text-slate-500">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading preview...
            </div>
          )}
          {!loading && error && <p className="py-8 text-center text-sm text-rose-600">{error}</p>}
          {!loading && !error && preview && (
            <table className="w-full border-collapse text-left text-xs">
              <thead>
                <tr>
                  {preview.column_names.map((column) => (
                    <th
                      key={column}
                      className="sticky top-0 whitespace-nowrap border-b border-slate-200 bg-white px-3 py-2 font-semibold text-slate-600"
                    >
                      {column}
                      <span className="ml-1 font-normal text-slate-400">
                        {preview.column_kinds[column] === 'numeric' ? 'num' : 'cat'}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.preview.map((row, index) => (
                  <tr key={index} className="border-b border-slate-100 last:border-0">
                    {preview.column_names.map((column) => (
                      <td key={column} className="px-3 py-2 text-slate-700">
                        {row[column] === null || row[column] === undefined ? (
                          <span className="text-slate-300">null</span>
                        ) : (
                          String(row[column])
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}

export default DatasetPreviewModal
