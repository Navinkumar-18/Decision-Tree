import { useRef, useState } from 'react'
import { Database, Eye, FileSpreadsheet, Loader2, Upload } from 'lucide-react'
import type { DatasetInfo, SampleDataset } from '../types'

interface Props {
  dataset: DatasetInfo | null
  uploading: boolean
  samples: SampleDataset[]
  onFile: (file: File) => void
  onSample: (sampleId: string) => void
  onPreview: () => void
}

const DatasetUpload = ({ dataset, uploading, samples, onFile, onSample, onPreview }: Props) => {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const sampleRef = useRef<HTMLSelectElement>(null)

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="mb-3 flex items-center gap-2">
        <Database className="h-4 w-4 text-indigo-500" />
        <h3 className="text-sm font-semibold text-slate-800">Dataset</h3>
      </div>

      {!dataset ? (
        <div
          onDragOver={(event) => {
            event.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault()
            setDragging(false)
            const file = event.dataTransfer.files?.[0]
            if (file) onFile(file)
          }}
          className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed px-3 py-6 text-center transition ${
            dragging ? 'border-indigo-400 bg-indigo-50' : 'border-slate-200 bg-slate-50'
          }`}
        >
          <Upload className={`mb-2 h-5 w-5 ${dragging ? 'text-indigo-500' : 'text-slate-400'}`} />
          <p className="text-xs text-slate-500">
            Drag &amp; drop your CSV here, or
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="mx-1 font-semibold text-indigo-600 hover:underline"
            >
              browse files
            </button>
          </p>
          <div className="mt-3 flex items-center gap-2">
            <span className="text-[11px] text-slate-400">or try a sample:</span>
            <select
              ref={sampleRef}
              defaultValue=""
              disabled={uploading}
              onChange={(event) => {
                if (event.target.value) onSample(event.target.value)
                event.target.value = ''
              }}
              className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-700"
            >
              <option value="" disabled>
                Choose dataset...
              </option>
              {samples.map((sample) => (
                <option key={sample.id} value={sample.id}>
                  {sample.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      ) : (
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-3">
          <div className="flex items-start justify-between gap-2">
            <div className="flex min-w-0 items-center gap-2">
              <FileSpreadsheet className="h-4 w-4 shrink-0 text-emerald-500" />
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-slate-800">{dataset.name}</p>
                <p className="text-[11px] text-slate-500">
                  {dataset.rows.toLocaleString()} rows · {dataset.columns} columns
                  {Object.values(dataset.missing_values).reduce((a, b) => a + b, 0) > 0 && (
                    <span className="text-amber-600">
                      {' '}
                      ·{' '}
                      {Object.values(dataset.missing_values).reduce((a, b) => a + b, 0)} missing
                      values
                    </span>
                  )}
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={onPreview}
              className="flex shrink-0 items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-[11px] font-medium text-slate-600 hover:bg-slate-100"
            >
              <Eye className="h-3.5 w-3.5" />
              Preview
            </button>
          </div>
          <div className="mt-2 flex items-center gap-2">
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="text-[11px] font-medium text-indigo-600 hover:underline"
            >
              Replace file
            </button>
            <span className="text-slate-300">·</span>
            <select
              defaultValue=""
              disabled={uploading}
              onChange={(event) => {
                if (event.target.value) onSample(event.target.value)
                event.target.value = ''
              }}
              className="border-0 bg-transparent p-0 text-[11px] font-medium text-indigo-600 focus:ring-0"
            >
              <option value="" disabled>
                Load a sample dataset
              </option>
              {samples.map((sample) => (
                <option key={sample.id} value={sample.id}>
                  {sample.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {uploading && (
        <div className="mt-2 flex items-center gap-1.5 text-xs text-indigo-600">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Uploading dataset...
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) onFile(file)
          event.target.value = ''
        }}
      />
    </div>
  )
}

export default DatasetUpload
