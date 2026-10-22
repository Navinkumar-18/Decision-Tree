import { useEffect, useRef, useState } from 'react'
import { ChevronDown, Sparkles, TreeDeciduous, Upload } from 'lucide-react'
import type { SampleDataset } from '../types'

interface Props {
  samples: SampleDataset[]
  loading: boolean
  onSample: (sampleId: string) => void
  onFile: (file: File) => void
}

const Header = ({ samples, loading, onSample, onFile }: Props) => {
  const [open, setOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!open) return
    const onClick = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [open])

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur">
      <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600 text-white">
            <TreeDeciduous className="h-5 w-5" />
          </div>
          <div className="leading-tight">
            <div className="flex items-center gap-2">
              <h1 className="text-[17px] font-bold tracking-tight text-slate-900">TreeViz AI</h1>
              <span className="hidden rounded-full border border-indigo-100 bg-indigo-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-indigo-600 sm:inline">
                Decision Tree Visualizer
              </span>
            </div>
            <p className="hidden text-xs text-slate-400 sm:block">
              Explore Your Data. Visualize Your Decisions.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="relative" ref={menuRef}>
            <button
              type="button"
              disabled={loading}
              onClick={() => setOpen((v) => !v)}
              className="flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
            >
              <Sparkles className="h-4 w-4 text-indigo-500" />
              Try Sample Dataset
              <ChevronDown className={`h-4 w-4 transition ${open ? 'rotate-180' : ''}`} />
            </button>
            {open && (
              <div className="absolute right-0 z-40 mt-2 w-72 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-lg">
                {samples.map((sample) => (
                  <button
                    key={sample.id}
                    type="button"
                    onClick={() => {
                      setOpen(false)
                      onSample(sample.id)
                    }}
                    className="block w-full border-b border-slate-100 px-4 py-3 text-left last:border-0 hover:bg-slate-50"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-semibold text-slate-800">{sample.name}</span>
                      <span
                        className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ${
                          sample.task === 'regression'
                            ? 'bg-amber-50 text-amber-700'
                            : 'bg-emerald-50 text-emerald-700'
                        }`}
                      >
                        {sample.task}
                      </span>
                    </div>
                    <p className="mt-0.5 text-xs text-slate-500">{sample.description}</p>
                  </button>
                ))}
              </div>
            )}
          </div>

          <button
            type="button"
            disabled={loading}
            onClick={() => inputRef.current?.click()}
            className="flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3.5 py-2 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:opacity-50"
          >
            <Upload className="h-4 w-4" />
            Load Dataset
          </button>
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
      </div>
    </header>
  )
}

export default Header
