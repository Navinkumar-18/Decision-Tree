import { TreeDeciduous } from 'lucide-react'

const Footer = () => {
  return (
    <footer className="border-t border-slate-200 bg-white">
      <div className="mx-auto flex max-w-[1400px] flex-col items-center justify-between gap-3 px-4 py-5 sm:flex-row sm:px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600 text-white">
            <TreeDeciduous className="h-4 w-4" />
          </div>
          <span className="text-sm font-semibold text-slate-700">TreeViz AI</span>
          <span className="hidden text-xs text-slate-400 sm:inline">
            Explore Your Data. Visualize Your Decisions.
          </span>
        </div>

        <p className="text-sm text-slate-500">
          by <span className="font-semibold text-slate-800">Navinkumar M</span>
        </p>
      </div>
    </footer>
  )
}

export default Footer
