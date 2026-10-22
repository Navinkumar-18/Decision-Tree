import { TreeDeciduous, Users } from 'lucide-react'

const Footer = () => {
  return (
    <footer className="border-t border-slate-200 bg-white">
      <div className="mx-auto flex max-w-[1400px] flex-col items-center justify-between gap-4 px-4 py-5 sm:flex-row sm:px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600 text-white">
            <TreeDeciduous className="h-4 w-4" />
          </div>
          <span className="text-sm font-semibold text-slate-700">TreeViz AI</span>
          <span className="hidden text-xs text-slate-400 sm:inline">
            Explore Your Data. Visualize Your Decisions.
          </span>
        </div>

        <div className="flex flex-col items-center gap-1 text-sm sm:flex-row sm:items-center sm:gap-4">
          <span className="flex items-center gap-1.5 text-slate-500">
            <Users className="h-4 w-4 text-indigo-500" />
            Team: <span className="font-semibold text-slate-800">Tree Wizards</span>
          </span>

          <span className="hidden h-4 w-px bg-slate-200 sm:block" />

          <span className="text-slate-500">
            by <span className="font-semibold text-slate-800">Navinkumar M</span>
            <span className="mx-1.5 text-slate-300">&middot;</span>
            <span className="font-semibold text-slate-800">Jeriel J</span>
          </span>
        </div>
      </div>
    </footer>
  )
}

export default Footer
