import { AlertCircle, RefreshCw } from 'lucide-react'

const ErrorMessage = ({ message, onRetry }: { message: string; onRetry?: () => void }) => (
  <div className="flex items-start justify-between gap-3 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3">
    <div className="flex items-start gap-2.5">
      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-500" />
      <div>
        <p className="text-sm font-semibold text-rose-700">Something went wrong</p>
        <p className="mt-0.5 text-xs leading-relaxed text-rose-600">{message}</p>
      </div>
    </div>
    {onRetry && (
      <button
        type="button"
        onClick={onRetry}
        className="flex shrink-0 items-center gap-1 rounded-lg border border-rose-200 bg-white px-2.5 py-1.5 text-xs font-semibold text-rose-600 hover:bg-rose-100"
      >
        <RefreshCw className="h-3.5 w-3.5" /> Retry
      </button>
    )}
  </div>
)

export default ErrorMessage
