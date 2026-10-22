import { Loader2 } from 'lucide-react'

const LoadingState = ({ label }: { label?: string }) => (
  <div className="flex items-center gap-2 text-sm font-medium text-indigo-600">
    <Loader2 className="h-4 w-4 animate-spin" />
    {label ?? 'Generating your decision tree...'}
  </div>
)

export default LoadingState
