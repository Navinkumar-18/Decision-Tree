import { Link, useLocation } from 'react-router-dom'
import { 
  LayoutDashboard, 
  Database, 
  Network, 
  GitBranch, 
  BarChart3, 
  Settings,
  Brain,
  Target
} from 'lucide-react'

const Sidebar = () => {
  const location = useLocation()
  
  const navItems = [
    { path: '/', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/data', label: 'Data Explorer', icon: Database },
    { path: '/decision-tree', label: 'Decision Tree', icon: Network },
    { path: '/random-forest', label: 'Random Forest', icon: GitBranch },
    { path: '/prediction', label: 'Prediction', icon: Target },
    { path: '/evaluation', label: 'Model Evaluation', icon: BarChart3 },
    { path: '/settings', label: 'Settings', icon: Settings },
  ]

  return (
    <div className="w-64 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 h-screen flex flex-col">
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2">
          <Brain className="h-8 w-8 text-blue-600" />
          <div>
            <h1 className="text-xl font-bold">TreeViz AI</h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">Explore Your Data. Visualize Your Decisions.</p>
          </div>
        </div>
      </div>
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path
            return (
              <li key={item.path}>
                <Link
                  to={item.path}
                  className={`flex items-center gap-3 px-3 py-2 rounded-md transition-colors ${
                    isActive
                      ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <span>{item.label}</span>
                </Link>
              </li>
            )
          })}
        </ul>
      </nav>
    </div>
  )
}

export default Sidebar
