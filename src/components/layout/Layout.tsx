import Sidebar from './Sidebar'
import { useState, useEffect } from 'react'

const Layout = ({ children }: { children: React.ReactNode }) => {
  const [darkMode, setDarkMode] = useState(false)

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-950">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">TreeViz AI Dashboard</h2>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="px-3 py-1 text-sm rounded border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800"
            >
              {darkMode ? 'Light Mode' : 'Dark Mode'}
            </button>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  )
}

export default Layout
