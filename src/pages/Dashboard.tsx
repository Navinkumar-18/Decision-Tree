const Dashboard = () => {
  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h1 className="text-3xl font-bold mb-2">TreeViz AI</h1>
        <p className="text-gray-600 dark:text-gray-400">Explore Your Data. Visualize Your Decisions.</p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold mb-2">Upload Dataset</h3>
          <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">Upload a CSV file to get started</p>
        </div>
        <div className="bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold mb-2">Train Models</h3>
          <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">Build Decision Trees and Random Forests</p>
        </div>
        <div className="bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold mb-2">Visualize</h3>
          <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">Interactive tree visualizations with React Flow</p>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
