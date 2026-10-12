import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import DataExplorer from './pages/DataExplorer'
import DecisionTreePage from './pages/DecisionTree'
import RandomForestPage from './pages/RandomForest'
import ModelEvaluation from './pages/ModelEvaluation'
import Settings from './pages/Settings'
import PredictionPage from './pages/Prediction'
import './App.css'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/data" element={<DataExplorer />} />
          <Route path="/decision-tree" element={<DecisionTreePage />} />
          <Route path="/random-forest" element={<RandomForestPage />} />
          <Route path="/evaluation" element={<ModelEvaluation />} />
          <Route path="/prediction" element={<PredictionPage />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App
