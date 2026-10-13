import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown, FlaskConical, GitBranch, Loader2, Network } from 'lucide-react'
import Header from './components/Header'
import DatasetUpload from './components/DatasetUpload'
import DatasetPreviewModal from './components/DatasetPreview'
import TargetSelector from './components/TargetSelector'
import ModelSelector from './components/ModelSelector'
import AdvancedSettingsPanel from './components/AdvancedSettings'
import { DEFAULT_SETTINGS } from './constants'
import DecisionTreeCanvas from './components/DecisionTreeCanvas'
import NodeInspector from './components/NodeInspector'
import TreeSelector from './components/TreeSelector'
import PredictionExplorer from './components/PredictionExplorer'
import ModelDetails from './components/ModelDetails'
import ErrorMessage from './components/ErrorMessage'
import { api, errorMessage, isCancelled } from './services/api'
import type {
  AdvancedSettings,
  AppStatus,
  DatasetInfo,
  DatasetPreview,
  GenerateModelBody,
  ModelResult,
  ModelType,
  ProblemType,
  SampleDataset,
  TreeOverview,
  TreeStructure,
} from './types'
import { formatNumber, formatPercent } from './utils/format'

function App() {
  const [samples, setSamples] = useState<SampleDataset[]>([])
  const [dataset, setDataset] = useState<DatasetInfo | null>(null)
  const [uploading, setUploading] = useState(false)
  const [targetColumn, setTargetColumn] = useState('')
  const [problemTypeHint, setProblemTypeHint] = useState<{
    datasetId: string
    targetColumn: string
    problemType: ProblemType
  } | null>(null)
  const [modelType, setModelType] = useState<ModelType>('decision_tree')
  const [settings, setSettings] = useState<AdvancedSettings>(DEFAULT_SETTINGS)
  const [settledSettings, setSettledSettings] = useState<AdvancedSettings>(DEFAULT_SETTINGS)
  const [advancedOpen, setAdvancedOpen] = useState(false)

  const [status, setStatus] = useState<AppStatus>('empty')
  const [error, setError] = useState<string | null>(null)
  const [model, setModel] = useState<ModelResult | null>(null)
  const [selectedTree, setSelectedTree] = useState(0)
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null)
  const [path, setPath] = useState<number[]>([])
  const [pathEdges, setPathEdges] = useState<[number, number][]>([])
  const [fetchedTree, setFetchedTree] = useState<{ modelId: string; tree: TreeStructure } | null>(null)
  const [forestOverviewCache, setForestOverviewCache] = useState<{ id: string; trees: TreeOverview[] } | null>(null)

  const modelId = model?.model_id
  const isForest = model?.type === 'random_forest'
  const activeTree =
    model?.type === 'decision_tree'
      ? (model.tree ?? null)
      : fetchedTree && model && fetchedTree.modelId === modelId
        ? fetchedTree.tree
        : null
  const forestOverview =
    forestOverviewCache && forestOverviewCache.id === modelId ? forestOverviewCache.trees : []

  const [previewOpen, setPreviewOpen] = useState(false)
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [predictionOpen, setPredictionOpen] = useState(false)
  const [detailsOpen, setDetailsOpen] = useState(false)
  const [retryToken, setRetryToken] = useState(0)

  const lastSignatureRef = useRef('')
  const lastOutcomeRef = useRef<'pending' | 'ok' | 'failed'>('pending')
  const requestIdRef = useRef(0)

  // -------------------------------------------------------------------------
  // Sample dataset list
  // -------------------------------------------------------------------------
  useEffect(() => {
    api
      .listSamples()
      .then(setSamples)
      .catch(() => setSamples([]))
  }, [])

  // -------------------------------------------------------------------------
  // Detected task for the selected target (drives the criterion options in
  // 'auto' mode; the backend still re-checks everything server-side)
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!dataset || !targetColumn) return
    let stale = false
    api
      .getProblemType(dataset.dataset_id, targetColumn)
      .then((hint) => {
        if (stale) return
        setProblemTypeHint({
          datasetId: dataset.dataset_id,
          targetColumn,
          problemType: hint.problem_type,
        })
      })
      .catch(() => undefined)
    return () => {
      stale = true
    }
  }, [dataset, targetColumn])

  // -------------------------------------------------------------------------
  // Debounce advanced settings so rapid edits trigger a single regeneration
  // -------------------------------------------------------------------------
  useEffect(() => {
    const timer = window.setTimeout(() => setSettledSettings(settings), 600)
    return () => window.clearTimeout(timer)
  }, [settings])

  // -------------------------------------------------------------------------
  // Automatic model generation (no Train Model button anywhere)
  // -------------------------------------------------------------------------
  const signature =
    dataset && targetColumn
      ? JSON.stringify([dataset.dataset_id, targetColumn, modelType, settledSettings])
      : ''

  // Reset model state whenever the configuration signature changes.
  // Done as a render-phase adjustment (React's documented pattern) rather
  // than inside an effect, so cascading renders are avoided.
  const [prevSignature, setPrevSignature] = useState<string | null>(null)
  if (signature !== prevSignature) {
    setPrevSignature(signature)
    setModel(null)
    setSelectedNodeId(null)
    setPath([])
    setPathEdges([])
    setStatus(dataset ? 'awaiting_target' : 'empty')
  }

  useEffect(() => {
    if (!signature) {
      lastSignatureRef.current = ''
      lastOutcomeRef.current = 'pending'
      return
    }

    // Skip duplicate requests when nothing about the configuration changed.
    if (signature === lastSignatureRef.current && lastOutcomeRef.current === 'ok') return

    const controller = new AbortController()
    const requestId = ++requestIdRef.current
    lastSignatureRef.current = signature
    lastOutcomeRef.current = 'pending'
    setStatus('generating')
    setError(null)

    const body: GenerateModelBody = {
      dataset_id: dataset!.dataset_id,
      target_column: targetColumn,
      algorithm: modelType,
      problem_type: settledSettings.problem_type,
      max_depth: settledSettings.max_depth,
      min_samples_leaf: settledSettings.min_samples_leaf,
      ...(modelType === 'decision_tree'
        ? { criterion: settledSettings.criterion }
        : {
            n_estimators: settledSettings.n_estimators,
            max_features: settledSettings.max_features,
          }),
    }

    api
      .generateModel(body, { signal: controller.signal })
      .then((result) => {
        if (requestId !== requestIdRef.current) return // stale response guard
        lastOutcomeRef.current = 'ok'
        setModel(result)
        setSelectedTree(0)
        setSelectedNodeId(null)
        setPath([])
        setPathEdges([])
        setPredictionOpen(false)
        setStatus('ready')
        setError(null)
      })
      .catch((err) => {
        if (isCancelled(err) || controller.signal.aborted) return
        if (requestId !== requestIdRef.current) return
        lastOutcomeRef.current = 'failed'
        setError(errorMessage(err))
        setStatus('error')
      })

    return () => controller.abort()
  }, [signature, retryToken, dataset, targetColumn, modelType, settledSettings])

  // -------------------------------------------------------------------------
  // Random forest: tree metadata + selected tree structure
  // (single decision trees are derived directly from the trained model)
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!modelId || !isForest) return
    let stale = false
    api
      .getForestTrees(modelId)
      .then((response) => {
        if (!stale) setForestOverviewCache({ id: modelId, trees: response.overview })
      })
      .catch(() => undefined)
    return () => {
      stale = true
    }
  }, [modelId, isForest])

  useEffect(() => {
    if (!modelId || !isForest) return
    let stale = false
    api
      .getForestTree(modelId, selectedTree)
      .then((tree) => {
        if (!stale) setFetchedTree({ modelId, tree })
      })
      .catch((err) => {
        if (!stale && !isCancelled(err)) setError(errorMessage(err))
      })
    return () => {
      stale = true
    }
  }, [modelId, isForest, selectedTree])

  // -------------------------------------------------------------------------
  // Dataset actions
  // -------------------------------------------------------------------------
  const resetModel = useCallback(() => {
    setModel(null)
    setSelectedNodeId(null)
    setPath([])
    setPathEdges([])
    setPredictionOpen(false)
  }, [])

  // Task-specific settings belong to the selected target column: reset the
  // problem-type override and criterion whenever the target changes so a
  // previous Classification/Regression choice cannot leak into the next column.
  const resetTaskSettings = useCallback(() => {
    setSettings((prev) =>
      prev.problem_type === DEFAULT_SETTINGS.problem_type &&
      prev.criterion === DEFAULT_SETTINGS.criterion
        ? prev
        : {
            ...prev,
            problem_type: DEFAULT_SETTINGS.problem_type,
            criterion: DEFAULT_SETTINGS.criterion,
          },
    )
  }, [])

  const handleFile = useCallback(
    async (file: File) => {
      setUploading(true)
      setError(null)
      try {
        const info = await api.uploadDataset(file)
        setDataset(info)
        setTargetColumn('')
        resetModel()
        resetTaskSettings()
        setStatus('awaiting_target')
        lastSignatureRef.current = ''
      } catch (err) {
        setError(errorMessage(err))
      } finally {
        setUploading(false)
      }
    },
    [resetModel, resetTaskSettings],
  )

  const handleSample = useCallback(
    async (sampleId: string) => {
      setUploading(true)
      setError(null)
      try {
        const info = await api.loadSample(sampleId)
        const sample = samples.find((entry) => entry.id === sampleId)
        const defaultTarget =
          sample?.default_target && info.column_names.includes(sample.default_target)
            ? sample.default_target
            : info.column_names[info.column_names.length - 1]
        setDataset(info)
        setTargetColumn(defaultTarget) // samples start generating immediately
        resetModel()
        resetTaskSettings()
        lastSignatureRef.current = ''
      } catch (err) {
        setError(errorMessage(err))
      } finally {
        setUploading(false)
      }
    },
    [samples, resetModel, resetTaskSettings],
  )

  const handlePreview = useCallback(async () => {
    if (!dataset) return
    setPreviewOpen(true)
    setPreviewLoading(true)
    setPreviewError(null)
    try {
      const data = await api.previewDataset(dataset.dataset_id, 10)
      setPreview(data)
    } catch (err) {
      setPreviewError(errorMessage(err))
    } finally {
      setPreviewLoading(false)
    }
  }, [dataset])

  // -------------------------------------------------------------------------
  // Node selection / prediction path highlighting
  // -------------------------------------------------------------------------
  const selectedNode = useMemo(() => {
    if (selectedNodeId == null || !activeTree) return null
    return activeTree.nodes.find((node) => node.id === selectedNodeId) ?? null
  }, [selectedNodeId, activeTree])

  const handlePath = useCallback((nextPath: number[], nextEdges: [number, number][]) => {
    setPath(nextPath)
    setPathEdges(nextEdges)
    setSelectedNodeId(nextPath.length ? nextPath[nextPath.length - 1] : null)
  }, [])

  const handleClearPath = useCallback(() => {
    setPath([])
    setPathEdges([])
  }, [])

  // -------------------------------------------------------------------------
  // Derived UI state
  // -------------------------------------------------------------------------
  // Only trust the hint when it matches the current dataset + target.
  const detectedProblemType =
    problemTypeHint &&
    dataset &&
    problemTypeHint.datasetId === dataset.dataset_id &&
    problemTypeHint.targetColumn === targetColumn
      ? problemTypeHint.problemType
      : null

  const generating = status === 'generating'
  const canvasMessage = !dataset
    ? 'Upload a CSV dataset or try a sample dataset to get started.'
    : !targetColumn
      ? 'Select the column you would like to predict to generate your tree.'
      : null

  const primaryMetric = useMemo(() => {
    if (!model) return null
    if (model.problem_type === 'classification') {
      return { label: 'Accuracy', value: formatPercent(model.metrics.accuracy) }
    }
    return { label: 'MAE', value: formatNumber(model.metrics.mae) }
  }, [model])

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <Header samples={samples} loading={uploading} onSample={handleSample} onFile={handleFile} />

      <main className="mx-auto max-w-[1400px] px-4 pb-16 pt-8 sm:px-6">
        {/* Page heading */}
        <div className="mb-6">
          <h2 className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
            Visualize Your Decision Tree
          </h2>
          <p className="mt-1.5 max-w-2xl text-sm text-slate-500">
            Upload your data, choose what to predict, and explore how the model makes decisions.
          </p>
        </div>

        {/* Controls */}
        <div className="grid gap-4 lg:grid-cols-3">
          <DatasetUpload
            dataset={dataset}
            uploading={uploading}
            samples={samples}
            onFile={handleFile}
            onSample={handleSample}
            onPreview={handlePreview}
          />
          <TargetSelector
            dataset={dataset}
            value={targetColumn}
            onChange={(column) => {
              setTargetColumn(column)
              resetTaskSettings()
              setSelectedNodeId(null)
              setPath([])
              setPathEdges([])
            }}
          />
          <ModelSelector
            value={modelType}
            onChange={(type) => {
              setModelType(type)
              setSelectedNodeId(null)
              setPath([])
              setPathEdges([])
            }}
          />
        </div>

        <div className="mt-4">
          <AdvancedSettingsPanel
            open={advancedOpen}
            modelType={modelType}
            settings={settings}
            detectedProblemType={detectedProblemType}
            onToggle={() => setAdvancedOpen((v) => !v)}
            onChange={setSettings}
          />
        </div>

        {error && (
          <div className="mt-4">
            <ErrorMessage message={error} onRetry={() => setRetryToken((t) => t + 1)} />
          </div>
        )}

        {uploading && (
          <div className="mt-4 flex items-center gap-2 rounded-xl border border-indigo-100 bg-indigo-50 px-4 py-2.5 text-sm font-medium text-indigo-700">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading dataset...
          </div>
        )}

        {/* Model bar */}
        {model && (
          <div className="mt-4 flex flex-wrap items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3">
            <span className="flex items-center gap-2 text-sm font-semibold text-slate-800">
              {model.type === 'random_forest' ? (
                <GitBranch className="h-4 w-4 text-indigo-500" />
              ) : (
                <Network className="h-4 w-4 text-indigo-500" />
              )}
              {model.type === 'random_forest' ? 'Random Forest' : 'Decision Tree'}
            </span>
            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-medium capitalize text-slate-600">
              {model.problem_type} detected
              {model.problem_type !== model.problem_type_detected && ' (override)'}
            </span>
            <span className="text-xs text-slate-400">
              predicting <span className="font-semibold text-slate-600">{model.target_column}</span>
            </span>

            <div className="ml-auto flex items-center gap-3">
              {primaryMetric && (
                <span className="flex items-baseline gap-1.5 rounded-lg bg-emerald-50 px-3 py-1.5">
                  <span className="text-[11px] font-medium text-emerald-600">
                    {primaryMetric.label}
                  </span>
                  <span className="text-sm font-bold text-emerald-700">
                    {primaryMetric.value}
                  </span>
                </span>
              )}
              {generating && (
                <span className="flex items-center gap-1.5 text-xs font-medium text-indigo-600">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" /> Updating...
                </span>
              )}
            </div>
          </div>
        )}

        {/* Toolbar */}
        {(model || generating) && (
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              {isForest && forestOverview.length > 0 && (
                <TreeSelector
                  overview={forestOverview}
                  value={selectedTree}
                  onChange={(index) => {
                    setSelectedTree(index)
                    setSelectedNodeId(null)
                    setPath([])
                    setPathEdges([])
                  }}
                />
              )}
              {isForest && forestOverview.length === 0 && generating && (
                <span className="text-xs text-slate-400">Loading trees...</span>
              )}
            </div>
            {model && (
              <button
                type="button"
                onClick={() => setPredictionOpen((v) => !v)}
                className={`flex items-center gap-1.5 rounded-lg border px-3 py-2 text-sm font-medium transition ${
                  predictionOpen
                    ? 'border-indigo-300 bg-indigo-50 text-indigo-700'
                    : 'border-slate-300 bg-white text-slate-600 hover:bg-slate-50'
                }`}
              >
                <FlaskConical className="h-4 w-4" />
                Try a Prediction
                <ChevronDown
                  className={`h-3.5 w-3.5 transition ${predictionOpen ? 'rotate-180' : ''}`}
                />
              </button>
            )}
          </div>
        )}

        {/* Visualization */}
        <section className="mt-3">
          <div className="flex flex-col gap-3 xl:flex-row">
            <div className="min-w-0 flex-1">
              <DecisionTreeCanvas
                tree={activeTree}
                selectedNodeId={selectedNodeId}
                highlightedPath={path}
                highlightedEdges={pathEdges}
                generating={generating}
                message={canvasMessage ?? undefined}
                onSelectNode={setSelectedNodeId}
              />
            </div>
            <NodeInspector
              node={selectedNode}
              problemType={model?.problem_type ?? 'classification'}
              targetColumn={model?.target_column ?? ''}
              onClose={() => setSelectedNodeId(null)}
            />
          </div>
        </section>

        {/* Prediction explorer */}
        {predictionOpen && dataset && model && (
          <div className="mt-4">
            <PredictionExplorer
              open={predictionOpen}
              dataset={dataset}
              model={model}
              selectedTree={selectedTree}
              onClose={() => setPredictionOpen(false)}
              onPath={handlePath}
              onClearPath={handleClearPath}
            />
          </div>
        )}

        {/* Model details */}
        {model && (
          <div className="mt-4">
            <ModelDetails
              model={model}
              open={detailsOpen}
              onToggle={() => setDetailsOpen((v) => !v)}
            />
          </div>
        )}

        {status === 'ready' && model && !generating && (
          <p className="mt-4 text-center text-[11px] text-slate-400">
            Model generated automatically from{' '}
            <span className="font-medium text-slate-500">{dataset?.name}</span> · trained on{' '}
            {model.n_train.toLocaleString()} samples · trees update whenever you change the
            configuration.
          </p>
        )}
      </main>

      <DatasetPreviewModal
        open={previewOpen}
        loading={previewLoading}
        preview={preview}
        error={previewError}
        onClose={() => setPreviewOpen(false)}
      />
    </div>
  )
}

export default App
