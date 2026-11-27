import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AIGenerateModal } from './components/AIGenerateModal';
import { ExecutionLog } from './components/ExecutionLog';
import { InspectorSidebar } from './components/InspectorSidebar';
import { NodeLibrarySidebar } from './components/NodeLibrarySidebar';
import { OSDEditorModal } from './components/OSDEditorModal';
import { PipelineCanvas } from './components/PipelineCanvas';
import { TopBar } from './components/TopBar';
import { DEFAULT_OSD_ELEMENTS, NODE_CATALOG_MAP } from './data/nodeCatalog';
import { PIPELINE_TEMPLATES } from './data/templates';
import { generatePythonOpenCVCode } from './engine/codeGenerator';
import {
  clearExecutionCache,
  ensureOpenCVReady,
  executePipeline,
} from './engine/cvEngine';
import { getSampleImageData } from './engine/sampleImages';
import {
  ExecutionLogEntry,
  NodeExecutionResult,
  OSDElement,
  PipelineEdge,
  PipelineNode,
  PipelineTemplate,
} from './types/pipeline';

function cloneNodes(nodes: PipelineNode[]): PipelineNode[] {
  return nodes.map((n) => ({
    ...n,
    position: { ...n.position },
    params: structuredClone(n.params),
  }));
}

function cloneEdges(edges: PipelineEdge[]): PipelineEdge[] {
  return edges.map((e) => ({ ...e }));
}

function triggerDownload(filename: string, content: string, mime = 'text/plain') {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function App() {
  const initialTemplate = PIPELINE_TEMPLATES[0];

  const [nodes, setNodes] = useState<PipelineNode[]>(() => cloneNodes(initialTemplate.nodes));
  const [edges, setEdges] = useState<PipelineEdge[]>(() => cloneEdges(initialTemplate.edges));
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(
    initialTemplate.defaultSelectedNodeId ?? null
  );
  const [seed, setSeed] = useState<number>(0);
  const [samples, setSamples] = useState<number>(1);
  const [results, setResults] = useState<Record<string, NodeExecutionResult>>({});
  const [logs, setLogs] = useState<ExecutionLogEntry[]>([]);
  const [totalMs, setTotalMs] = useState<number>(0);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [logCollapsed, setLogCollapsed] = useState<boolean>(false);
  const [fitTrigger, setFitTrigger] = useState<number>(0);
  const [wasmReady, setWasmReady] = useState<boolean>(false);

  // Responsive mobile/tablet drawers
  const [leftDrawerOpen, setLeftDrawerOpen] = useState(false);
  const [rightDrawerOpen, setRightDrawerOpen] = useState(false);

  // OSD Drag & Drop Canvas Editor modal state
  const [osdEditorNodeId, setOsdEditorNodeId] = useState<string | null>(null);
  const [aiModalOpen, setAiModalOpen] = useState<boolean>(false);

  useEffect(() => {
    let mounted = true;
    ensureOpenCVReady()
      .then(() => {
        if (mounted) setWasmReady(true);
      })
      .catch((err) => {
        console.error('OpenCV WASM initialization failed:', err);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const runGraph = useCallback(
    async (
      targetNodes: PipelineNode[],
      targetEdges: PipelineEdge[],
      stopAtNodeId?: string
    ) => {
      setIsRunning(true);
      try {
        const exec = await executePipeline({
          nodes: targetNodes,
          edges: targetEdges,
          seed,
          stopAtNodeId,
        });
        setResults((prev) => (stopAtNodeId ? { ...prev, ...exec.results } : exec.results));
        setLogs(exec.logs);
        setTotalMs(exec.totalMs);
      } catch (err) {
        setLogs((prev) => [
          ...prev,
          {
            id: `err-${Date.now()}`,
            message: err instanceof Error ? err.message : 'OpenCV execution error',
            kind: 'error',
            timestamp: Date.now(),
          },
        ]);
      } finally {
        setIsRunning(false);
      }
    },
    [seed]
  );

  // Execute initial pipeline on mount
  useEffect(() => {
    runGraph(nodes, edges);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRunAll = () => {
    runGraph(nodes, edges);
  };

  const handleRunToSelected = () => {
    if (!selectedNodeId) return;
    runGraph(nodes, edges, selectedNodeId);
  };

  const handleAddNodeAt = useCallback(
    (type: string, position: { x: number; y: number }) => {
      const meta = NODE_CATALOG_MAP[type];
      if (!meta) return;
      const params: Record<string, unknown> = {};
      meta.params.forEach((p) => {
        params[p.key] = structuredClone(p.defaultValue);
      });
      const newNode: PipelineNode = {
        id: `n_${type}_${Date.now().toString(36)}`,
        type,
        position,
        params,
      };
      const nextNodes = [...nodes, newNode];
      setNodes(nextNodes);
      setSelectedNodeId(newNode.id);
      setLeftDrawerOpen(false);
      runGraph(nextNodes, edges);
    },
    [nodes, edges, runGraph]
  );

  const handleAddNodeDefaultPos = useCallback(
    (type: string) => {
      const offset = (nodes.length % 6) * 28;
      handleAddNodeAt(type, { x: 280 + offset, y: 180 + offset });
    },
    [nodes.length, handleAddNodeAt]
  );

  const handleMoveNode = useCallback((id: string, pos: { x: number; y: number }) => {
    setNodes((prev) => prev.map((n) => (n.id === id ? { ...n, position: pos } : n)));
  }, []);

  const handleConnect = useCallback(
    (source: string, sourceHandle: string, target: string, targetHandle: string) => {
      // Prevent duplicate target port connection or self-loop
      if (source === target) return;
      const filtered = edges.filter(
        (e) => !(e.target === target && e.targetHandle === targetHandle)
      );
      const newEdge: PipelineEdge = {
        id: `e_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
        source,
        sourceHandle,
        target,
        targetHandle,
      };
      const nextEdges = [...filtered, newEdge];
      setEdges(nextEdges);
      runGraph(nodes, nextEdges);
    },
    [edges, nodes, runGraph]
  );

  const handleDeleteEdge = useCallback(
    (edgeId: string) => {
      const nextEdges = edges.filter((e) => e.id !== edgeId);
      setEdges(nextEdges);
      runGraph(nodes, nextEdges);
    },
    [edges, nodes, runGraph]
  );

  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      const nextNodes = nodes.filter((n) => n.id !== nodeId);
      const nextEdges = edges.filter((e) => e.source !== nodeId && e.target !== nodeId);
      setNodes(nextNodes);
      setEdges(nextEdges);
      if (selectedNodeId === nodeId) setSelectedNodeId(null);
      runGraph(nextNodes, nextEdges);
    },
    [nodes, edges, selectedNodeId, runGraph]
  );

  const handleDuplicateNode = useCallback(
    (nodeId: string) => {
      const src = nodes.find((n) => n.id === nodeId);
      if (!src) return;
      const copy: PipelineNode = {
        id: `n_${src.type}_${Date.now().toString(36)}`,
        type: src.type,
        position: { x: src.position.x + 36, y: src.position.y + 36 },
        params: structuredClone(src.params),
      };
      const nextNodes = [...nodes, copy];
      setNodes(nextNodes);
      setSelectedNodeId(copy.id);
      runGraph(nextNodes, edges);
    },
    [nodes, edges, runGraph]
  );

  const handleUpdateParam = useCallback(
    (nodeId: string, key: string, value: unknown) => {
      const nextNodes = nodes.map((n) =>
        n.id === nodeId ? { ...n, params: { ...n.params, [key]: value } } : n
      );
      setNodes(nextNodes);
      runGraph(nextNodes, edges);
    },
    [nodes, edges, runGraph]
  );

  const handleSelectTemplate = useCallback(
    (tpl: PipelineTemplate) => {
      clearExecutionCache();
      const nextNodes = cloneNodes(tpl.nodes);
      const nextEdges = cloneEdges(tpl.edges);
      setNodes(nextNodes);
      setEdges(nextEdges);
      setSelectedNodeId(tpl.defaultSelectedNodeId ?? nextNodes[0]?.id ?? null);
      setFitTrigger((c) => c + 1);
      runGraph(nextNodes, nextEdges);
    },
    [runGraph]
  );

  const handleLoadJsonFile = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const parsed = JSON.parse(String(reader.result));
          if (Array.isArray(parsed.nodes) && Array.isArray(parsed.edges)) {
            clearExecutionCache();
            setNodes(parsed.nodes);
            setEdges(parsed.edges);
            setSelectedNodeId(parsed.nodes[0]?.id ?? null);
            setFitTrigger((c) => c + 1);
            runGraph(parsed.nodes, parsed.edges);
          }
        } catch {
          setLogs((prev) => [
            ...prev,
            {
              id: `err-${Date.now()}`,
              message: 'Failed to parse pipeline JSON file.',
              kind: 'error',
              timestamp: Date.now(),
            },
          ]);
        }
      };
      reader.readAsText(file);
    },
    [runGraph]
  );

  const handleExportPython = () => {
    const code = generatePythonOpenCVCode(nodes, edges);
    triggerDownload('vision_studio_pipeline.py', code, 'text/x-python');
  };

  const handleExportJson = () => {
    const payload = JSON.stringify({ name: 'Vision Studio Pipeline', nodes, edges }, null, 2);
    triggerDownload('vision_studio_pipeline.json', payload, 'application/json');
  };

  const handleExportNodeImage = (nodeId?: string) => {
    const targetId = nodeId || selectedNodeId || nodes[nodes.length - 1]?.id;
    const res = targetId ? results[targetId] : undefined;
    if (!res?.previewDataUrl) return;
    const a = document.createElement('a');
    a.href = res.previewDataUrl;
    a.download = `vision_studio_${targetId}.png`;
    a.click();
  };

  // Launch OSD Drag & Drop Editor from TopBar or Node
  const handleOpenOSDEditorQuick = () => {
    const existingOsd =
      nodes.find((n) => n.id === selectedNodeId && n.type === 'osd_canvas') ||
      nodes.find((n) => n.type === 'osd_canvas');

    if (existingOsd) {
      setSelectedNodeId(existingOsd.id);
      setOsdEditorNodeId(existingOsd.id);
      return;
    }

    // Create an OSD Canvas node attached to the currently selected or last node!
    const anchor =
      (selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : null) ||
      nodes[nodes.length - 1];
    const newPos = anchor
      ? { x: anchor.position.x + 195, y: anchor.position.y }
      : { x: 340, y: 200 };
    const newNode: PipelineNode = {
      id: `n_osd_canvas_${Date.now().toString(36)}`,
      type: 'osd_canvas',
      position: newPos,
      params: {
        elements: structuredClone(DEFAULT_OSD_ELEMENTS),
        globalOpacity: 1.0,
        showGridBurnIn: false,
      },
    };
    const nextNodes = [...nodes, newNode];
    const nextEdges = anchor
      ? [
          ...edges,
          {
            id: `e_osd_${Date.now().toString(36)}`,
            source: anchor.id,
            sourceHandle: NODE_CATALOG_MAP[anchor.type]?.outputs[0]?.id || 'image',
            target: newNode.id,
            targetHandle: 'image',
          },
        ]
      : edges;
    setNodes(nextNodes);
    setEdges(nextEdges);
    setSelectedNodeId(newNode.id);
    setOsdEditorNodeId(newNode.id);
    runGraph(nextNodes, nextEdges);
  };

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedNodeId) || null,
    [nodes, selectedNodeId]
  );

  const osdEditorNode = useMemo(
    () => nodes.find((n) => n.id === osdEditorNodeId) || null,
    [nodes, osdEditorNodeId]
  );

  const osdBaseImageData = useMemo(() => {
    if (!osdEditorNodeId) return getSampleImageData('portrait');
    const incomingEdge = edges.find((e) => e.target === osdEditorNodeId);
    if (incomingEdge && results[incomingEdge.source]?.imageData) {
      return results[incomingEdge.source].imageData!;
    }
    if (results[osdEditorNodeId]?.imageData) {
      return results[osdEditorNodeId].imageData!;
    }
    return getSampleImageData('portrait');
  }, [osdEditorNodeId, edges, results]);

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0B0D10] text-[#E6EDF3] overflow-hidden">
      <TopBar
        nodeCount={nodes.length}
        linkCount={edges.length}
        seed={seed}
        samples={samples}
        onSeedChange={setSeed}
        onSamplesChange={setSamples}
        onRun={handleRunAll}
        onRunToSelected={handleRunToSelected}
        hasSelectedNode={Boolean(selectedNodeId)}
        isRunning={isRunning}
        onSelectTemplate={handleSelectTemplate}
        onLoadJsonFile={handleLoadJsonFile}
        onExportPython={handleExportPython}
        onExportJson={handleExportJson}
        onExportImage={() => handleExportNodeImage()}
        onOpenOSDEditorQuick={handleOpenOSDEditorQuick}
        onOpenAIGenerate={() => setAiModalOpen(true)}
        wasmReady={wasmReady}
        leftDrawerOpen={leftDrawerOpen}
        rightDrawerOpen={rightDrawerOpen}
        onToggleLeftDrawer={() => {
          setLeftDrawerOpen((v) => !v);
          setRightDrawerOpen(false);
        }}
        onToggleRightDrawer={() => {
          setRightDrawerOpen((v) => !v);
          setLeftDrawerOpen(false);
        }}
      />

      {/* Main 3-Pane Studio Workspace */}
      <div className="relative flex-1 flex min-h-0 overflow-hidden">
        {/* Desktop Left Sidebar */}
        <NodeLibrarySidebar onAddNode={handleAddNodeDefaultPos} className="hidden lg:flex" />

        {/* Mobile/Tablet Left Drawer */}
        {leftDrawerOpen && (
          <div className="lg:hidden fixed inset-0 top-14 z-40 flex">
            <NodeLibrarySidebar onAddNode={handleAddNodeDefaultPos} className="flex shadow-2xl" />
            <div
              onClick={() => setLeftDrawerOpen(false)}
              className="flex-1 bg-black/60"
            />
          </div>
        )}

        {/* Center Viewport: Pipeline Graph + Bottom Execution Log */}
        <main className="flex-1 flex flex-col min-w-0 min-h-0">
          <PipelineCanvas
            nodes={nodes}
            edges={edges}
            selectedNodeId={selectedNodeId}
            results={results}
            onSelectNode={(id) => setSelectedNodeId(id)}
            onMoveNode={handleMoveNode}
            onAddNodeAt={handleAddNodeAt}
            onConnect={handleConnect}
            onDeleteEdge={handleDeleteEdge}
            onDeleteNode={handleDeleteNode}
            onDuplicateNode={handleDuplicateNode}
            onRunToNode={(id) => runGraph(nodes, edges, id)}
            onOpenOSDEditor={(id) => setOsdEditorNodeId(id)}
            fitTrigger={fitTrigger}
          />

          <ExecutionLog
            logs={logs}
            nodeCount={nodes.length}
            totalMs={totalMs}
            collapsed={logCollapsed}
            onToggleCollapse={() => setLogCollapsed((v) => !v)}
            onClear={() => setLogs([])}
          />
        </main>

        {/* Desktop Right Inspector Sidebar */}
        <InspectorSidebar
          selectedNode={selectedNode}
          selectedResult={selectedNodeId ? results[selectedNodeId] ?? null : null}
          nodes={nodes}
          edges={edges}
          onUpdateParam={handleUpdateParam}
          onExportPython={handleExportPython}
          onExportNodeImage={(id) => handleExportNodeImage(id)}
          onOpenOSDEditor={(id) => setOsdEditorNodeId(id)}
          className="hidden lg:flex"
        />

        {/* Mobile/Tablet Right Drawer */}
        {rightDrawerOpen && (
          <div className="lg:hidden fixed inset-0 top-14 z-40 flex justify-end">
            <div
              onClick={() => setRightDrawerOpen(false)}
              className="flex-1 bg-black/60"
            />
            <InspectorSidebar
              selectedNode={selectedNode}
              selectedResult={selectedNodeId ? results[selectedNodeId] ?? null : null}
              nodes={nodes}
              edges={edges}
              onUpdateParam={handleUpdateParam}
              onExportPython={handleExportPython}
              onExportNodeImage={(id) => handleExportNodeImage(id)}
              onOpenOSDEditor={(id) => {
                setRightDrawerOpen(false);
                setOsdEditorNodeId(id);
              }}
              className="flex shadow-2xl"
            />
          </div>
        )}
      </div>

      {/* Interactive Drag & Drop OSD Drawing Canvas Editor Modal */}
      {osdEditorNode && (
        <OSDEditorModal
          open={Boolean(osdEditorNode)}
          nodeTitle={NODE_CATALOG_MAP[osdEditorNode.type]?.title || 'OSD Canvas'}
          baseImageData={osdBaseImageData}
          initialElements={
            Array.isArray(osdEditorNode.params.elements)
              ? (osdEditorNode.params.elements as OSDElement[])
              : DEFAULT_OSD_ELEMENTS
          }
          initialOpacity={Number(osdEditorNode.params.globalOpacity ?? 1)}
          initialGridBurnIn={Boolean(osdEditorNode.params.showGridBurnIn)}
          onClose={() => setOsdEditorNodeId(null)}
          onApply={(elements, globalOpacity, showGridBurnIn) => {
            const nextNodes = nodes.map((n) =>
              n.id === osdEditorNode.id
                ? {
                    ...n,
                    params: {
                      ...n.params,
                      elements,
                      globalOpacity,
                      showGridBurnIn,
                    },
                  }
                : n
            );
            setNodes(nextNodes);
            setOsdEditorNodeId(null);
            runGraph(nextNodes, edges);
          }}
        />
      )}

      {/* AI Prompt Pipeline Generator Modal */}
      <AIGenerateModal
        open={aiModalOpen}
        onClose={() => setAiModalOpen(false)}
        onApplyPipeline={(generated) => {
          clearExecutionCache();
          const nextNodes = cloneNodes(generated.nodes);
          const nextEdges = cloneEdges(generated.edges);
          setNodes(nextNodes);
          setEdges(nextEdges);
          setSelectedNodeId(nextNodes[nextNodes.length - 1]?.id ?? nextNodes[0]?.id ?? null);
          setFitTrigger((c) => c + 1);
          runGraph(nextNodes, nextEdges);
        }}
      />
    </div>
  );
}
