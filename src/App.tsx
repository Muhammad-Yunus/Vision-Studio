import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AIGenerateModal } from './components/AIGenerateModal';
import { BatchSamplesModal } from './components/BatchSamplesModal';
import { ExecutionLog } from './components/ExecutionLog';
import { GeometryEditorModal } from './components/GeometryEditorModal';
import { InspectorSidebar } from './components/InspectorSidebar';
import { NodeLibrarySidebar } from './components/NodeLibrarySidebar';
import { OSDEditorModal } from './components/OSDEditorModal';
import { PipelineCanvas } from './components/PipelineCanvas';
import { ResultPreviewModal } from './components/ResultPreviewModal';
import { TopBar } from './components/TopBar';
import { DEFAULT_OSD_ELEMENTS, NODE_CATALOG_MAP } from './data/nodeCatalog';
import { PIPELINE_TEMPLATES } from './data/templates';
import { generatePythonOpenCVCode } from './engine/codeGenerator';
import {
  clearExecutionCache,
  ensureOpenCVReady,
  executeBatchSamples,
  executePipeline,
} from './engine/cvEngine';
import { getSampleImageData } from './engine/sampleImages';
import {
  BatchSampleResult,
  ExecutionLogEntry,
  NodeExecutionResult,
  OSDElement,
  PipelineEdge,
  PipelineNode,
  PipelineTemplate,
} from './types/pipeline';

interface GraphSnapshot {
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  selectedNodeId: string | null;
}

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

function computeAutoLayout(nodes: PipelineNode[], edges: PipelineEdge[]): PipelineNode[] {
  if (nodes.length === 0) return nodes;
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const parents = new Map<string, string[]>();
  const children = new Map<string, string[]>();
  const inDeg = new Map<string, number>();

  nodes.forEach((n) => {
    parents.set(n.id, []);
    children.set(n.id, []);
    inDeg.set(n.id, 0);
  });

  edges.forEach((e) => {
    if (nodeMap.has(e.source) && nodeMap.has(e.target)) {
      parents.get(e.target)!.push(e.source);
      children.get(e.source)!.push(e.target);
      inDeg.set(e.target, (inDeg.get(e.target) ?? 0) + 1);
    }
  });

  const level = new Map<string, number>();
  const queue: string[] = [];
  inDeg.forEach((d, id) => {
    if (d === 0) {
      level.set(id, 0);
      queue.push(id);
    }
  });

  while (queue.length > 0) {
    const id = queue.shift()!;
    const curLvl = level.get(id) ?? 0;
    for (const next of children.get(id) || []) {
      level.set(next, Math.max(level.get(next) ?? 0, curLvl + 1));
      const nd = (inDeg.get(next) ?? 1) - 1;
      inDeg.set(next, nd);
      if (nd === 0) queue.push(next);
    }
  }

  nodes.forEach((n) => {
    if (!level.has(n.id)) level.set(n.id, 0);
  });

  const columns = new Map<number, PipelineNode[]>();
  nodes.forEach((n) => {
    const col = level.get(n.id) ?? 0;
    const arr = columns.get(col) || [];
    arr.push(n);
    columns.set(col, arr);
  });

  const sortedCols = Array.from(columns.keys()).sort((a, b) => a - b);
  const posMap = new Map<string, { x: number; y: number }>();

  sortedCols.forEach((colIdx) => {
    const colNodes = columns.get(colIdx)!;
    colNodes.sort((a, b) => {
      const pA = parents.get(a.id) || [];
      const pB = parents.get(b.id) || [];
      const avgA =
        pA.length > 0
          ? pA.reduce((s, pid) => s + (posMap.get(pid)?.y ?? nodeMap.get(pid)?.position.y ?? 0), 0) / pA.length
          : a.position.y;
      const avgB =
        pB.length > 0
          ? pB.reduce((s, pid) => s + (posMap.get(pid)?.y ?? nodeMap.get(pid)?.position.y ?? 0), 0) / pB.length
          : b.position.y;
      return avgA - avgB;
    });

    const x = 36 + colIdx * 210;
    const totalHeight = (colNodes.length - 1) * 172;
    const baseY = Math.max(28, Math.round(240 - totalHeight / 2));
    colNodes.forEach((n, rowIdx) => {
      posMap.set(n.id, { x, y: baseY + rowIdx * 172 });
    });
  });

  return nodes.map((n) => ({
    ...n,
    position: posMap.get(n.id) || n.position,
  }));
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

  // Undo / Redo History
  const [past, setPast] = useState<GraphSnapshot[]>([]);
  const [future, setFuture] = useState<GraphSnapshot[]>([]);

  // Responsive mobile/tablet drawers
  const [leftDrawerOpen, setLeftDrawerOpen] = useState(false);
  const [rightDrawerOpen, setRightDrawerOpen] = useState(false);

  // Modals
  const [osdEditorNodeId, setOsdEditorNodeId] = useState<string | null>(null);
  const [geometryEditorNodeId, setGeometryEditorNodeId] = useState<string | null>(null);
  const [aiModalOpen, setAiModalOpen] = useState<boolean>(false);
  const [batchModalOpen, setBatchModalOpen] = useState<boolean>(false);
  const [batchResults, setBatchResults] = useState<BatchSampleResult[]>([]);
  const [batchLoading, setBatchLoading] = useState<boolean>(false);
  const [resultModalOpen, setResultModalOpen] = useState<boolean>(false);

  // Live Webcam Stream State
  const [webcamActive, setWebcamActive] = useState<boolean>(false);
  const [webcamFps, setWebcamFps] = useState<number>(0);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const webcamStreamRef = useRef<MediaStream | null>(null);
  const webcamVideoRef = useRef<HTMLVideoElement | null>(null);
  const webcamCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  const selectedNodeIdRef = useRef(selectedNodeId);
  const seedRef = useRef(seed);
  nodesRef.current = nodes;
  edgesRef.current = edges;
  selectedNodeIdRef.current = selectedNodeId;
  seedRef.current = seed;

  const pushHistorySnapshot = useCallback(() => {
    const snap: GraphSnapshot = {
      nodes: cloneNodes(nodesRef.current),
      edges: cloneEdges(edgesRef.current),
      selectedNodeId: selectedNodeIdRef.current,
    };
    setPast((prev) => [...prev.slice(-49), snap]);
    setFuture([]);
  }, []);

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
      stopAtNodeId?: string,
      overrideSeed?: number,
      webcamFrameData?: ImageData | null
    ) => {
      setIsRunning(true);
      try {
        const exec = await executePipeline({
          nodes: targetNodes,
          edges: targetEdges,
          seed: overrideSeed ?? seedRef.current,
          stopAtNodeId,
          webcamFrameData,
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
    []
  );

  // Execute initial pipeline on mount
  useEffect(() => {
    runGraph(nodes, edges);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-execute when seed changes
  const handleSeedChange = useCallback(
    (nextSeed: number) => {
      setSeed(nextSeed);
      runGraph(nodesRef.current, edgesRef.current, undefined, nextSeed);
    },
    [runGraph]
  );

  const stopWebcam = useCallback(() => {
    if (webcamStreamRef.current) {
      webcamStreamRef.current.getTracks().forEach((t) => t.stop());
      webcamStreamRef.current = null;
    }
    if (webcamVideoRef.current) {
      webcamVideoRef.current.srcObject = null;
    }
    setWebcamActive(false);
    setWebcamFps(0);
  }, []);

  const startWebcam = useCallback(async () => {
    setWebcamError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setWebcamError('Browser camera API (getUserMedia) is unavailable in this context.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
      webcamStreamRef.current = stream;
      if (!webcamVideoRef.current) {
        const v = document.createElement('video');
        v.muted = true;
        v.playsInline = true;
        webcamVideoRef.current = v;
      }
      webcamVideoRef.current.srcObject = stream;
      await webcamVideoRef.current.play();
      if (!webcamCanvasRef.current) {
        webcamCanvasRef.current = document.createElement('canvas');
      }
      setWebcamActive(true);
    } catch (err) {
      setWebcamError(
        err instanceof Error ? err.message : 'Camera permission denied or no webcam found.'
      );
      stopWebcam();
    }
  }, [stopWebcam]);

  const handleToggleWebcam = useCallback(() => {
    if (webcamActive) stopWebcam();
    else startWebcam();
  }, [webcamActive, startWebcam, stopWebcam]);

  useEffect(() => {
    return () => stopWebcam();
  }, [stopWebcam]);

  // Live webcam frame loop when active
  useEffect(() => {
    if (!webcamActive) return;
    let cancelled = false;
    let busy = false;
    let lastFrameAt = 0;
    let fpsFrames = 0;
    let fpsWindowStart = performance.now();

    const tick = async () => {
      if (cancelled) return;
      const curNodes = nodesRef.current;
      const camNode = curNodes.find((n) => n.type === 'webcam_input');
      const video = webcamVideoRef.current;
      const canvas = webcamCanvasRef.current;

      if (!camNode) {
        stopWebcam();
        return;
      }

      const targetFps = Math.max(2, Math.min(24, Number(camNode.params.targetFps ?? 10)));
      const minInterval = 1000 / targetFps;
      const now = performance.now();

      if (!busy && video && canvas && video.readyState >= 2 && now - lastFrameAt >= minInterval) {
        busy = true;
        lastFrameAt = now;
        try {
          const [rw, rh] = String(camNode.params.resolution || '360x270')
            .split('x')
            .map(Number);
          const cw = rw || 360;
          const ch = rh || 270;
          canvas.width = cw;
          canvas.height = ch;
          const ctx = canvas.getContext('2d', { willReadFrequently: true });
          if (ctx) {
            ctx.save();
            if (camNode.params.mirror !== false) {
              ctx.translate(cw, 0);
              ctx.scale(-1, 1);
            }
            ctx.drawImage(video, 0, 0, cw, ch);
            ctx.restore();
            const frameData = ctx.getImageData(0, 0, cw, ch);
            const exec = await executePipeline({
              nodes: curNodes,
              edges: edgesRef.current,
              seed: seedRef.current,
              webcamFrameData: frameData,
              skipCache: true,
            });
            if (!cancelled) {
              setResults(exec.results);
              setTotalMs(exec.totalMs);
              fpsFrames++;
              const elapsed = performance.now() - fpsWindowStart;
              if (elapsed >= 800) {
                setWebcamFps(Math.round((fpsFrames * 1000) / elapsed));
                fpsFrames = 0;
                fpsWindowStart = performance.now();
              }
            }
          }
        } catch {
          // ignore transient frame read error
        } finally {
          busy = false;
        }
      }

      if (!cancelled) {
        requestAnimationFrame(tick);
      }
    };

    const rafId = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
    };
  }, [webcamActive, stopWebcam]);

  const handleOpenBatchModal = useCallback(async () => {
    setBatchModalOpen(true);
    setBatchLoading(true);
    try {
      const items = await executeBatchSamples({
        nodes: nodesRef.current,
        edges: edgesRef.current,
        baseSeed: seedRef.current,
        count: samples,
        targetNodeId: selectedNodeIdRef.current,
      });
      setBatchResults(items);
    } finally {
      setBatchLoading(false);
    }
  }, [samples]);

  const handleRunAll = () => {
    runGraph(nodes, edges);
    if (samples > 1) {
      handleOpenBatchModal();
    }
  };

  const handleRunToSelected = () => {
    if (!selectedNodeId) return;
    runGraph(nodes, edges, selectedNodeId);
  };

  const handleUndo = useCallback(() => {
    setPast((prev) => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      setFuture((f) => [
        {
          nodes: cloneNodes(nodesRef.current),
          edges: cloneEdges(edgesRef.current),
          selectedNodeId: selectedNodeIdRef.current,
        },
        ...f.slice(0, 49),
      ]);
      setNodes(last.nodes);
      setEdges(last.edges);
      setSelectedNodeId(last.selectedNodeId);
      runGraph(last.nodes, last.edges);
      return prev.slice(0, -1);
    });
  }, [runGraph]);

  const handleRedo = useCallback(() => {
    setFuture((prev) => {
      if (prev.length === 0) return prev;
      const [next, ...rest] = prev;
      setPast((p) => [
        ...p.slice(-49),
        {
          nodes: cloneNodes(nodesRef.current),
          edges: cloneEdges(edgesRef.current),
          selectedNodeId: selectedNodeIdRef.current,
        },
      ]);
      setNodes(next.nodes);
      setEdges(next.edges);
      setSelectedNodeId(next.selectedNodeId);
      runGraph(next.nodes, next.edges);
      return rest;
    });
  }, [runGraph]);

  const handleToggleBypass = useCallback(
    (nodeId: string) => {
      pushHistorySnapshot();
      const nextNodes = nodesRef.current.map((n) =>
        n.id === nodeId ? { ...n, bypassed: !n.bypassed } : n
      );
      setNodes(nextNodes);
      runGraph(nextNodes, edgesRef.current);
    },
    [pushHistorySnapshot, runGraph]
  );

  const handleAutoLayout = useCallback(() => {
    if (nodesRef.current.length === 0) return;
    pushHistorySnapshot();
    const laidOut = computeAutoLayout(nodesRef.current, edgesRef.current);
    setNodes(laidOut);
    setFitTrigger((c) => c + 1);
  }, [pushHistorySnapshot]);

  // Global keyboard shortcuts: Ctrl+Z (Undo), Ctrl+Y / Ctrl+Shift+Z (Redo), M (Bypass selected node)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || target?.isContentEditable) {
        return;
      }

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) handleRedo();
        else handleUndo();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        handleRedo();
        return;
      }
      if (!e.ctrlKey && !e.metaKey && e.key.toLowerCase() === 'm' && selectedNodeIdRef.current) {
        const sel = nodesRef.current.find((n) => n.id === selectedNodeIdRef.current);
        if (sel && sel.type !== 'load_image' && sel.type !== 'webcam_input') {
          e.preventDefault();
          handleToggleBypass(sel.id);
        }
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handleUndo, handleRedo, handleToggleBypass]);

  const handleAddNodeAt = useCallback(
    (type: string, position: { x: number; y: number }) => {
      const meta = NODE_CATALOG_MAP[type];
      if (!meta) return;
      pushHistorySnapshot();
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
    [nodes, edges, pushHistorySnapshot, runGraph]
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
      if (source === target) return;
      pushHistorySnapshot();
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
    [edges, nodes, pushHistorySnapshot, runGraph]
  );

  const handleDeleteEdge = useCallback(
    (edgeId: string) => {
      pushHistorySnapshot();
      const nextEdges = edges.filter((e) => e.id !== edgeId);
      setEdges(nextEdges);
      runGraph(nodes, nextEdges);
    },
    [edges, nodes, pushHistorySnapshot, runGraph]
  );

  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      pushHistorySnapshot();
      const nextNodes = nodes.filter((n) => n.id !== nodeId);
      const nextEdges = edges.filter((e) => e.source !== nodeId && e.target !== nodeId);
      setNodes(nextNodes);
      setEdges(nextEdges);
      if (selectedNodeId === nodeId) setSelectedNodeId(null);
      runGraph(nextNodes, nextEdges);
    },
    [nodes, edges, selectedNodeId, pushHistorySnapshot, runGraph]
  );

  const handleDuplicateNode = useCallback(
    (nodeId: string) => {
      const src = nodes.find((n) => n.id === nodeId);
      if (!src) return;
      pushHistorySnapshot();
      const copy: PipelineNode = {
        id: `n_${src.type}_${Date.now().toString(36)}`,
        type: src.type,
        position: { x: src.position.x + 36, y: src.position.y + 36 },
        params: structuredClone(src.params),
        bypassed: src.bypassed,
      };
      const nextNodes = [...nodes, copy];
      setNodes(nextNodes);
      setSelectedNodeId(copy.id);
      runGraph(nextNodes, edges);
    },
    [nodes, edges, pushHistorySnapshot, runGraph]
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
      pushHistorySnapshot();
      clearExecutionCache();
      stopWebcam();
      const nextNodes = cloneNodes(tpl.nodes);
      const nextEdges = cloneEdges(tpl.edges);
      setNodes(nextNodes);
      setEdges(nextEdges);
      setSelectedNodeId(tpl.defaultSelectedNodeId ?? nextNodes[0]?.id ?? null);
      setFitTrigger((c) => c + 1);
      runGraph(nextNodes, nextEdges);
    },
    [pushHistorySnapshot, runGraph, stopWebcam]
  );

  const handleLoadJsonFile = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const parsed = JSON.parse(String(reader.result));
          if (Array.isArray(parsed.nodes) && Array.isArray(parsed.edges)) {
            pushHistorySnapshot();
            clearExecutionCache();
            stopWebcam();
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
    [pushHistorySnapshot, runGraph, stopWebcam]
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

    pushHistorySnapshot();
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

  const geometryEditorNode = useMemo(
    () => nodes.find((n) => n.id === geometryEditorNodeId) || null,
    [nodes, geometryEditorNodeId]
  );

  const getUpstreamBaseImage = useCallback(
    (nodeId: string | null): ImageData => {
      if (!nodeId) return getSampleImageData('portrait');
      const incomingEdge = edges.find((e) => e.target === nodeId);
      if (incomingEdge && results[incomingEdge.source]?.imageData) {
        return results[incomingEdge.source].imageData!;
      }
      if (results[nodeId]?.imageData) {
        return results[nodeId].imageData!;
      }
      return getSampleImageData('portrait');
    },
    [edges, results]
  );

  const osdBaseImageData = useMemo(
    () => getUpstreamBaseImage(osdEditorNodeId),
    [osdEditorNodeId, getUpstreamBaseImage]
  );

  const osdBaseMetrics = useMemo(() => {
    if (!osdEditorNodeId) return undefined;
    const metricsEdge = edges.find((e) => e.target === osdEditorNodeId && e.targetHandle === 'metrics');
    if (metricsEdge && results[metricsEdge.source]?.metrics) {
      return results[metricsEdge.source].metrics;
    }
    const imageEdge = edges.find((e) => e.target === osdEditorNodeId);
    if (imageEdge && results[imageEdge.source]?.metrics) {
      return results[imageEdge.source].metrics;
    }
    return undefined;
  }, [osdEditorNodeId, edges, results]);

  const geometryBaseImageData = useMemo(
    () => getUpstreamBaseImage(geometryEditorNodeId),
    [geometryEditorNodeId, getUpstreamBaseImage]
  );

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0B0D10] text-[#E6EDF3] overflow-hidden">
      <TopBar
        nodeCount={nodes.length}
        linkCount={edges.length}
        seed={seed}
        samples={samples}
        onSeedChange={handleSeedChange}
        onSamplesChange={setSamples}
        onRun={handleRunAll}
        onRunToSelected={handleRunToSelected}
        hasSelectedNode={Boolean(selectedNodeId)}
        isRunning={isRunning}
        canUndo={past.length > 0}
        canRedo={future.length > 0}
        onUndo={handleUndo}
        onRedo={handleRedo}
        onAutoLayout={handleAutoLayout}
        onOpenBatchModal={handleOpenBatchModal}
        webcamActive={webcamActive}
        webcamFps={webcamFps}
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
            onNodeDragStart={pushHistorySnapshot}
            onMoveNode={handleMoveNode}
            onAddNodeAt={handleAddNodeAt}
            onConnect={handleConnect}
            onDeleteEdge={handleDeleteEdge}
            onDeleteNode={handleDeleteNode}
            onDuplicateNode={handleDuplicateNode}
            onToggleBypass={handleToggleBypass}
            onRunToNode={(id) => runGraph(nodes, edges, id)}
            onOpenOSDEditor={(id) => setOsdEditorNodeId(id)}
            onOpenGeometryEditor={(id) => setGeometryEditorNodeId(id)}
            webcamActive={webcamActive}
            onToggleWebcam={handleToggleWebcam}
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
          onToggleBypass={handleToggleBypass}
          onExportPython={handleExportPython}
          onExportNodeImage={(id) => handleExportNodeImage(id)}
          onOpenOSDEditor={(id) => setOsdEditorNodeId(id)}
          onOpenGeometryEditor={(id) => setGeometryEditorNodeId(id)}
          webcamActive={webcamActive}
          webcamFps={webcamFps}
          webcamError={webcamError}
          onToggleWebcam={handleToggleWebcam}
          samplesCount={samples}
          onOpenBatchModal={handleOpenBatchModal}
          onExpandResult={() => setResultModalOpen(true)}
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
              onToggleBypass={handleToggleBypass}
              onExportPython={handleExportPython}
              onExportNodeImage={(id) => handleExportNodeImage(id)}
              onOpenOSDEditor={(id) => {
                setRightDrawerOpen(false);
                setOsdEditorNodeId(id);
              }}
              onOpenGeometryEditor={(id) => {
                setRightDrawerOpen(false);
                setGeometryEditorNodeId(id);
              }}
              webcamActive={webcamActive}
              webcamFps={webcamFps}
              webcamError={webcamError}
              onToggleWebcam={handleToggleWebcam}
              samplesCount={samples}
              onOpenBatchModal={handleOpenBatchModal}
              onExpandResult={() => {
                setRightDrawerOpen(false);
                setResultModalOpen(true);
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
          metrics={osdBaseMetrics}
          initialElements={
            Array.isArray(osdEditorNode.params.elements)
              ? (osdEditorNode.params.elements as OSDElement[])
              : DEFAULT_OSD_ELEMENTS
          }
          initialOpacity={Number(osdEditorNode.params.globalOpacity ?? 1)}
          initialGridBurnIn={Boolean(osdEditorNode.params.showGridBurnIn)}
          onClose={() => setOsdEditorNodeId(null)}
          onApply={(elements, globalOpacity, showGridBurnIn) => {
            pushHistorySnapshot();
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

      {/* Interactive Perspective Corner & Crop ROI Visual Picker Modal */}
      {geometryEditorNode &&
        (geometryEditorNode.type === 'warp_perspective' || geometryEditorNode.type === 'crop_roi') && (
          <GeometryEditorModal
            open={Boolean(geometryEditorNode)}
            mode={geometryEditorNode.type}
            nodeTitle={NODE_CATALOG_MAP[geometryEditorNode.type]?.title || 'Geometry Picker'}
            baseImageData={geometryBaseImageData}
            initialParams={geometryEditorNode.params}
            onClose={() => setGeometryEditorNodeId(null)}
            onApply={(updatedParams) => {
              pushHistorySnapshot();
              const nextNodes = nodes.map((n) =>
                n.id === geometryEditorNode.id
                  ? { ...n, params: { ...n.params, ...updatedParams } }
                  : n
              );
              setNodes(nextNodes);
              setGeometryEditorNodeId(null);
              runGraph(nextNodes, edges);
            }}
          />
        )}

      {/* Multi-Seed Batch Samples Modal */}
      <BatchSamplesModal
        open={batchModalOpen}
        targetTitle={
          (selectedNodeId ? NODE_CATALOG_MAP[nodes.find((n) => n.id === selectedNodeId)?.type || '']?.title : null) ||
          'Final Output'
        }
        samples={batchResults}
        activeSeed={seed}
        isLoading={batchLoading}
        onClose={() => setBatchModalOpen(false)}
        onAdoptSeed={(chosenSeed) => handleSeedChange(chosenSeed)}
      />

      {/* Large / Resizable Result Preview Popup Modal */}
      <ResultPreviewModal
        open={resultModalOpen}
        nodeTitle={selectedNode ? NODE_CATALOG_MAP[selectedNode.type]?.title || selectedNode.type : 'Result'}
        nodeType={selectedNode?.type || ''}
        result={selectedNodeId ? results[selectedNodeId] ?? null : null}
        onClose={() => setResultModalOpen(false)}
        onExportImage={() => handleExportNodeImage()}
      />

      {/* AI Prompt Pipeline Generator Modal */}
      <AIGenerateModal
        open={aiModalOpen}
        currentInputDataUrl={(() => {
          const loadNode = nodes.find((n) => n.type === 'load_image');
          if (!loadNode) return null;
          const customUrl = typeof loadNode.params.customDataUrl === 'string' ? loadNode.params.customDataUrl : '';
          return customUrl || results[loadNode.id]?.previewDataUrl || null;
        })()}
        onClose={() => setAiModalOpen(false)}
        onApplyPipeline={(generated) => {
          pushHistorySnapshot();
          clearExecutionCache();
          stopWebcam();
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
