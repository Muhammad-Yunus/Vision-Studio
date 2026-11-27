import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Copy,
  Lock,
  Maximize2,
  Minus,
  MoreVertical,
  Paintbrush,
  Play,
  Plus,
  Trash2,
  Unlock,
  X,
} from 'lucide-react';
import { CATEGORY_META, NODE_CATALOG_MAP } from '../data/nodeCatalog';
import { NodeExecutionResult, PipelineEdge, PipelineNode } from '../types/pipeline';

interface PipelineCanvasProps {
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  selectedNodeId: string | null;
  results: Record<string, NodeExecutionResult>;
  onSelectNode: (id: string | null) => void;
  onMoveNode: (id: string, pos: { x: number; y: number }) => void;
  onAddNodeAt: (type: string, pos: { x: number; y: number }) => void;
  onConnect: (source: string, sourceHandle: string, target: string, targetHandle: string) => void;
  onDeleteEdge: (edgeId: string) => void;
  onDeleteNode: (nodeId: string) => void;
  onDuplicateNode: (nodeId: string) => void;
  onRunToNode: (nodeId: string) => void;
  onOpenOSDEditor: (nodeId: string) => void;
  fitTrigger: number;
}

const DEFAULT_NODE_W = 136;
const MULTI_PORT_NODE_W = 162;
const HEADER_H = 44;
const PREVIEW_H = 92;

function getNodeWidth(nodeType: string): number {
  const meta = NODE_CATALOG_MAP[nodeType];
  if (!meta) return DEFAULT_NODE_W;
  if (meta.inputs.length > 1 || meta.outputs.length > 1) return MULTI_PORT_NODE_W;
  return DEFAULT_NODE_W;
}

function getNodeHeight(nodeType: string): number {
  const meta = NODE_CATALOG_MAP[nodeType];
  const extraPorts = Math.max(meta?.inputs.length ?? 1, meta?.outputs.length ?? 1);
  if (extraPorts > 2) return HEADER_H + PREVIEW_H + 14 + (extraPorts - 2) * 16;
  return HEADER_H + PREVIEW_H + 10;
}

function getInputPortOffsetY(nodeType: string, index: number, total: number): number {
  if (total <= 1) return HEADER_H + PREVIEW_H * 0.45;
  const startY = HEADER_H + 22;
  const step = 24;
  return startY + index * step;
}

function getOutputPortOffsetY(nodeType: string, index: number, total: number): number {
  if (total <= 1) return HEADER_H + PREVIEW_H * 0.45;
  const startY = HEADER_H + 22;
  const step = 24;
  return startY + index * step;
}

function buildSmoothStepPath(
  x1: number,
  y1: number,
  x2: number,
  y2: number
): { path: string; midX: number; midY: number } {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const stub = 22;

  if (dx >= 28) {
    const midX = x1 + dx * 0.5;
    const r = Math.min(8, Math.abs(dy) / 2, Math.abs(dx) / 4);
    if (Math.abs(dy) < 2) {
      return { path: `M ${x1} ${y1} L ${x2} ${y2}`, midX, midY: (y1 + y2) / 2 };
    }
    const signY = dy > 0 ? 1 : -1;
    const p = [
      `M ${x1} ${y1}`,
      `L ${midX - r} ${y1}`,
      `Q ${midX} ${y1} ${midX} ${y1 + signY * r}`,
      `L ${midX} ${y2 - signY * r}`,
      `Q ${midX} ${y2} ${midX + r} ${y2}`,
      `L ${x2} ${y2}`,
    ].join(' ');
    return { path: p, midX, midY: (y1 + y2) / 2 };
  }

  // Wrap-around routing when target is to the left of source
  const outX = x1 + stub;
  const inX = x2 - stub;
  const detourY = dy >= 0 ? Math.max(y1, y2) + 56 : Math.min(y1, y2) - 56;
  const p = `M ${x1} ${y1} L ${outX} ${y1} L ${outX} ${detourY} L ${inX} ${detourY} L ${inX} ${y2} L ${x2} ${y2}`;
  return { path: p, midX: (outX + inX) / 2, midY: detourY };
}

export const PipelineCanvas: React.FC<PipelineCanvasProps> = ({
  nodes,
  edges,
  selectedNodeId,
  results,
  onSelectNode,
  onMoveNode,
  onAddNodeAt,
  onConnect,
  onDeleteEdge,
  onDeleteNode,
  onDuplicateNode,
  onRunToNode,
  onOpenOSDEditor,
  fitTrigger,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [viewport, setViewport] = useState({ x: 32, y: 20, zoom: 0.92 });
  const [locked, setLocked] = useState(false);
  const [menuNodeId, setMenuNodeId] = useState<string | null>(null);
  const [hoveredEdgeId, setHoveredEdgeId] = useState<string | null>(null);

  const [draggingNode, setDraggingNode] = useState<{
    id: string;
    startClientX: number;
    startClientY: number;
    origX: number;
    origY: number;
  } | null>(null);

  const [panning, setPanning] = useState<{
    startClientX: number;
    startClientY: number;
    origX: number;
    origY: number;
  } | null>(null);

  const [connecting, setConnecting] = useState<{
    sourceId: string;
    sourceHandle: string;
    startX: number;
    startY: number;
    currX: number;
    currY: number;
  } | null>(null);

  const fitView = useCallback(() => {
    const el = containerRef.current;
    if (!el || nodes.length === 0) return;
    const rect = el.getBoundingClientRect();
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    nodes.forEach((n) => {
      const w = getNodeWidth(n.type);
      const h = getNodeHeight(n.type);
      minX = Math.min(minX, n.position.x);
      minY = Math.min(minY, n.position.y);
      maxX = Math.max(maxX, n.position.x + w);
      maxY = Math.max(maxY, n.position.y + h);
    });
    const contentW = Math.max(200, maxX - minX + 70);
    const contentH = Math.max(200, maxY - minY + 70);
    const zoom = Math.min(1.1, Math.max(0.35, Math.min(rect.width / contentW, rect.height / contentH)));
    const x = (rect.width - (maxX - minX) * zoom) / 2 - minX * zoom;
    const y = (rect.height - (maxY - minY) * zoom) / 2 - minY * zoom;
    setViewport({ x, y, zoom });
  }, [nodes]);

  useEffect(() => {
    const t = setTimeout(() => fitView(), 40);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitTrigger]);

  useEffect(() => {
    const handlePointerMove = (e: PointerEvent) => {
      if (draggingNode && !locked) {
        const dx = (e.clientX - draggingNode.startClientX) / viewport.zoom;
        const dy = (e.clientY - draggingNode.startClientY) / viewport.zoom;
        onMoveNode(draggingNode.id, {
          x: Math.round(draggingNode.origX + dx),
          y: Math.round(draggingNode.origY + dy),
        });
      } else if (panning) {
        setViewport((prev) => ({
          ...prev,
          x: panning.origX + (e.clientX - panning.startClientX),
          y: panning.origY + (e.clientY - panning.startClientY),
        }));
      } else if (connecting && containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setConnecting((prev) =>
          prev
            ? {
                ...prev,
                currX: (e.clientX - rect.left - viewport.x) / viewport.zoom,
                currY: (e.clientY - rect.top - viewport.y) / viewport.zoom,
              }
            : null
        );
      }
    };

    const handlePointerUp = () => {
      setDraggingNode(null);
      setPanning(null);
      setConnecting(null);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [draggingNode, panning, connecting, viewport.zoom, viewport.x, viewport.y, locked, onMoveNode]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.09 : 0.92;
    setViewport((prev) => {
      const nextZoom = Math.min(2.0, Math.max(0.28, prev.zoom * factor));
      const worldX = (mouseX - prev.x) / prev.zoom;
      const worldY = (mouseY - prev.y) / prev.zoom;
      return {
        zoom: nextZoom,
        x: mouseX - worldX * nextZoom,
        y: mouseY - worldY * nextZoom,
      };
    });
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const nodeType = e.dataTransfer.getData('application/vision-node');
    if (!nodeType || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left - viewport.x) / viewport.zoom - 70);
    const y = Math.round((e.clientY - rect.top - viewport.y) / viewport.zoom - 40);
    onAddNodeAt(nodeType, { x, y });
  };

  const nodeMap = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);

  const computedEdges = useMemo(() => {
    return edges
      .map((edge) => {
        const src = nodeMap.get(edge.source);
        const tgt = nodeMap.get(edge.target);
        if (!src || !tgt) return null;
        const srcMeta = NODE_CATALOG_MAP[src.type];
        const tgtMeta = NODE_CATALOG_MAP[tgt.type];
        const srcIdx = Math.max(0, srcMeta?.outputs.findIndex((o) => o.id === edge.sourceHandle) ?? 0);
        const tgtIdx = Math.max(0, tgtMeta?.inputs.findIndex((i) => i.id === edge.targetHandle) ?? 0);

        const x1 = src.position.x + getNodeWidth(src.type);
        const y1 =
          src.position.y + getOutputPortOffsetY(src.type, srcIdx, srcMeta?.outputs.length ?? 1);
        const x2 = tgt.position.x;
        const y2 =
          tgt.position.y + getInputPortOffsetY(tgt.type, tgtIdx, tgtMeta?.inputs.length ?? 1);

        const { path, midX, midY } = buildSmoothStepPath(x1, y1, x2, y2);
        return { edge, path, midX, midY };
      })
      .filter(Boolean) as Array<{ edge: PipelineEdge; path: string; midX: number; midY: number }>;
  }, [edges, nodeMap]);

  // MiniMap bounding calculations
  const miniBounds = useMemo(() => {
    let minX = 0;
    let minY = 0;
    let maxX = 1250;
    let maxY = 660;
    nodes.forEach((n) => {
      minX = Math.min(minX, n.position.x - 40);
      minY = Math.min(minY, n.position.y - 40);
      maxX = Math.max(maxX, n.position.x + getNodeWidth(n.type) + 60);
      maxY = Math.max(maxY, n.position.y + getNodeHeight(n.type) + 60);
    });
    return { minX, minY, width: maxX - minX, height: maxY - minY };
  }, [nodes]);

  return (
    <div
      ref={containerRef}
      onWheel={handleWheel}
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
      }}
      onDrop={handleDrop}
      onPointerDown={(e) => {
        if (e.target === e.currentTarget || (e.target as HTMLElement).dataset.canvasBg === 'true') {
          setMenuNodeId(null);
          onSelectNode(null);
          setPanning({
            startClientX: e.clientX,
            startClientY: e.clientY,
            origX: viewport.x,
            origY: viewport.y,
          });
        }
      }}
      className="relative flex-1 bg-[#090B0E] overflow-hidden select-none cursor-grab active:cursor-grabbing"
      style={{
        backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.045) 1px, transparent 1px)',
        backgroundSize: `${24 * viewport.zoom}px ${24 * viewport.zoom}px`,
        backgroundPosition: `${viewport.x}px ${viewport.y}px`,
      }}
    >
      {/* Transformed World Layer */}
      <div
        data-canvas-bg="true"
        className="absolute inset-0 origin-top-left"
        style={{
          transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`,
        }}
      >
        {/* SVG Edge Layer */}
        <svg
          className="overflow-visible pointer-events-none absolute top-0 left-0 w-1 h-1"
          style={{ zIndex: 1 }}
        >
          {computedEdges.map(({ edge, path, midX, midY }) => {
            const isHighlighted =
              hoveredEdgeId === edge.id ||
              selectedNodeId === edge.source ||
              selectedNodeId === edge.target;
            return (
              <g key={edge.id} className="pointer-events-auto">
                {/* Invisible wider hit path */}
                <path
                  d={path}
                  fill="none"
                  stroke="transparent"
                  strokeWidth={14}
                  onMouseEnter={() => setHoveredEdgeId(edge.id)}
                  onMouseLeave={() => setHoveredEdgeId(null)}
                />
                <path
                  d={path}
                  fill="none"
                  stroke={isHighlighted ? '#9AE6C7' : '#73B399'}
                  strokeWidth={isHighlighted ? 2.1 : 1.55}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {/* Midpoint circular delete badge matching mockup */}
                <g
                  transform={`translate(${midX}, ${midY})`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteEdge(edge.id);
                  }}
                  onMouseEnter={() => setHoveredEdgeId(edge.id)}
                  onMouseLeave={() => setHoveredEdgeId(null)}
                  className="cursor-pointer"
                >
                  <circle
                    r={6.5}
                    fill="#181C23"
                    stroke={isHighlighted ? '#EF4444' : '#323946'}
                    strokeWidth={1}
                  />
                  <path
                    d="M -2.2 -2.2 L 2.2 2.2 M 2.2 -2.2 L -2.2 2.2"
                    stroke={isHighlighted ? '#FCA5A5' : '#9CA3AF'}
                    strokeWidth={1.2}
                    strokeLinecap="round"
                  />
                </g>
              </g>
            );
          })}

          {/* Active Wire Drag Preview */}
          {connecting && (
            <path
              d={buildSmoothStepPath(connecting.startX, connecting.startY, connecting.currX, connecting.currY).path}
              fill="none"
              stroke="#F58220"
              strokeWidth={2}
              strokeDasharray="5 4"
            />
          )}
        </svg>

        {/* Nodes Layer */}
        {nodes.map((node) => {
          const meta = NODE_CATALOG_MAP[node.type];
          if (!meta) return null;
          const catMeta = CATEGORY_META[meta.category];
          const isSelected = selectedNodeId === node.id;
          const res = results[node.id];
          const w = getNodeWidth(node.type);

          return (
            <div
              key={node.id}
              style={{
                left: node.position.x,
                top: node.position.y,
                width: w,
                zIndex: isSelected ? 20 : 10,
              }}
              onPointerDown={(e) => {
                e.stopPropagation();
                onSelectNode(node.id);
                setMenuNodeId(null);
                if (!locked) {
                  setDraggingNode({
                    id: node.id,
                    startClientX: e.clientX,
                    startClientY: e.clientY,
                    origX: node.position.x,
                    origY: node.position.y,
                  });
                }
              }}
              className={`absolute rounded-[7px] bg-[#15181E] border transition-shadow ${
                isSelected
                  ? 'border-[#389E82] ring-1 ring-[#389E82]/50 shadow-[0_8px_28px_rgba(0,0,0,0.65)]'
                  : 'border-[#282D37] hover:border-[#3A414E] shadow-[0_6px_20px_rgba(0,0,0,0.5)]'
              }`}
            >
              {/* Node Header */}
              <div className="flex items-start justify-between px-2.5 pt-2 pb-1.5 gap-1">
                <div className="min-w-0">
                  <div
                    className="text-[8.5px] font-semibold tracking-[0.06em] leading-none uppercase"
                    style={{ color: '#7D8590' }}
                  >
                    {catMeta.label}
                  </div>
                  <div className="text-[11.5px] font-semibold text-[#F3F4F6] truncate mt-0.5 leading-tight">
                    {meta.title}
                  </div>
                </div>

                <div className="flex items-center gap-0.5 shrink-0">
                  <button
                    type="button"
                    title="Run to this node"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectNode(node.id);
                      onRunToNode(node.id);
                    }}
                    className="p-0.5 text-[#8B949E] hover:text-[#F58220] transition-colors cursor-pointer"
                  >
                    <Play className="w-3 h-3 fill-current" />
                  </button>
                  <div className="relative">
                    <button
                      type="button"
                      title="Node actions"
                      onPointerDown={(e) => e.stopPropagation()}
                      onClick={(e) => {
                        e.stopPropagation();
                        setMenuNodeId(menuNodeId === node.id ? null : node.id);
                      }}
                      className="p-0.5 text-[#8B949E] hover:text-white transition-colors cursor-pointer"
                    >
                      <MoreVertical className="w-3 h-3" />
                    </button>

                    {menuNodeId === node.id && (
                      <div
                        onPointerDown={(e) => e.stopPropagation()}
                        className="absolute right-0 top-5 w-40 rounded-md bg-[#1B1F27] border border-[#303642] shadow-2xl py-1 z-50 text-[11px]"
                      >
                        {node.type === 'osd_canvas' && (
                          <button
                            type="button"
                            onClick={() => {
                              setMenuNodeId(null);
                              onOpenOSDEditor(node.id);
                            }}
                            className="w-full px-2.5 py-1.5 text-left text-[#F58220] hover:bg-[#252A35] flex items-center gap-2 cursor-pointer"
                          >
                            <Paintbrush className="w-3 h-3" />
                            <span>Open OSD Editor</span>
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => {
                            setMenuNodeId(null);
                            onDuplicateNode(node.id);
                          }}
                          className="w-full px-2.5 py-1.5 text-left text-[#E6EDF3] hover:bg-[#252A35] flex items-center gap-2 cursor-pointer"
                        >
                          <Copy className="w-3 h-3 text-[#9CA3AF]" />
                          <span>Duplicate</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setMenuNodeId(null);
                            onDeleteNode(node.id);
                          }}
                          className="w-full px-2.5 py-1.5 text-left text-[#F87171] hover:bg-[#252A35] flex items-center gap-2 cursor-pointer"
                        >
                          <Trash2 className="w-3 h-3" />
                          <span>Delete Node</span>
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Preview Image Box */}
              <div
                onDoubleClick={(e) => {
                  if (node.type === 'osd_canvas') {
                    e.stopPropagation();
                    onOpenOSDEditor(node.id);
                  }
                }}
                className={`mx-2 mb-2 rounded-[4px] overflow-hidden relative border border-[#222730] flex items-center justify-center ${
                  meta.transparentPreview ? 'bg-checkerboard' : 'bg-[#090B0E]'
                }`}
                style={{ height: PREVIEW_H }}
              >
                {res?.previewDataUrl ? (
                  <img
                    src={res.previewDataUrl}
                    alt={meta.title}
                    draggable={false}
                    className="w-full h-full object-contain pointer-events-none"
                  />
                ) : (
                  <span className="text-[10px] text-[#636B76] font-mono-tabular">Pending</span>
                )}

                {/* Timing + Cache Pill Overlay (top-right inside thumbnail) */}
                {res && (
                  <div className="absolute top-1 right-1 px-1.5 py-[1px] rounded-[3px] bg-[#0B2216]/85 border border-[#1E6B45]/75 text-[#4ADE80] text-[8.5px] font-mono-tabular leading-tight pointer-events-none">
                    {res.durationMs.toFixed(1)}ms{res.cached ? ' · cache' : ''}
                  </div>
                )}

                {/* Direct OSD Canvas Editor Quick Button on OSD Nodes */}
                {node.type === 'osd_canvas' && (
                  <button
                    type="button"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={(e) => {
                      e.stopPropagation();
                      onOpenOSDEditor(node.id);
                    }}
                    className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded bg-[#F58220] hover:bg-[#FF9233] text-[#0B0D10] text-[9px] font-semibold flex items-center gap-1 shadow cursor-pointer"
                  >
                    <Paintbrush className="w-2.5 h-2.5" />
                    <span>OSD</span>
                  </button>
                )}
              </div>

              {/* Extra spacing if >2 ports */}
              {Math.max(meta.inputs.length, meta.outputs.length) > 2 && (
                <div style={{ height: (Math.max(meta.inputs.length, meta.outputs.length) - 2) * 18 }} />
              )}

              {/* Input Handles */}
              {meta.inputs.map((port, idx) => {
                const topY = getInputPortOffsetY(node.type, idx, meta.inputs.length);
                const showLabel = meta.inputs.length > 1;
                return (
                  <div
                    key={port.id}
                    style={{ top: topY - 5 }}
                    className="absolute -left-[5px] flex items-center gap-1.5"
                  >
                    <div
                      title={`Input: ${port.label}`}
                      onPointerDown={(e) => e.stopPropagation()}
                      onPointerUp={(e) => {
                        e.stopPropagation();
                        if (connecting && connecting.sourceId !== node.id) {
                          onConnect(connecting.sourceId, connecting.sourceHandle, node.id, port.id);
                          setConnecting(null);
                        }
                      }}
                      className={`w-[9px] h-[9px] rounded-[2px] border border-[#0B0D10] transition-transform hover:scale-125 cursor-crosshair ${
                        connecting ? 'bg-[#F58220] ring-2 ring-[#F58220]/40' : 'bg-[#D1D5DB]'
                      }`}
                    />
                    {showLabel && (
                      <span className="px-1 py-[1px] rounded bg-[#0E1116]/90 text-[7.5px] font-mono-tabular text-[#B6BEC8] tracking-tight pointer-events-none whitespace-nowrap">
                        {port.label}
                      </span>
                    )}
                  </div>
                );
              })}

              {/* Output Handles */}
              {meta.outputs.map((port, idx) => {
                const topY = getOutputPortOffsetY(node.type, idx, meta.outputs.length);
                return (
                  <div
                    key={port.id}
                    style={{ top: topY - 5 }}
                    className="absolute -right-[5px] flex items-center"
                  >
                    <div
                      title={`Output: ${port.label} (Drag to connect)`}
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        if (locked) return;
                        setConnecting({
                          sourceId: node.id,
                          sourceHandle: port.id,
                          startX: node.position.x + w,
                          startY: node.position.y + topY,
                          currX: node.position.x + w + 24,
                          currY: node.position.y + topY,
                        });
                      }}
                      className="w-[9px] h-[9px] rounded-[2px] bg-[#D1D5DB] hover:bg-[#F58220] border border-[#0B0D10] transition-transform hover:scale-125 cursor-crosshair"
                    />
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>

      {/* Bottom-Left Floating Controls (+, -, Fit, Lock) */}
      <div className="absolute left-3.5 bottom-3.5 z-30 flex flex-col bg-[#16191E] border border-[#262A31] rounded-[6px] overflow-hidden shadow-lg">
        <button
          type="button"
          title="Zoom in"
          onClick={() =>
            setViewport((p) => ({ ...p, zoom: Math.min(2.0, +(p.zoom * 1.15).toFixed(2)) }))
          }
          className="w-7 h-7 flex items-center justify-center text-[#9CA3AF] hover:text-white hover:bg-[#222730] border-b border-[#232830] cursor-pointer"
        >
          <Plus className="w-3.5 h-3.5" />
        </button>
        <button
          type="button"
          title="Zoom out"
          onClick={() =>
            setViewport((p) => ({ ...p, zoom: Math.max(0.28, +(p.zoom * 0.87).toFixed(2)) }))
          }
          className="w-7 h-7 flex items-center justify-center text-[#9CA3AF] hover:text-white hover:bg-[#222730] border-b border-[#232830] cursor-pointer"
        >
          <Minus className="w-3.5 h-3.5" />
        </button>
        <button
          type="button"
          title="Fit view"
          onClick={fitView}
          className="w-7 h-7 flex items-center justify-center text-[#9CA3AF] hover:text-white hover:bg-[#222730] border-b border-[#232830] cursor-pointer"
        >
          <Maximize2 className="w-3.5 h-3.5" />
        </button>
        <button
          type="button"
          title={locked ? 'Unlock canvas' : 'Lock node positions'}
          onClick={() => setLocked((v) => !v)}
          className={`w-7 h-7 flex items-center justify-center hover:bg-[#222730] cursor-pointer ${
            locked ? 'text-[#F58220]' : 'text-[#9CA3AF] hover:text-white'
          }`}
        >
          {locked ? <Lock className="w-3.5 h-3.5" /> : <Unlock className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* Bottom-Right Live MiniMap + Subtle Attribution */}
      <div className="hidden sm:flex flex-col items-end gap-1.5 absolute right-3.5 bottom-3 z-30 pointer-events-none">
        <div className="w-[162px] h-[108px] rounded-[6px] bg-[#0F1115]/90 border border-[#262A31] p-1.5 shadow-xl overflow-hidden">
          <svg
            viewBox={`${miniBounds.minX} ${miniBounds.minY} ${miniBounds.width} ${miniBounds.height}`}
            className="w-full h-full"
          >
            {nodes.map((n) => (
              <rect
                key={n.id}
                x={n.position.x}
                y={n.position.y}
                width={getNodeWidth(n.type)}
                height={getNodeHeight(n.type)}
                rx={8}
                fill={selectedNodeId === n.id ? '#389E82' : '#2C333E'}
              />
            ))}
          </svg>
        </div>
        <span className="px-2 py-0.5 rounded bg-[#14171C]/90 border border-[#232830] text-[9.5px] text-[#7D8590]">
          Vision Studio Graph
        </span>
      </div>

      {/* Empty Canvas Hint */}
      {nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center space-y-1.5">
            <p className="text-sm font-medium text-[#C9D1D9]">Canvas is empty</p>
            <p className="text-xs text-[#7D8590]">
              Drag a node from the left library or pick a preset from Templates
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
