import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowDown,
  ArrowUp,
  ArrowUpRight,
  Circle,
  Copy,
  Crosshair,
  Eye,
  EyeOff,
  Grid,
  Minus,
  Paintbrush,
  Square,
  Tag,
  Trash2,
  Type,
  X,
} from 'lucide-react';
import { ensureOpenCVReady, isOpenCVWasmReady, renderOSDElementsToImageData } from '../engine/cvEngine';
import { OSDElement, OSDElementKind } from '../types/pipeline';

interface OSDEditorModalProps {
  open: boolean;
  nodeTitle: string;
  baseImageData: ImageData;
  initialElements: OSDElement[];
  initialOpacity: number;
  initialGridBurnIn: boolean;
  onClose: () => void;
  onApply: (elements: OSDElement[], globalOpacity: number, showGridBurnIn: boolean) => void;
}

const STENCILS: Array<{
  kind: OSDElementKind;
  label: string;
  cvCall: string;
  icon: React.ReactNode;
  defaultColor: string;
}> = [
  {
    kind: 'bbox_label',
    label: 'Detection BBox',
    cvCall: 'cv.rectangle + putText',
    icon: <Tag className="w-4 h-4" />,
    defaultColor: '#22C55E',
  },
  {
    kind: 'rect',
    label: 'Rectangle ROI',
    cvCall: 'cv.rectangle',
    icon: <Square className="w-4 h-4" />,
    defaultColor: '#F58220',
  },
  {
    kind: 'circle',
    label: 'Circle Marker',
    cvCall: 'cv.circle',
    icon: <Circle className="w-4 h-4" />,
    defaultColor: '#38BDF8',
  },
  {
    kind: 'line',
    label: 'Line Segment',
    cvCall: 'cv.line',
    icon: <Minus className="w-4 h-4" />,
    defaultColor: '#FACC15',
  },
  {
    kind: 'arrow',
    label: 'Vector Arrow',
    cvCall: 'cv.arrowedLine',
    icon: <ArrowUpRight className="w-4 h-4" />,
    defaultColor: '#38BDF8',
  },
  {
    kind: 'text',
    label: 'Text Label',
    cvCall: 'cv.putText',
    icon: <Type className="w-4 h-4" />,
    defaultColor: '#F8FAFC',
  },
  {
    kind: 'reticle',
    label: 'Crosshair Reticle',
    cvCall: 'cv.circle + cv.line',
    icon: <Crosshair className="w-4 h-4" />,
    defaultColor: '#EF4444',
  },
];

const COLOR_SWATCHES = ['#22C55E', '#F58220', '#38BDF8', '#EF4444', '#FACC15', '#A855F7', '#F8FAFC'];

function createElementFromKind(kind: OSDElementKind, cx = 0.5, cy = 0.5): OSDElement {
  const id = `osd-${kind}-${Date.now().toString(36)}-${Math.floor(Math.random() * 999)}`;
  const clamp = (v: number, min = 0.04, max = 0.85) => Math.min(max, Math.max(min, v));
  const w = kind === 'text' ? 0.36 : kind === 'reticle' || kind === 'circle' ? 0.18 : 0.28;
  const h = kind === 'text' ? 0.08 : kind === 'reticle' || kind === 'circle' ? 0.18 : 0.24;
  const x = kind === 'reticle' ? clamp(cx, 0.08, 0.92) : clamp(cx - w / 2, 0.02, 1 - w - 0.02);
  const y = kind === 'reticle' ? clamp(cy, 0.08, 0.92) : clamp(cy - h / 2, 0.04, 1 - h - 0.02);

  const base: OSDElement = {
    id,
    kind,
    name: STENCILS.find((s) => s.kind === kind)?.label || kind,
    visible: true,
    x,
    y,
    width: w,
    height: h,
    color: STENCILS.find((s) => s.kind === kind)?.defaultColor || '#22C55E',
    thickness: 2,
    filled: false,
    opacity: 0.95,
    fontScale: 0.52,
    lineType: 'LINE_AA',
  };

  if (kind === 'bbox_label') base.text = 'TARGET 0.95';
  if (kind === 'text') base.text = 'OSD_LABEL_01';
  if (kind === 'line' || kind === 'arrow') {
    base.x = clamp(cx - 0.12, 0.05, 0.85);
    base.y = clamp(cy + 0.08, 0.05, 0.92);
    base.x2 = clamp(cx + 0.14, 0.1, 0.95);
    base.y2 = clamp(cy - 0.1, 0.05, 0.92);
  }
  return base;
}

export const OSDEditorModal: React.FC<OSDEditorModalProps> = ({
  open,
  nodeTitle,
  baseImageData,
  initialElements,
  initialOpacity,
  initialGridBurnIn,
  onClose,
  onApply,
}) => {
  const [elements, setElements] = useState<OSDElement[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [globalOpacity, setGlobalOpacity] = useState(initialOpacity);
  const [showGridBurnIn, setShowGridBurnIn] = useState(initialGridBurnIn);
  const [snapToGrid, setSnapToGrid] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);

  const [dragState, setDragState] = useState<{
    mode: 'move' | 'resize-se' | 'endpoint-1' | 'endpoint-2';
    id: string;
    startX: number;
    startY: number;
    orig: OSDElement;
  } | null>(null);

  useEffect(() => {
    if (open) {
      const cloned = initialElements.map((e) => ({ ...e }));
      setElements(cloned);
      setSelectedId(cloned[0]?.id ?? null);
      setGlobalOpacity(initialOpacity);
      setShowGridBurnIn(initialGridBurnIn);
    }
  }, [open, initialElements, initialOpacity, initialGridBurnIn]);

  // Live render via OpenCV drawing engine onto the editor canvas
  useEffect(() => {
    if (!open || !canvasRef.current) return;
    let cancelled = false;
    const draw = () => {
      if (cancelled || !canvasRef.current) return;
      const rendered = renderOSDElementsToImageData(baseImageData, elements, globalOpacity, showGridBurnIn);
      const canvas = canvasRef.current;
      canvas.width = rendered.width;
      canvas.height = rendered.height;
      const ctx = canvas.getContext('2d');
      ctx?.putImageData(rendered, 0, 0);
    };
    if (isOpenCVWasmReady()) {
      draw();
    } else {
      ensureOpenCVReady().then(draw).catch(() => {});
    }
    return () => {
      cancelled = true;
    };
  }, [open, baseImageData, elements, globalOpacity, showGridBurnIn]);

  useEffect(() => {
    if (!dragState) return;
    const handleMove = (e: PointerEvent) => {
      const stage = stageRef.current;
      if (!stage) return;
      const rect = stage.getBoundingClientRect();
      const dx = (e.clientX - dragState.startX) / rect.width;
      const dy = (e.clientY - dragState.startY) / rect.height;
      const snap = (v: number) => (snapToGrid ? Math.round(v * 20) / 20 : v);

      setElements((prev) =>
        prev.map((el) => {
          if (el.id !== dragState.id) return el;
          const orig = dragState.orig;
          if (dragState.mode === 'move') {
            if (el.kind === 'line' || el.kind === 'arrow') {
              const nx = Math.min(0.96, Math.max(0.02, snap(orig.x + dx)));
              const ny = Math.min(0.96, Math.max(0.02, snap(orig.y + dy)));
              const ddx = nx - orig.x;
              const ddy = ny - orig.y;
              return {
                ...el,
                x: nx,
                y: ny,
                x2: Math.min(0.98, Math.max(0.02, (orig.x2 ?? orig.x + orig.width) + ddx)),
                y2: Math.min(0.98, Math.max(0.02, (orig.y2 ?? orig.y + orig.height) + ddy)),
              };
            }
            return {
              ...el,
              x: Math.min(0.96, Math.max(0.01, snap(orig.x + dx))),
              y: Math.min(0.96, Math.max(0.01, snap(orig.y + dy))),
            };
          }
          if (dragState.mode === 'resize-se') {
            return {
              ...el,
              width: Math.min(0.95, Math.max(0.05, snap(orig.width + dx))),
              height: Math.min(0.95, Math.max(0.04, snap(orig.height + dy))),
            };
          }
          if (dragState.mode === 'endpoint-1') {
            return {
              ...el,
              x: Math.min(0.98, Math.max(0.02, snap(orig.x + dx))),
              y: Math.min(0.98, Math.max(0.02, snap(orig.y + dy))),
            };
          }
          if (dragState.mode === 'endpoint-2') {
            return {
              ...el,
              x2: Math.min(0.98, Math.max(0.02, snap((orig.x2 ?? orig.x + orig.width) + dx))),
              y2: Math.min(0.98, Math.max(0.02, snap((orig.y2 ?? orig.y + orig.height) + dy))),
            };
          }
          return el;
        })
      );
    };

    const handleUp = () => setDragState(null);
    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
    return () => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
    };
  }, [dragState, snapToGrid]);

  const selectedEl = useMemo(
    () => elements.find((e) => e.id === selectedId) || null,
    [elements, selectedId]
  );

  if (!open) return null;

  const updateSelected = (patch: Partial<OSDElement>) => {
    if (!selectedId) return;
    setElements((prev) => prev.map((el) => (el.id === selectedId ? { ...el, ...patch } : el)));
  };

  const addElement = (kind: OSDElementKind, x = 0.5, y = 0.5) => {
    const item = createElementFromKind(kind, x, y);
    setElements((prev) => [...prev, item]);
    setSelectedId(item.id);
  };

  const moveLayer = (id: string, dir: -1 | 1) => {
    setElements((prev) => {
      const idx = prev.findIndex((e) => e.id === id);
      if (idx < 0) return prev;
      const target = idx + dir;
      if (target < 0 || target >= prev.length) return prev;
      const copy = [...prev];
      const [item] = copy.splice(idx, 1);
      copy.splice(target, 0, item);
      return copy;
    });
  };

  const duplicateSelected = (id: string) => {
    const src = elements.find((e) => e.id === id);
    if (!src) return;
    const copy: OSDElement = {
      ...src,
      id: `osd-${src.kind}-${Date.now().toString(36)}`,
      name: `${src.name} copy`,
      x: Math.min(0.85, src.x + 0.04),
      y: Math.min(0.85, src.y + 0.04),
      x2: src.x2 !== undefined ? Math.min(0.95, src.x2 + 0.04) : undefined,
      y2: src.y2 !== undefined ? Math.min(0.95, src.y2 + 0.04) : undefined,
    };
    setElements((prev) => [...prev, copy]);
    setSelectedId(copy.id);
  };

  const deleteElement = (id: string) => {
    setElements((prev) => {
      const next = prev.filter((e) => e.id !== id);
      if (selectedId === id) setSelectedId(next[0]?.id ?? null);
      return next;
    });
  };

  const aspect = baseImageData.width / Math.max(1, baseImageData.height);

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-xs flex items-center justify-center p-2 sm:p-5">
      <div className="w-full max-w-[1220px] h-[92vh] rounded-xl bg-[#121417] border border-[#2A2F38] shadow-2xl flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-[#15181D] border-b border-[#22262C] shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-7 h-7 rounded-[6px] bg-[#251A12] border border-[#F58220]/40 flex items-center justify-center text-[#F58220]">
              <Paintbrush className="w-4 h-4" />
            </div>
            <div className="min-w-0">
              <h2 className="text-[14.5px] font-semibold text-white truncate">
                OpenCV OSD Drag &amp; Drop Canvas Editor — {nodeTitle}
              </h2>
              <p className="text-[11px] text-[#8B949E] truncate">
                Drag primitives onto the frame, reposition/resize handles, and render via OpenCV drawing
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="w-8 h-8 rounded-[6px] bg-[#1C2027] hover:bg-[#272D37] text-[#9CA3AF] hover:text-white flex items-center justify-center cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* 3-Column Editor Body */}
        <div className="flex-1 flex flex-col md:flex-row min-h-0 overflow-hidden">
          {/* Left Stencil Palette */}
          <div className="w-full md:w-[220px] bg-[#121417] border-b md:border-b-0 md:border-r border-[#22262C] p-3 overflow-y-auto shrink-0">
            <div className="text-[10.5px] font-semibold uppercase tracking-wider text-[#7D8590] mb-2">
              Drag or Click Primitives
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-1 gap-1.5">
              {STENCILS.map((s) => (
                <div
                  key={s.kind}
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData('application/osd-stencil', s.kind);
                    e.dataTransfer.effectAllowed = 'copy';
                  }}
                  onClick={() => addElement(s.kind)}
                  className="flex items-center gap-2.5 px-2.5 py-2 rounded-[6px] bg-[#171A20] hover:bg-[#1E232B] border border-[#252A33] hover:border-[#3A424E] cursor-grab active:cursor-grabbing transition-colors"
                >
                  <span style={{ color: s.defaultColor }}>{s.icon}</span>
                  <div className="min-w-0">
                    <div className="text-[12px] font-medium text-[#F3F4F6] truncate">{s.label}</div>
                    <div className="text-[10px] font-mono-tabular text-[#7D8590] truncate">
                      {s.cvCall}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Center Interactive Canvas Stage */}
          <div className="flex-1 bg-[#090B0E] flex flex-col min-w-0 min-h-0">
            {/* Stage Controls */}
            <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2 bg-[#0F1115] border-b border-[#1E222A] text-[11.5px]">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setSnapToGrid((v) => !v)}
                  className={`flex items-center gap-1.5 px-2.5 py-1 rounded border cursor-pointer transition-colors ${
                    snapToGrid
                      ? 'bg-[#1D2B24] border-[#2E7D67] text-[#86EFAC]'
                      : 'bg-[#16191E] border-[#262B33] text-[#9CA3AF] hover:text-white'
                  }`}
                >
                  <Grid className="w-3.5 h-3.5" />
                  <span>Snap 5%</span>
                </button>
                <label className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#16191E] border border-[#262B33] text-[#C9D1D9] cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showGridBurnIn}
                    onChange={(e) => setShowGridBurnIn(e.target.checked)}
                    className="accent-[#F58220]"
                  />
                  <span>Corner Safe Frame</span>
                </label>
              </div>

              <div className="font-mono-tabular text-[11px] text-[#8B949E]">
                Frame {baseImageData.width}×{baseImageData.height}px · {elements.length} OSD layers
              </div>
            </div>

            {/* Interactive Drop & Drag Surface */}
            <div className="flex-1 flex items-center justify-center p-4 overflow-auto">
              <div
                ref={stageRef}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.dataTransfer.dropEffect = 'copy';
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  const kind = e.dataTransfer.getData('application/osd-stencil') as OSDElementKind;
                  if (!kind || !stageRef.current) return;
                  const rect = stageRef.current.getBoundingClientRect();
                  const nx = (e.clientX - rect.left) / rect.width;
                  const ny = (e.clientY - rect.top) / rect.height;
                  addElement(kind, nx, ny);
                }}
                style={{
                  aspectRatio: `${aspect}`,
                  width: 'min(100%, 540px)',
                }}
                className="relative rounded-md overflow-hidden border border-[#2D333E] shadow-2xl bg-black select-none"
              >
                <canvas
                  ref={canvasRef}
                  className="w-full h-full block pointer-events-none"
                />

                {/* Interactive Hitboxes & Handles */}
                {elements.map((el) => {
                  if (!el.visible) return null;
                  const isSel = selectedId === el.id;

                  if (el.kind === 'line' || el.kind === 'arrow') {
                    const x2 = el.x2 ?? el.x + el.width;
                    const y2 = el.y2 ?? el.y + el.height;
                    const minX = Math.min(el.x, x2);
                    const minY = Math.min(el.y, y2);
                    const maxX = Math.max(el.x, x2);
                    const maxY = Math.max(el.y, y2);
                    return (
                      <React.Fragment key={el.id}>
                        <div
                          onPointerDown={(e) => {
                            e.stopPropagation();
                            setSelectedId(el.id);
                            setDragState({
                              mode: 'move',
                              id: el.id,
                              startX: e.clientX,
                              startY: e.clientY,
                              orig: { ...el },
                            });
                          }}
                          style={{
                            left: `${Math.max(0, minX - 0.02) * 100}%`,
                            top: `${Math.max(0, minY - 0.02) * 100}%`,
                            width: `${Math.max(0.06, maxX - minX + 0.04) * 100}%`,
                            height: `${Math.max(0.06, maxY - minY + 0.04) * 100}%`,
                          }}
                          className={`absolute cursor-move ${
                            isSel ? 'outline-1 outline-dashed outline-[#F58220]/70' : ''
                          }`}
                        />
                        {isSel && (
                          <>
                            <div
                              onPointerDown={(e) => {
                                e.stopPropagation();
                                setDragState({
                                  mode: 'endpoint-1',
                                  id: el.id,
                                  startX: e.clientX,
                                  startY: e.clientY,
                                  orig: { ...el },
                                });
                              }}
                              style={{ left: `${el.x * 100}%`, top: `${el.y * 100}%` }}
                              className="absolute -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-[#F58220] border border-black cursor-crosshair shadow"
                            />
                            <div
                              onPointerDown={(e) => {
                                e.stopPropagation();
                                setDragState({
                                  mode: 'endpoint-2',
                                  id: el.id,
                                  startX: e.clientX,
                                  startY: e.clientY,
                                  orig: { ...el },
                                });
                              }}
                              style={{ left: `${x2 * 100}%`, top: `${y2 * 100}%` }}
                              className="absolute -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-[#38BDF8] border border-black cursor-crosshair shadow"
                            />
                          </>
                        )}
                      </React.Fragment>
                    );
                  }

                  const boxLeft = el.kind === 'reticle' ? el.x - el.width / 2 : el.x;
                  const boxTop = el.kind === 'reticle' ? el.y - el.height / 2 : el.y;

                  return (
                    <div
                      key={el.id}
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        setSelectedId(el.id);
                        setDragState({
                          mode: 'move',
                          id: el.id,
                          startX: e.clientX,
                          startY: e.clientY,
                          orig: { ...el },
                        });
                      }}
                      style={{
                        left: `${boxLeft * 100}%`,
                        top: `${boxTop * 100}%`,
                        width: `${el.width * 100}%`,
                        height: `${el.height * 100}%`,
                      }}
                      className={`absolute cursor-move ${
                        isSel
                          ? 'outline-1 outline-dashed outline-[#F58220] bg-[#F58220]/5'
                          : 'hover:outline-1 hover:outline-dashed hover:outline-white/40'
                      }`}
                    >
                      {isSel && (
                        <div
                          onPointerDown={(e) => {
                            e.stopPropagation();
                            setDragState({
                              mode: 'resize-se',
                              id: el.id,
                              startX: e.clientX,
                              startY: e.clientY,
                              orig: { ...el },
                            });
                          }}
                          className="absolute -right-1.5 -bottom-1.5 w-3 h-3 rounded-[2px] bg-[#F58220] border border-black cursor-nwse-resize"
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Right Layers & OpenCV Element Inspector */}
          <div className="w-full md:w-[280px] bg-[#121417] border-t md:border-t-0 md:border-l border-[#22262C] flex flex-col min-h-0 shrink-0">
            {/* Layer List */}
            <div className="p-3 border-b border-[#22262C] max-h-48 overflow-y-auto space-y-1">
              <div className="text-[10.5px] font-semibold uppercase tracking-wider text-[#7D8590] mb-1.5">
                OSD Layers ({elements.length})
              </div>
              {elements.length === 0 ? (
                <p className="text-[11px] text-[#6E7681]">Drop any primitive onto the image.</p>
              ) : (
                elements.map((el) => (
                  <div
                    key={el.id}
                    onClick={() => setSelectedId(el.id)}
                    className={`flex items-center justify-between gap-1.5 px-2 py-1.5 rounded-[5px] text-[11.5px] cursor-pointer ${
                      selectedId === el.id
                        ? 'bg-[#1F242D] border border-[#389E82]/60 text-white'
                        : 'bg-[#16191E] border border-transparent text-[#C9D1D9] hover:bg-[#1B1F26]'
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span
                        className="w-2.5 h-2.5 rounded-xs shrink-0"
                        style={{ backgroundColor: el.color }}
                      />
                      <span className="truncate">{el.name}</span>
                    </div>
                    <div className="flex items-center gap-0.5 shrink-0">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setElements((p) =>
                            p.map((x) => (x.id === el.id ? { ...x, visible: !x.visible } : x))
                          );
                        }}
                        className="p-1 text-[#8B949E] hover:text-white cursor-pointer"
                      >
                        {el.visible ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          moveLayer(el.id, -1);
                        }}
                        className="p-0.5 text-[#8B949E] hover:text-white cursor-pointer"
                      >
                        <ArrowUp className="w-3 h-3" />
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          moveLayer(el.id, 1);
                        }}
                        className="p-0.5 text-[#8B949E] hover:text-white cursor-pointer"
                      >
                        <ArrowDown className="w-3 h-3" />
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteElement(el.id);
                        }}
                        className="p-1 text-[#8B949E] hover:text-[#F87171] cursor-pointer"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* Selected Primitive Properties */}
            <div className="flex-1 overflow-y-auto p-3 space-y-3">
              {selectedEl ? (
                <>
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-[#7D8590]">
                      OpenCV Drawing Props
                    </span>
                    <button
                      type="button"
                      onClick={() => duplicateSelected(selectedEl.id)}
                      className="text-[11px] text-[#8B949E] hover:text-white flex items-center gap-1 cursor-pointer"
                    >
                      <Copy className="w-3 h-3" />
                      <span>Duplicate</span>
                    </button>
                  </div>

                  <label className="block space-y-1">
                    <span className="text-[11px] text-[#9CA3AF]">Layer Name</span>
                    <input
                      type="text"
                      value={selectedEl.name}
                      onChange={(e) => updateSelected({ name: e.target.value })}
                      className="w-full rounded bg-[#0F1115] border border-[#262B34] px-2.5 py-1.5 text-[12px] text-white outline-none focus:border-[#389E82]"
                    />
                  </label>

                  {(selectedEl.kind === 'text' || selectedEl.kind === 'bbox_label') && (
                    <label className="block space-y-1">
                      <span className="text-[11px] text-[#9CA3AF]">OSD Text (cv.putText)</span>
                      <input
                        type="text"
                        value={selectedEl.text || ''}
                        onChange={(e) => updateSelected({ text: e.target.value })}
                        className="w-full rounded bg-[#0F1115] border border-[#262B34] px-2.5 py-1.5 text-[12px] font-mono-tabular text-white outline-none focus:border-[#389E82]"
                      />
                    </label>
                  )}

                  <div className="space-y-1.5">
                    <span className="text-[11px] text-[#9CA3AF] block">Stroke / Scalar Color</span>
                    <div className="flex items-center flex-wrap gap-1.5">
                      {COLOR_SWATCHES.map((swatch) => (
                        <button
                          key={swatch}
                          type="button"
                          onClick={() => updateSelected({ color: swatch })}
                          style={{ backgroundColor: swatch }}
                          className={`w-5 h-5 rounded-xs border cursor-pointer ${
                            selectedEl.color.toUpperCase() === swatch
                              ? 'border-white scale-110'
                              : 'border-black/40'
                          }`}
                        />
                      ))}
                      <input
                        type="color"
                        value={selectedEl.color}
                        onChange={(e) => updateSelected({ color: e.target.value })}
                        className="w-6 h-5 bg-transparent border border-[#2C323C] rounded cursor-pointer"
                      />
                    </div>
                  </div>

                  <label className="block space-y-1">
                    <div className="flex justify-between text-[11px]">
                      <span className="text-[#9CA3AF]">Thickness (px)</span>
                      <span className="font-mono-tabular text-white">{selectedEl.thickness}</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={8}
                      step={0.5}
                      value={selectedEl.thickness}
                      onChange={(e) => updateSelected({ thickness: Number(e.target.value) })}
                      className="w-full studio-range"
                    />
                  </label>

                  {(selectedEl.kind === 'text' || selectedEl.kind === 'bbox_label') && (
                    <label className="block space-y-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-[#9CA3AF]">Font Scale</span>
                        <span className="font-mono-tabular text-white">
                          {(selectedEl.fontScale ?? 0.52).toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.3}
                        max={1.2}
                        step={0.05}
                        value={selectedEl.fontScale ?? 0.52}
                        onChange={(e) => updateSelected({ fontScale: Number(e.target.value) })}
                        className="w-full studio-range"
                      />
                    </label>
                  )}

                  {(selectedEl.kind === 'rect' || selectedEl.kind === 'circle') && (
                    <label className="flex items-center justify-between text-[11.5px] text-[#C9D1D9] cursor-pointer">
                      <span>Fill Shape (thickness = -1)</span>
                      <input
                        type="checkbox"
                        checked={Boolean(selectedEl.filled)}
                        onChange={(e) => updateSelected({ filled: e.target.checked })}
                        className="accent-[#F58220]"
                      />
                    </label>
                  )}

                  <div className="rounded bg-[#0B0D10] border border-[#22262C] p-2 text-[10.5px] font-mono-tabular text-[#8B949E] space-y-0.5">
                    <div>
                      x: {Math.round(selectedEl.x * baseImageData.width)}px · y:{' '}
                      {Math.round(selectedEl.y * baseImageData.height)}px
                    </div>
                    <div>
                      w: {Math.round(selectedEl.width * baseImageData.width)}px · h:{' '}
                      {Math.round(selectedEl.height * baseImageData.height)}px
                    </div>
                  </div>
                </>
              ) : (
                <p className="text-[11.5px] text-[#7D8590]">
                  Click an OSD element on the canvas or layer stack to edit its OpenCV drawing properties.
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Footer Action Bar */}
        <div className="flex items-center justify-between px-4 py-3 bg-[#15181D] border-t border-[#22262C] shrink-0">
          <div className="flex items-center gap-3">
            <span className="text-[11.5px] text-[#8B949E] hidden sm:inline">
              Global Overlay Opacity
            </span>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={globalOpacity}
              onChange={(e) => setGlobalOpacity(Number(e.target.value))}
              className="w-24 studio-range"
            />
            <span className="text-[11px] font-mono-tabular text-white">
              {Math.round(globalOpacity * 100)}%
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3.5 py-1.5 rounded-[6px] bg-[#1A1E25] hover:bg-[#242933] border border-[#2D333E] text-[12.5px] font-medium text-[#E6EDF3] cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => onApply(elements, globalOpacity, showGridBurnIn)}
              className="px-4 py-1.5 rounded-[6px] bg-[#F58220] hover:bg-[#FF9233] text-[#0B0D10] text-[12.5px] font-semibold cursor-pointer"
            >
              Apply OSD &amp; Run Pipeline
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
