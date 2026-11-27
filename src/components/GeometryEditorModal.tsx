import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Check, Crop, Move, RotateCcw, X } from 'lucide-react';
import { previewCropWithOpenCV, previewPerspectiveWithOpenCV } from '../engine/cvEngine';

interface GeometryEditorModalProps {
  open: boolean;
  mode: 'warp_perspective' | 'crop_roi';
  nodeTitle: string;
  baseImageData: ImageData;
  initialParams: Record<string, unknown>;
  onClose: () => void;
  onApply: (updatedParams: Record<string, unknown>) => void;
}

type CornerKey = 'tl' | 'tr' | 'br' | 'bl';

function imageDataToUrl(img: ImageData): string {
  const c = document.createElement('canvas');
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext('2d')!;
  ctx.putImageData(img, 0, 0);
  return c.toDataURL('image/png');
}

export const GeometryEditorModal: React.FC<GeometryEditorModalProps> = ({
  open,
  mode,
  nodeTitle,
  baseImageData,
  initialParams,
  onClose,
  onApply,
}) => {
  const stageRef = useRef<HTMLDivElement>(null);

  const [perspectivePts, setPerspectivePts] = useState({
    tlX: Number(initialParams.tlX ?? 12),
    tlY: Number(initialParams.tlY ?? 8),
    trX: Number(initialParams.trX ?? 88),
    trY: Number(initialParams.trY ?? 12),
    brX: Number(initialParams.brX ?? 98),
    brY: Number(initialParams.brY ?? 94),
    blX: Number(initialParams.blX ?? 2),
    blY: Number(initialParams.blY ?? 92),
  });

  const [roiBox, setRoiBox] = useState({
    x: Number(initialParams.x ?? 15),
    y: Number(initialParams.y ?? 15),
    w: Number(initialParams.w ?? 70),
    h: Number(initialParams.h ?? 70),
  });

  const [dragCorner, setDragCorner] = useState<CornerKey | null>(null);
  const [dragRoi, setDragRoi] = useState<{
    kind: 'move' | 'se' | 'nw';
    startX: number;
    startY: number;
    orig: typeof roiBox;
  } | null>(null);

  useEffect(() => {
    if (!open) return;
    setPerspectivePts({
      tlX: Number(initialParams.tlX ?? 12),
      tlY: Number(initialParams.tlY ?? 8),
      trX: Number(initialParams.trX ?? 88),
      trY: Number(initialParams.trY ?? 12),
      brX: Number(initialParams.brX ?? 98),
      brY: Number(initialParams.brY ?? 94),
      blX: Number(initialParams.blX ?? 2),
      blY: Number(initialParams.blY ?? 92),
    });
    setRoiBox({
      x: Number(initialParams.x ?? 15),
      y: Number(initialParams.y ?? 15),
      w: Number(initialParams.w ?? 70),
      h: Number(initialParams.h ?? 70),
    });
  }, [open, initialParams]);

  const baseUrl = useMemo(() => imageDataToUrl(baseImageData), [baseImageData]);

  const livePreviewUrl = useMemo(() => {
    if (!open) return '';
    try {
      if (mode === 'warp_perspective') {
        const out = previewPerspectiveWithOpenCV(baseImageData, {
          ...initialParams,
          ...perspectivePts,
        });
        return imageDataToUrl(out);
      } else {
        const out = previewCropWithOpenCV(baseImageData, roiBox);
        return imageDataToUrl(out);
      }
    } catch {
      return baseUrl;
    }
  }, [open, mode, baseImageData, initialParams, perspectivePts, roiBox, baseUrl]);

  useEffect(() => {
    if (!open) return;
    const onMove = (e: PointerEvent) => {
      const stage = stageRef.current;
      if (!stage) return;
      const rect = stage.getBoundingClientRect();
      const pctX = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
      const pctY = Math.max(0, Math.min(100, ((e.clientY - rect.top) / rect.height) * 100));

      if (dragCorner) {
        setPerspectivePts((prev) => {
          const rx = Math.round(pctX);
          const ry = Math.round(pctY);
          if (dragCorner === 'tl') return { ...prev, tlX: Math.min(48, rx), tlY: Math.min(48, ry) };
          if (dragCorner === 'tr') return { ...prev, trX: Math.max(52, rx), trY: Math.min(48, ry) };
          if (dragCorner === 'br') return { ...prev, brX: Math.max(52, rx), brY: Math.max(52, ry) };
          return { ...prev, blX: Math.min(48, rx), blY: Math.max(52, ry) };
        });
      } else if (dragRoi) {
        const dx = ((e.clientX - dragRoi.startX) / rect.width) * 100;
        const dy = ((e.clientY - dragRoi.startY) / rect.height) * 100;
        if (dragRoi.kind === 'move') {
          const nx = Math.max(0, Math.min(100 - dragRoi.orig.w, Math.round(dragRoi.orig.x + dx)));
          const ny = Math.max(0, Math.min(100 - dragRoi.orig.h, Math.round(dragRoi.orig.y + dy)));
          setRoiBox((prev) => ({ ...prev, x: nx, y: ny }));
        } else if (dragRoi.kind === 'se') {
          const nw = Math.max(12, Math.min(100 - dragRoi.orig.x, Math.round(dragRoi.orig.w + dx)));
          const nh = Math.max(12, Math.min(100 - dragRoi.orig.y, Math.round(dragRoi.orig.h + dy)));
          setRoiBox((prev) => ({ ...prev, w: nw, h: nh }));
        } else {
          const nx = Math.max(0, Math.min(dragRoi.orig.x + dragRoi.orig.w - 12, Math.round(dragRoi.orig.x + dx)));
          const ny = Math.max(0, Math.min(dragRoi.orig.y + dragRoi.orig.h - 12, Math.round(dragRoi.orig.y + dy)));
          const nw = dragRoi.orig.x + dragRoi.orig.w - nx;
          const nh = dragRoi.orig.y + dragRoi.orig.h - ny;
          setRoiBox({ x: nx, y: ny, w: nw, h: nh });
        }
      }
    };
    const onUp = () => {
      setDragCorner(null);
      setDragRoi(null);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [open, dragCorner, dragRoi]);

  if (!open) return null;

  const cropPxW = Math.max(24, Math.round((roiBox.w / 100) * baseImageData.width));
  const cropPxH = Math.max(24, Math.round((roiBox.h / 100) * baseImageData.height));

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-[2px] flex items-center justify-center p-3 sm:p-5">
      <div className="w-full max-w-4xl bg-[#121417] border border-[#272C35] rounded-xl shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#22262C] bg-[#15181D]">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-md bg-[#2DD4BF]/15 border border-[#2DD4BF]/40 flex items-center justify-center text-[#2DD4BF]">
              {mode === 'warp_perspective' ? <Move className="w-4 h-4" /> : <Crop className="w-4 h-4" />}
            </div>
            <div>
              <h2 className="text-[14px] font-semibold text-white">
                {mode === 'warp_perspective'
                  ? `Interactive Homography Corner Picker — ${nodeTitle}`
                  : `Interactive Visual ROI Cropper — ${nodeTitle}`}
              </h2>
              <p className="text-[11px] text-[#7D8590]">
                {mode === 'warp_perspective'
                  ? 'Drag the 4 destination handles (TL, TR, BR, BL) to warp via cv.warpPerspective.'
                  : 'Drag or resize the ROI box to slice the normalized image region.'}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="w-7 h-7 rounded-md bg-[#1B1F27] hover:bg-[#252A35] text-[#9CA3AF] hover:text-white flex items-center justify-center cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Workspace */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 bg-[#0B0D10]">
          {/* Left: Interactive Source Stage */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-[11px] text-[#8B949E]">
              <span className="font-semibold uppercase tracking-wider">Interactive Source Overlay</span>
              <span className="font-mono-tabular">
                {baseImageData.width}×{baseImageData.height}px
              </span>
            </div>
            <div className="rounded-lg border border-[#262B34] bg-[#080A0D] p-3 flex items-center justify-center">
              <div
                ref={stageRef}
                className="relative select-none touch-none inline-block max-w-full"
                style={{
                  aspectRatio: `${baseImageData.width} / ${baseImageData.height}`,
                  width: '100%',
                  maxWidth: 380,
                }}
              >
                <img
                  src={baseUrl}
                  alt="Base input"
                  draggable={false}
                  className="w-full h-full object-fill rounded pointer-events-none block"
                />

                {mode === 'warp_perspective' ? (
                  <>
                    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 w-full h-full pointer-events-none">
                      <polygon
                        points={`${perspectivePts.tlX},${perspectivePts.tlY} ${perspectivePts.trX},${perspectivePts.trY} ${perspectivePts.brX},${perspectivePts.brY} ${perspectivePts.blX},${perspectivePts.blY}`}
                        fill="rgba(45, 212, 191, 0.14)"
                        stroke="#2DD4BF"
                        strokeWidth="0.9"
                        strokeDasharray="2 1.5"
                      />
                    </svg>
                    {(
                      [
                        { key: 'tl' as const, label: 'TL', x: perspectivePts.tlX, y: perspectivePts.tlY },
                        { key: 'tr' as const, label: 'TR', x: perspectivePts.trX, y: perspectivePts.trY },
                        { key: 'br' as const, label: 'BR', x: perspectivePts.brX, y: perspectivePts.brY },
                        { key: 'bl' as const, label: 'BL', x: perspectivePts.blX, y: perspectivePts.blY },
                      ]
                    ).map((pt) => (
                      <div
                        key={pt.key}
                        onPointerDown={(e) => {
                          e.stopPropagation();
                          setDragCorner(pt.key);
                        }}
                        style={{ left: `${pt.x}%`, top: `${pt.y}%` }}
                        className="absolute -translate-x-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-[#121417] border-2 border-[#F58220] text-[#F58220] text-[8.5px] font-bold font-mono-tabular flex items-center justify-center shadow-lg cursor-grab active:cursor-grabbing hover:scale-110 transition-transform"
                      >
                        {pt.label}
                      </div>
                    ))}
                  </>
                ) : (
                  <div
                    onPointerDown={(e) => {
                      e.stopPropagation();
                      setDragRoi({
                        kind: 'move',
                        startX: e.clientX,
                        startY: e.clientY,
                        orig: { ...roiBox },
                      });
                    }}
                    style={{
                      left: `${roiBox.x}%`,
                      top: `${roiBox.y}%`,
                      width: `${roiBox.w}%`,
                      height: `${roiBox.h}%`,
                      boxShadow: '0 0 0 9999px rgba(0,0,0,0.52)',
                    }}
                    className="absolute border-2 border-[#F58220] cursor-move"
                  >
                    <div
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        setDragRoi({
                          kind: 'nw',
                          startX: e.clientX,
                          startY: e.clientY,
                          orig: { ...roiBox },
                        });
                      }}
                      className="absolute -left-1.5 -top-1.5 w-3.5 h-3.5 rounded-sm bg-[#F58220] border border-[#0B0D10] cursor-nwse-resize"
                    />
                    <div
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        setDragRoi({
                          kind: 'se',
                          startX: e.clientX,
                          startY: e.clientY,
                          orig: { ...roiBox },
                        });
                      }}
                      className="absolute -right-1.5 -bottom-1.5 w-3.5 h-3.5 rounded-sm bg-[#F58220] border border-[#0B0D10] cursor-nwse-resize"
                    />
                    <span className="absolute left-1 top-1 px-1 py-0.5 rounded bg-[#0B0D10]/85 text-[9px] font-mono-tabular text-[#F58220]">
                      {cropPxW}×{cropPxH}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right: Live OpenCV Output Preview + Numeric Coordinates */}
          <div className="space-y-2.5 flex flex-col">
            <div className="flex items-center justify-between text-[11px] text-[#8B949E]">
              <span className="font-semibold uppercase tracking-wider">Live OpenCV WASM Output</span>
              <span className="font-mono-tabular text-[#4ADE80]">cv.Mat</span>
            </div>
            <div className="flex-1 rounded-lg border border-[#262B34] bg-[#080A0D] p-3 flex items-center justify-center min-h-[240px]">
              <img
                src={livePreviewUrl}
                alt="Live OpenCV Geometry Preview"
                className="max-w-full max-h-[290px] object-contain rounded"
              />
            </div>

            {mode === 'warp_perspective' ? (
              <div className="grid grid-cols-2 gap-1.5 text-[11px] font-mono-tabular bg-[#15181D] border border-[#242933] rounded-md p-2.5 text-[#C9D1D9]">
                <span>TL: ({perspectivePts.tlX}%, {perspectivePts.tlY}%)</span>
                <span>TR: ({perspectivePts.trX}%, {perspectivePts.trY}%)</span>
                <span>BL: ({perspectivePts.blX}%, {perspectivePts.blY}%)</span>
                <span>BR: ({perspectivePts.brX}%, {perspectivePts.brY}%)</span>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-1.5 text-[11px] font-mono-tabular bg-[#15181D] border border-[#242933] rounded-md p-2.5 text-[#C9D1D9]">
                <span>X: {roiBox.x}% · Y: {roiBox.y}%</span>
                <span>W: {roiBox.w}% · H: {roiBox.h}%</span>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-[#22262C] bg-[#15181D]">
          <button
            type="button"
            onClick={() => {
              if (mode === 'warp_perspective') {
                setPerspectivePts({
                  tlX: 12,
                  tlY: 8,
                  trX: 88,
                  trY: 12,
                  brX: 98,
                  brY: 94,
                  blX: 2,
                  blY: 92,
                });
              } else {
                setRoiBox({ x: 15, y: 15, w: 70, h: 70 });
              }
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-[6px] bg-[#1A1E25] hover:bg-[#232934] border border-[#2A303B] text-[12px] text-[#C9D1D9] cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3.5 py-1.5 rounded-[6px] bg-[#181B20] hover:bg-[#22262E] border border-[#2A2F38] text-[12px] font-medium text-[#E6EDF3] cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => onApply(mode === 'warp_perspective' ? perspectivePts : roiBox)}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-[6px] bg-[#F58220] hover:bg-[#FF9233] text-[#0B0D10] font-semibold text-[12px] cursor-pointer"
            >
              <Check className="w-3.5 h-3.5" />
              <span>Apply &amp; Run</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
