import React, { useEffect, useRef, useState } from 'react';
import {
  Download,
  Maximize2,
  Minimize2,
  SplitSquareHorizontal,
  X,
} from 'lucide-react';
import { NodeExecutionResult } from '../types/pipeline';

interface ResultPreviewModalProps {
  open: boolean;
  nodeTitle: string;
  nodeType: string;
  result: NodeExecutionResult | null;
  onClose: () => void;
  onExportImage: () => void;
}

type PopupSize = 'medium' | 'large' | 'full';
type ZoomMode = 'fit' | 1 | 2 | 3;

interface ProbeValues {
  x: number;
  y: number;
  r: number;
  g: number;
  b: number;
  hCv: number;
  sCv: number;
  vCv: number;
  yCbCrY: number;
  cr: number;
  cb: number;
}

function rgbToProbe(x: number, y: number, r: number, g: number, b: number): ProbeValues {
  const rf = r / 255;
  const gf = g / 255;
  const bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const d = max - min;
  let h = 0;
  if (d > 0) {
    if (max === rf) h = 60 * (((gf - bf) / d) % 6);
    else if (max === gf) h = 60 * ((bf - rf) / d + 2);
    else h = 60 * ((rf - gf) / d + 4);
  }
  if (h < 0) h += 360;
  const hCv = Math.round(h / 2) % 180;
  const sCv = max === 0 ? 0 : Math.round((d / max) * 255);
  const vCv = Math.round(max * 255);

  const yVal = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  const crVal = Math.max(0, Math.min(255, Math.round(128 + 0.713 * (r - yVal))));
  const cbVal = Math.max(0, Math.min(255, Math.round(128 + 0.564 * (b - yVal))));

  return { x, y, r, g, b, hCv, sCv, vCv, yCbCrY: yVal, cr: crVal, cb: cbVal };
}

export const ResultPreviewModal: React.FC<ResultPreviewModalProps> = ({
  open,
  nodeTitle,
  nodeType,
  result,
  onClose,
  onExportImage,
}) => {
  const [popupSize, setPopupSize] = useState<PopupSize>('large');
  const [viewMode, setViewMode] = useState<'output' | 'compare'>('output');
  const [zoomMode, setZoomMode] = useState<ZoomMode>('fit');
  const [splitPct, setSplitPct] = useState<number>(50);
  const [probe, setProbe] = useState<ProbeValues | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open || !result) return null;

  const canCompare = Boolean(result.inputPreviewDataUrl);
  const activeCompare = viewMode === 'compare' && canCompare;

  const handleImageMouseMove = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!result.imageData || !imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const relX = e.clientX - rect.left;
    const relY = e.clientY - rect.top;
    if (relX < 0 || relY < 0 || relX > rect.width || relY > rect.height) {
      setProbe(null);
      return;
    }

    const px = Math.min(result.width - 1, Math.max(0, Math.floor((relX / rect.width) * result.width)));
    const py = Math.min(result.height - 1, Math.max(0, Math.floor((relY / rect.height) * result.height)));
    const idx = (py * result.width + px) * 4;
    const d = result.imageData.data;
    setProbe(rgbToProbe(px, py, d[idx], d[idx + 1], d[idx + 2]));
  };

  const sizeClass =
    popupSize === 'medium'
      ? 'max-w-2xl h-[70vh]'
      : popupSize === 'full'
        ? 'max-w-[96vw] h-[94vh]'
        : 'max-w-5xl h-[84vh]';

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 bg-black/80 backdrop-blur-[2px] flex items-center justify-center p-3 sm:p-5"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className={`w-full ${sizeClass} bg-[#121417] border border-[#272C35] rounded-xl shadow-2xl flex flex-col overflow-hidden transition-all duration-150`}
      >
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 border-b border-[#22262C] bg-[#15181D] shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h2 className="text-[14px] font-semibold text-white truncate">{nodeTitle}</h2>
                <span className="text-[11px] font-mono-tabular text-[#7D8590]">{nodeType}</span>
              </div>
              <div className="flex items-center gap-2.5 text-[11px] font-mono-tabular text-[#8B949E] mt-0.5">
                <span>
                  {result.width}×{result.height}px · {result.channels}ch
                </span>
                <span className={result.bypassed ? 'text-[#FBBF24]' : 'text-[#4ADE80]'}>
                  {result.bypassed ? 'BYPASSED' : `${result.durationMs.toFixed(1)}ms`}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            {/* Window Size Selector (Medium / Large / Full) */}
            <div className="flex items-center rounded-[6px] bg-[#0B0D10] border border-[#242932] p-0.5 text-[11px]">
              <button
                type="button"
                onClick={() => setPopupSize('medium')}
                className={`px-2 py-1 rounded-[4px] cursor-pointer transition-colors ${
                  popupSize === 'medium' ? 'bg-[#222731] text-white font-medium' : 'text-[#8B949E] hover:text-white'
                }`}
              >
                Compact
              </button>
              <button
                type="button"
                onClick={() => setPopupSize('large')}
                className={`px-2 py-1 rounded-[4px] cursor-pointer transition-colors ${
                  popupSize === 'large' ? 'bg-[#222731] text-white font-medium' : 'text-[#8B949E] hover:text-white'
                }`}
              >
                Large
              </button>
              <button
                type="button"
                onClick={() => setPopupSize('full')}
                className={`px-2 py-1 rounded-[4px] flex items-center gap-1 cursor-pointer transition-colors ${
                  popupSize === 'full' ? 'bg-[#222731] text-white font-medium' : 'text-[#8B949E] hover:text-white'
                }`}
              >
                {popupSize === 'full' ? <Minimize2 className="w-3 h-3" /> : <Maximize2 className="w-3 h-3" />}
                <span>Full</span>
              </button>
            </div>

            <button
              type="button"
              onClick={onClose}
              title="Close popup (Esc)"
              className="w-7 h-7 rounded-md bg-[#1B1F27] hover:bg-[#252A35] text-[#9CA3AF] hover:text-white flex items-center justify-center cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Sub-Toolbar: Output/Compare + Zoom + Save PNG */}
        <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2 border-b border-[#22262C] bg-[#111419] shrink-0">
          <div className="flex items-center gap-2">
            <div className="flex items-center rounded-[5px] bg-[#0B0D10] border border-[#242932] p-0.5 text-[11px]">
              <button
                type="button"
                onClick={() => setViewMode('output')}
                className={`px-2.5 py-1 rounded-[4px] cursor-pointer transition-colors ${
                  !activeCompare ? 'bg-[#222731] text-white font-medium' : 'text-[#8B949E]'
                }`}
              >
                Output
              </button>
              <button
                type="button"
                disabled={!canCompare}
                onClick={() => setViewMode('compare')}
                title={canCompare ? 'Before/After Comparison Slider' : 'Connect an upstream node to compare'}
                className={`px-2.5 py-1 rounded-[4px] flex items-center gap-1.5 cursor-pointer transition-colors disabled:opacity-35 ${
                  activeCompare ? 'bg-[#389E82] text-[#0B0D10] font-semibold' : 'text-[#8B949E]'
                }`}
              >
                <SplitSquareHorizontal className="w-3.5 h-3.5" />
                <span>Compare Before / After</span>
              </button>
            </div>

            {!activeCompare && (
              <div className="flex items-center rounded-[5px] bg-[#0B0D10] border border-[#242932] p-0.5 text-[11px] font-mono-tabular">
                {(['fit', 1, 2, 3] as ZoomMode[]).map((z) => (
                  <button
                    key={String(z)}
                    type="button"
                    onClick={() => setZoomMode(z)}
                    className={`px-2 py-1 rounded-[4px] cursor-pointer transition-colors ${
                      zoomMode === z ? 'bg-[#F58220] text-[#0B0D10] font-semibold' : 'text-[#8B949E] hover:text-white'
                    }`}
                  >
                    {z === 'fit' ? 'Fit' : `${z}×`}
                  </button>
                ))}
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={onExportImage}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-[6px] bg-[#191D24] hover:bg-[#232933] border border-[#2D333E] text-[12px] font-medium text-[#E6EDF3] cursor-pointer"
          >
            <Download className="w-3.5 h-3.5 text-[#F58220]" />
            <span>Save Output PNG</span>
          </button>
        </div>

        {/* Main Image Viewport */}
        <div className="flex-1 min-h-0 bg-[#080A0D] bg-checkerboard overflow-auto flex items-center justify-center p-4 relative">
          {activeCompare && result.inputPreviewDataUrl ? (
            <div
              className="relative select-none flex items-center justify-center max-w-full max-h-full"
              style={{
                width: 'min(100%, 760px)',
                aspectRatio: `${result.width} / ${result.height}`,
              }}
            >
              <img
                src={result.previewDataUrl}
                alt="After output"
                className="w-full h-full object-contain pointer-events-none rounded"
              />
              <div
                className="absolute inset-0 overflow-hidden pointer-events-none"
                style={{ clipPath: `inset(0 ${100 - splitPct}% 0 0)` }}
              >
                <img
                  src={result.inputPreviewDataUrl}
                  alt="Before input"
                  className="w-full h-full object-contain rounded"
                />
              </div>
              <div
                className="absolute top-0 bottom-0 w-[2px] bg-[#F58220] pointer-events-none shadow-lg"
                style={{ left: `${splitPct}%` }}
              >
                <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-6 h-9 rounded-md bg-[#F58220] text-[#0B0D10] text-xs font-bold flex items-center justify-center shadow-lg">
                  ↔
                </div>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                value={splitPct}
                onChange={(e) => setSplitPct(Number(e.target.value))}
                aria-label="Large comparison split slider"
                className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize"
              />
              <span className="absolute top-2.5 left-2.5 px-2 py-0.5 rounded bg-[#0B0D10]/85 border border-[#242933] text-[10px] font-mono-tabular text-[#C9D1D9] pointer-events-none">
                BEFORE
              </span>
              <span className="absolute top-2.5 right-2.5 px-2 py-0.5 rounded bg-[#0B0D10]/85 border border-[#242933] text-[10px] font-mono-tabular text-[#4ADE80] pointer-events-none">
                AFTER
              </span>
            </div>
          ) : (
            <img
              ref={imgRef}
              src={result.previewDataUrl}
              alt={nodeTitle}
              onMouseMove={handleImageMouseMove}
              onMouseLeave={() => setProbe(null)}
              style={
                zoomMode === 'fit'
                  ? undefined
                  : {
                      width: result.width * zoomMode,
                      height: result.height * zoomMode,
                      maxWidth: 'none',
                      maxHeight: 'none',
                      imageRendering: zoomMode >= 2 ? 'pixelated' : 'auto',
                    }
              }
              className={
                zoomMode === 'fit'
                  ? 'max-w-full max-h-full object-contain cursor-crosshair rounded shadow-2xl'
                  : 'cursor-crosshair rounded shadow-2xl shrink-0'
              }
            />
          )}
        </div>

        {/* Bottom Pixel Probe Readout Bar */}
        <div className="px-4 py-2 border-t border-[#22262C] bg-[#15181D] flex flex-wrap items-center justify-between gap-2 text-[11.5px] font-mono-tabular shrink-0">
          {probe ? (
            <>
              <div className="flex items-center gap-3">
                <span className="text-[#38BDF8]">
                  XY ({probe.x}, {probe.y})
                </span>
                <span className="flex items-center gap-1.5 text-[#E6EDF3]">
                  <span
                    className="inline-block w-3 h-3 rounded-xs border border-white/30"
                    style={{ backgroundColor: `rgb(${probe.r},${probe.g},${probe.b})` }}
                  />
                  RGB({probe.r}, {probe.g}, {probe.b})
                </span>
              </div>
              <div className="flex items-center gap-4 text-[#9CA3AF]">
                <span>HSV({probe.hCv}, {probe.sCv}, {probe.vCv})</span>
                <span>YCrCb({probe.yCbCrY}, {probe.cr}, {probe.cb})</span>
              </div>
            </>
          ) : (
            <span className="text-[#7D8590]">
              {activeCompare
                ? 'Drag slider horizontally to compare upstream input vs processed OpenCV output.'
                : 'Hover over the image to inspect live XY coordinates, RGB, HSV, and YCrCb values.'}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
