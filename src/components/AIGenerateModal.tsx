import React, { useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  ImagePlus,
  Loader2,
  Sparkles,
  Trash2,
  Upload,
  Wand2,
  X,
} from 'lucide-react';
import { PipelineEdge, PipelineNode } from '../types/pipeline';

interface AIGenerateModalProps {
  open: boolean;
  currentInputDataUrl?: string | null;
  onClose: () => void;
  onApplyPipeline: (generated: {
    name: string;
    description: string;
    nodes: PipelineNode[];
    edges: PipelineEdge[];
  }) => void;
}

interface ReferenceImageState {
  name: string;
  width: number;
  height: number;
  applyDataUrl: string;
  visionDataUrl: string;
}

const EXAMPLE_PROMPTS = [
  'Apply CLAHE enhancement, detect Canny edges, dilate them, and blend back onto the image with an OSD bounding box.',
  'Rotate and scale the image with Warp Affine, apply Bilateral Filter, then compute Otsu Threshold and Find Contours.',
  'Use Warp Perspective keystone transform, convert to HSV, draw histogram, and overlay OSD Telemetry HUD.',
];

async function prepareReferenceImage(src: string, name: string): Promise<ReferenceImageState> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const origW = img.naturalWidth || img.width;
      const origH = img.naturalHeight || img.height;
      if (!origW || !origH) {
        reject(new Error('Invalid image dimensions.'));
        return;
      }

      const renderScaled = (maxDim: number, mime: string, quality?: number) => {
        const scale = Math.min(1, maxDim / Math.max(origW, origH));
        const w = Math.max(32, Math.round(origW * scale));
        const h = Math.max(32, Math.round(origH * scale));
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('Canvas 2D context unavailable.');
        ctx.drawImage(img, 0, 0, w, h);
        return canvas.toDataURL(mime, quality);
      };

      try {
        const applyDataUrl = renderScaled(720, 'image/png');
        const visionDataUrl = renderScaled(512, 'image/jpeg', 0.84);
        resolve({
          name,
          width: origW,
          height: origH,
          applyDataUrl,
          visionDataUrl,
        });
      } catch (err) {
        reject(err instanceof Error ? err : new Error('Failed to process image.'));
      }
    };
    img.onerror = () => reject(new Error('Failed to decode selected image file.'));
    img.src = src;
  });
}

export const AIGenerateModal: React.FC<AIGenerateModalProps> = ({
  open,
  currentInputDataUrl,
  onClose,
  onApplyPipeline,
}) => {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<{ configured: boolean; model: string } | null>(null);
  const [referenceImage, setReferenceImage] = useState<ReferenceImageState | null>(null);
  const [applyAsInput, setApplyAsInput] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!open) return;
    setError(null);
    fetch('/api/ai/status')
      .then((r) => r.json())
      .then((data) => setStatus(data))
      .catch(() => setStatus(null));
  }, [open]);

  if (!open) return null;

  const handleFileSelected = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file (PNG, JPG, WebP, or GIF).');
      return;
    }
    setError(null);
    const reader = new FileReader();
    reader.onload = async () => {
      if (typeof reader.result !== 'string') return;
      try {
        const prepared = await prepareReferenceImage(reader.result, file.name);
        setReferenceImage(prepared);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load reference image.');
      }
    };
    reader.onerror = () => setError('Failed to read image file.');
    reader.readAsDataURL(file);
  };

  const handleUseCurrentCanvas = async () => {
    if (!currentInputDataUrl) return;
    setError(null);
    try {
      const prepared = await prepareReferenceImage(currentInputDataUrl, 'canvas_input.png');
      setReferenceImage(prepared);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to capture canvas input.');
    }
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if ((!prompt.trim() && !referenceImage) || loading) return;
    setLoading(true);
    setError(null);

    try {
      const resp = await fetch('/api/ai/generate-pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          referenceImage: referenceImage?.visionDataUrl || undefined,
          referenceMeta: referenceImage
            ? {
                name: referenceImage.name,
                width: referenceImage.width,
                height: referenceImage.height,
              }
            : undefined,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data?.error || `Request failed (${resp.status})`);
      }
      if (data?.pipeline) {
        const pipeline = data.pipeline as {
          name: string;
          description: string;
          nodes: PipelineNode[];
          edges: PipelineEdge[];
        };

        if (referenceImage && applyAsInput) {
          pipeline.nodes = pipeline.nodes.map((node) =>
            node.type === 'load_image'
              ? {
                  ...node,
                  params: {
                    ...node.params,
                    customDataUrl: referenceImage.applyDataUrl,
                  },
                }
              : node
          );
        }

        onApplyPipeline(pipeline);
        onClose();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate pipeline');
    } finally {
      setLoading(false);
    }
  };

  const canSubmit = Boolean(prompt.trim() || referenceImage) && !loading;

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="w-full max-w-lg max-h-[90vh] flex flex-col rounded-xl bg-[#121417] border border-[#2A2F38] shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="shrink-0 flex items-center justify-between px-4 py-3 bg-[#15181D] border-b border-[#22262C]">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-[6px] bg-[#251A12] border border-[#F58220]/40 flex items-center justify-center text-[#F58220]">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-[14.5px] font-semibold text-white">
                AI Pipeline Generator
              </h2>
              <p className="text-[11px] text-[#8B949E]">
                Describe your OpenCV workflow or upload a reference input image
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="w-7 h-7 rounded-[6px] bg-[#1C2027] hover:bg-[#272D37] text-[#9CA3AF] hover:text-white flex items-center justify-center cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <form onSubmit={handleGenerate} className="p-4 space-y-4 overflow-y-auto">
          {/* Server Env Status */}
          {status && (
            <div className="flex items-center justify-between px-3 py-2 rounded-[6px] bg-[#0D0F13] border border-[#22262C] text-[11.5px]">
              <span className="text-[#8B949E]">
                Model: <span className="font-mono-tabular text-[#E6EDF3]">{status.model}</span>
              </span>
              <span
                className={`font-mono-tabular ${
                  status.configured ? 'text-[#4ADE80]' : 'text-[#FBBF24]'
                }`}
              >
                {status.configured ? 'Server Connected' : 'AI_SECRET Unset'}
              </span>
            </div>
          )}

          {/* Reference Input Image Upload */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[12px] font-medium text-[#C9D1D9]">
                Reference Input Image <span className="text-[#6E7681] font-normal">(Optional)</span>
              </span>
              {currentInputDataUrl && !referenceImage && (
                <button
                  type="button"
                  onClick={handleUseCurrentCanvas}
                  className="text-[11px] text-[#389E82] hover:text-[#4BC4A3] font-medium cursor-pointer"
                >
                  Use Current Input
                </button>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/gif"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileSelected(file);
                e.target.value = '';
              }}
            />

            {!referenceImage ? (
              <div
                role="button"
                tabIndex={0}
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    fileInputRef.current?.click();
                  }
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  const file = e.dataTransfer.files?.[0];
                  if (file) handleFileSelected(file);
                }}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-[7px] border border-dashed transition-colors cursor-pointer ${
                  isDragging
                    ? 'bg-[#17221E] border-[#389E82]'
                    : 'bg-[#0B0D10] border-[#282E38] hover:border-[#3D4654] hover:bg-[#101318]'
                }`}
              >
                <div className="w-9 h-9 rounded-[6px] bg-[#161B22] border border-[#262C36] flex items-center justify-center text-[#8B949E] shrink-0">
                  <ImagePlus className="w-4 h-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-[12px] text-[#D0D7DE] font-medium">
                    Click to upload or drag & drop reference image
                  </p>
                  <p className="text-[10.5px] text-[#6E7681] truncate">
                    Sent to AI for visual context & applied directly to Load Image node
                  </p>
                </div>
                <Upload className="w-3.5 h-3.5 text-[#6E7681] shrink-0" />
              </div>
            ) : (
              <div className="p-2.5 rounded-[7px] bg-[#0B0D10] border border-[#2B333E] space-y-2">
                <div className="flex items-center gap-3">
                  <img
                    src={referenceImage.applyDataUrl}
                    alt="Reference input preview"
                    className="w-12 h-12 rounded-[5px] object-cover border border-[#262C36] bg-[#16191E] shrink-0"
                  />
                  <div className="min-w-0 flex-1">
                    <p className="text-[12px] font-medium text-[#E6EDF3] truncate">
                      {referenceImage.name}
                    </p>
                    <p className="text-[11px] font-mono-tabular text-[#8B949E]">
                      {referenceImage.width}×{referenceImage.height} px · Ready for AI + Load Image
                    </p>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      title="Replace reference image"
                      className="px-2 py-1 rounded-[5px] bg-[#171B22] hover:bg-[#222833] border border-[#282E38] text-[11px] text-[#C9D1D9] cursor-pointer"
                    >
                      Change
                    </button>
                    <button
                      type="button"
                      onClick={() => setReferenceImage(null)}
                      title="Remove reference image"
                      className="w-7 h-7 rounded-[5px] bg-[#1B1517] hover:bg-[#2C1B1F] border border-[#3B2328] text-[#F87171] flex items-center justify-center cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                <label className="flex items-center gap-2 pt-1 border-t border-[#1A1F26] cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={applyAsInput}
                    onChange={(e) => setApplyAsInput(e.target.checked)}
                    className="accent-[#389E82] w-3.5 h-3.5 rounded cursor-pointer"
                  />
                  <span className="text-[11px] text-[#9CA3AF]">
                    Use uploaded image as the generated <span className="font-mono-tabular text-[#D0D7DE]">Load Image</span> input
                  </span>
                </label>
              </div>
            )}
          </div>

          <label className="block space-y-1.5">
            <span className="text-[12px] font-medium text-[#C9D1D9] block">
              Pipeline Prompt
            </span>
            <textarea
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={
                referenceImage
                  ? 'Describe what to do with this reference image (or leave blank to auto-design an optimal pipeline)...'
                  : 'e.g. Load image, apply Warp Perspective, enhance with CLAHE, extract Canny edges, and add an OSD Telemetry HUD...'
              }
              className="w-full rounded-[7px] bg-[#0B0D10] border border-[#282E38] focus:border-[#389E82] p-3 text-[12.5px] text-white placeholder:text-[#636B76] outline-none resize-none leading-relaxed"
            />
          </label>

          {/* Example Prompts */}
          <div className="space-y-1.5">
            <span className="text-[11px] font-medium text-[#7D8590] block">Quick Prompts</span>
            <div className="space-y-1.5">
              {EXAMPLE_PROMPTS.map((ex, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setPrompt(ex)}
                  className="w-full text-left px-2.5 py-1.5 rounded-[6px] bg-[#16191E] hover:bg-[#1D2229] border border-[#242932] text-[11px] text-[#9CA3AF] hover:text-[#E6EDF3] transition-colors cursor-pointer leading-snug"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>

          {error && (
            <div className="flex items-start gap-2 px-3 py-2.5 rounded-[6px] bg-[#261416] border border-[#7F1D1D] text-[11.5px] text-[#FCA5A5]">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-[#F87171]" />
              <span>{error}</span>
            </div>
          )}

          <div className="flex items-center justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={onClose}
              className="px-3.5 py-1.5 rounded-[6px] bg-[#1A1E25] hover:bg-[#242933] border border-[#2D333E] text-[12.5px] font-medium text-[#E6EDF3] cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!canSubmit}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-[6px] bg-[#F58220] hover:bg-[#FF9233] disabled:opacity-50 text-[#0B0D10] text-[12.5px] font-semibold cursor-pointer"
            >
              {loading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Generating...</span>
                </>
              ) : (
                <>
                  <Wand2 className="w-3.5 h-3.5" />
                  <span>Generate Pipeline</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
