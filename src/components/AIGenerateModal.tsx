import React, { useEffect, useState } from 'react';
import { AlertCircle, Loader2, Sparkles, Wand2, X } from 'lucide-react';
import { PipelineEdge, PipelineNode } from '../types/pipeline';

interface AIGenerateModalProps {
  open: boolean;
  onClose: () => void;
  onApplyPipeline: (generated: {
    name: string;
    description: string;
    nodes: PipelineNode[];
    edges: PipelineEdge[];
  }) => void;
}

const EXAMPLE_PROMPTS = [
  'Apply CLAHE enhancement, detect Canny edges, dilate them, and blend back onto Lena with an OSD bounding box.',
  'Rotate and scale the image with Warp Affine, apply Bilateral Filter, then compute Otsu Threshold and Find Contours.',
  'Use Warp Perspective keystone transform, convert to HSV, draw histogram, and overlay OSD Telemetry HUD.',
];

export const AIGenerateModal: React.FC<AIGenerateModalProps> = ({
  open,
  onClose,
  onApplyPipeline,
}) => {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<{ configured: boolean; model: string } | null>(null);

  useEffect(() => {
    if (!open) return;
    setError(null);
    fetch('/api/ai/status')
      .then((r) => r.json())
      .then((data) => setStatus(data))
      .catch(() => setStatus(null));
  }, [open]);

  if (!open) return null;

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);

    try {
      const resp = await fetch('/api/ai/generate-pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.trim() }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data?.error || `Request failed (${resp.status})`);
      }
      if (data?.pipeline) {
        onApplyPipeline(data.pipeline);
        onClose();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate pipeline');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="w-full max-w-lg rounded-xl bg-[#121417] border border-[#2A2F38] shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-[#15181D] border-b border-[#22262C]">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-[6px] bg-[#251A12] border border-[#F58220]/40 flex items-center justify-center text-[#F58220]">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-[14.5px] font-semibold text-white">
                AI Pipeline Generator
              </h2>
              <p className="text-[11px] text-[#8B949E]">
                Describe your OpenCV workflow to synthesize nodes and connections
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

        <form onSubmit={handleGenerate} className="p-4 space-y-4">
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

          <label className="block space-y-1.5">
            <span className="text-[12px] font-medium text-[#C9D1D9] block">
              Pipeline Prompt
            </span>
            <textarea
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g. Load image, apply Warp Perspective, enhance with CLAHE, extract Canny edges, and add an OSD Telemetry HUD..."
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
              disabled={!prompt.trim() || loading}
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
