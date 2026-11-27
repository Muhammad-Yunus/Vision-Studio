import React, { useRef, useState } from 'react';
import {
  Check,
  ChevronDown,
  Copy,
  Download,
  Layers,
  Paintbrush,
  Upload,
} from 'lucide-react';
import { NODE_CATALOG_MAP } from '../data/nodeCatalog';
import { generatePythonOpenCVCode } from '../engine/codeGenerator';
import {
  NodeExecutionResult,
  OSDElement,
  PipelineEdge,
  PipelineNode,
} from '../types/pipeline';

interface InspectorSidebarProps {
  selectedNode: PipelineNode | null;
  selectedResult: NodeExecutionResult | null;
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  onUpdateParam: (nodeId: string, key: string, value: unknown) => void;
  onExportPython: () => void;
  onExportNodeImage: (nodeId: string) => void;
  onOpenOSDEditor: (nodeId: string) => void;
  className?: string;
}

export const InspectorSidebar: React.FC<InspectorSidebarProps> = ({
  selectedNode,
  selectedResult,
  nodes,
  edges,
  onUpdateParam,
  onExportPython,
  onExportNodeImage,
  onOpenOSDEditor,
  className = '',
}) => {
  const [bottomTab, setBottomTab] = useState<'code' | 'results'>('code');
  const [showCodePreview, setShowCodePreview] = useState(false);
  const [copied, setCopied] = useState(false);
  const uploadRef = useRef<HTMLInputElement>(null);

  const meta = selectedNode ? NODE_CATALOG_MAP[selectedNode.type] : null;
  const pythonCode = generatePythonOpenCVCode(nodes, edges);

  const handleCopyCode = async () => {
    try {
      await navigator.clipboard.writeText(pythonCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // ignore
    }
  };

  const handleImageUpload = (file: File) => {
    if (!selectedNode) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        onUpdateParam(selectedNode.id, 'customDataUrl', reader.result);
      }
    };
    reader.readAsDataURL(file);
  };

  return (
    <aside
      className={`w-[292px] sm:w-[304px] bg-[#121417] border-l border-[#22262C] flex flex-col h-full shrink-0 ${className}`}
    >
      {/* Top Half: Selected Node Parameters */}
      <div className="flex-1 overflow-y-auto px-4 pt-3.5 pb-4">
        {selectedNode && meta ? (
          <div>
            <h2 className="text-[16px] font-semibold tracking-[-0.01em] text-[#F3F4F6]">
              {meta.title}
            </h2>
            <div className="text-[11px] font-mono-tabular text-[#7D8590] mt-0.5 mb-4">
              {selectedNode.type}
            </div>

            <div className="space-y-3.5">
              {meta.params.length === 0 && (
                <p className="text-[12px] text-[#7D8590]">
                  This node operates automatically with zero configurable parameters.
                </p>
              )}

              {meta.params.map((param) => {
                const val = selectedNode.params[param.key] ?? param.defaultValue;

                if (param.type === 'select') {
                  return (
                    <label key={param.key} className="block space-y-1.5">
                      <span className="text-[11.5px] font-medium text-[#9CA3AF] block">
                        {param.label}
                      </span>
                      <div className="relative">
                        <select
                          value={String(val)}
                          onChange={(e) => onUpdateParam(selectedNode.id, param.key, e.target.value)}
                          className="w-full appearance-none rounded-[6px] bg-[#0F1115] border border-[#262B34] focus:border-[#389E82] px-3 py-2 pr-8 text-[12.5px] text-[#F3F4F6] outline-none cursor-pointer"
                        >
                          {param.options?.map((opt) => (
                            <option key={String(opt.value)} value={String(opt.value)}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                        <ChevronDown className="w-3.5 h-3.5 text-[#8B949E] absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                      </div>
                    </label>
                  );
                }

                if (param.type === 'image_source') {
                  return (
                    <div key={param.key} className="space-y-2.5">
                      <label className="block space-y-1.5">
                        <span className="text-[11.5px] font-medium text-[#9CA3AF] block">
                          {param.label}
                        </span>
                        <div className="relative">
                          <select
                            value={String(val)}
                            onChange={(e) => {
                              onUpdateParam(selectedNode.id, 'customDataUrl', '');
                              onUpdateParam(selectedNode.id, param.key, e.target.value);
                            }}
                            className="w-full appearance-none rounded-[6px] bg-[#0F1115] border border-[#262B34] focus:border-[#389E82] px-3 py-2 pr-8 text-[12.5px] text-[#F3F4F6] outline-none cursor-pointer"
                          >
                            {param.options?.map((opt) => (
                              <option key={String(opt.value)} value={String(opt.value)}>
                                {opt.label}
                              </option>
                            ))}
                          </select>
                          <ChevronDown className="w-3.5 h-3.5 text-[#8B949E] absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                        </div>
                      </label>

                      <button
                        type="button"
                        onClick={() => uploadRef.current?.click()}
                        className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-[6px] bg-[#181B21] hover:bg-[#21262F] border border-[#2B313C] text-[12px] font-medium text-[#E6EDF3] transition-colors cursor-pointer"
                      >
                        <Upload className="w-3.5 h-3.5 text-[#F58220]" />
                        <span>Upload Custom Image</span>
                      </button>
                      <input
                        ref={uploadRef}
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) handleImageUpload(file);
                          e.target.value = '';
                        }}
                      />
                    </div>
                  );
                }

                if (param.type === 'number') {
                  return (
                    <label key={param.key} className="block space-y-1.5">
                      <span className="text-[11.5px] font-medium text-[#9CA3AF] block">
                        {param.label}
                      </span>
                      <input
                        type="number"
                        value={Number(val)}
                        min={param.min}
                        max={param.max}
                        step={param.step ?? 1}
                        onChange={(e) =>
                          onUpdateParam(selectedNode.id, param.key, Number(e.target.value))
                        }
                        className="w-full rounded-[6px] bg-[#0F1115] border border-[#262B34] focus:border-[#389E82] px-3 py-2 text-[12.5px] font-mono-tabular text-[#F3F4F6] outline-none"
                      />
                    </label>
                  );
                }

                if (param.type === 'range') {
                  return (
                    <label key={param.key} className="block space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[11.5px] font-medium text-[#9CA3AF]">
                          {param.label}
                        </span>
                        <span className="text-[11.5px] font-mono-tabular text-[#E6EDF3]">
                          {Number(val)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={param.min ?? 0}
                        max={param.max ?? 100}
                        step={param.step ?? 1}
                        value={Number(val)}
                        onChange={(e) =>
                          onUpdateParam(selectedNode.id, param.key, Number(e.target.value))
                        }
                        className="w-full studio-range"
                      />
                    </label>
                  );
                }

                if (param.type === 'boolean') {
                  return (
                    <label
                      key={param.key}
                      className="flex items-center justify-between py-1 cursor-pointer"
                    >
                      <span className="text-[12px] font-medium text-[#C9D1D9]">{param.label}</span>
                      <input
                        type="checkbox"
                        checked={Boolean(val)}
                        onChange={(e) =>
                          onUpdateParam(selectedNode.id, param.key, e.target.checked)
                        }
                        className="w-4 h-4 accent-[#F58220] rounded cursor-pointer"
                      />
                    </label>
                  );
                }

                if (param.type === 'color') {
                  return (
                    <label key={param.key} className="flex items-center justify-between py-1">
                      <span className="text-[11.5px] font-medium text-[#9CA3AF]">{param.label}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] font-mono-tabular text-[#C9D1D9]">
                          {String(val)}
                        </span>
                        <input
                          type="color"
                          value={String(val)}
                          onChange={(e) =>
                            onUpdateParam(selectedNode.id, param.key, e.target.value)
                          }
                          className="w-7 h-6 rounded border border-[#2C323C] bg-transparent cursor-pointer"
                        />
                      </div>
                    </label>
                  );
                }

                if (param.type === 'text' && param.key !== 'customDataUrl') {
                  return (
                    <label key={param.key} className="block space-y-1.5">
                      <span className="text-[11.5px] font-medium text-[#9CA3AF] block">
                        {param.label}
                      </span>
                      <input
                        type="text"
                        value={String(val)}
                        onChange={(e) =>
                          onUpdateParam(selectedNode.id, param.key, e.target.value)
                        }
                        className="w-full rounded-[6px] bg-[#0F1115] border border-[#262B34] focus:border-[#389E82] px-3 py-2 text-[12.5px] font-mono-tabular text-[#F3F4F6] outline-none"
                      />
                    </label>
                  );
                }

                if (param.type === 'osd_layers') {
                  const elements = Array.isArray(val) ? (val as OSDElement[]) : [];
                  return (
                    <div key={param.key} className="space-y-2.5 pt-1">
                      <button
                        type="button"
                        onClick={() => onOpenOSDEditor(selectedNode.id)}
                        className="w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-[6px] bg-[#F58220] hover:bg-[#FF9233] text-[#0B0D10] font-semibold text-[12.5px] transition-colors shadow cursor-pointer"
                      >
                        <Paintbrush className="w-3.5 h-3.5" />
                        <span>Open OSD Drag &amp; Drop Editor</span>
                      </button>

                      <div className="rounded-[6px] bg-[#0F1115] border border-[#242932] p-2.5 space-y-1.5">
                        <div className="flex items-center justify-between text-[11px] text-[#8B949E]">
                          <span className="flex items-center gap-1.5">
                            <Layers className="w-3 h-3" />
                            Active OSD Primitives
                          </span>
                          <span className="font-mono-tabular text-white">{elements.length}</span>
                        </div>
                        {elements.slice(0, 5).map((el) => (
                          <div
                            key={el.id}
                            className="flex items-center justify-between text-[11px] py-0.5 text-[#C9D1D9]"
                          >
                            <span className="truncate">{el.name}</span>
                            <span
                              className="font-mono-tabular text-[10px] uppercase"
                              style={{ color: el.color }}
                            >
                              {el.kind}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                }

                return null;
              })}
            </div>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center px-4">
            <p className="text-[13px] font-medium text-[#C9D1D9]">No node selected</p>
            <p className="text-[11.5px] text-[#7D8590] mt-1">
              Click any node in the graph to inspect parameters and preview output.
            </p>
          </div>
        )}
      </div>

      {/* Bottom Half: Code | Results Tabs matching Mockup */}
      <div className="border-t border-[#22262C] bg-[#121417] flex flex-col shrink-0">
        {/* Tab Bar with Emerald Active Indicator */}
        <div className="grid grid-cols-2 border-b border-[#22262C] select-none">
          <button
            type="button"
            onClick={() => setBottomTab('code')}
            className={`py-2.5 text-[12px] font-medium text-center border-b-2 transition-colors cursor-pointer ${
              bottomTab === 'code'
                ? 'text-[#F3F4F6] border-[#22C55E]'
                : 'text-[#7D8590] hover:text-[#C9D1D9] border-transparent'
            }`}
          >
            Code
          </button>
          <button
            type="button"
            onClick={() => setBottomTab('results')}
            className={`py-2.5 text-[12px] font-medium text-center border-b-2 transition-colors cursor-pointer ${
              bottomTab === 'results'
                ? 'text-[#F3F4F6] border-[#22C55E]'
                : 'text-[#7D8590] hover:text-[#C9D1D9] border-transparent'
            }`}
          >
            Results
          </button>
        </div>

        {/* Tab Body */}
        <div className="p-4 min-h-[195px] max-h-[260px] overflow-y-auto">
          {bottomTab === 'code' ? (
            <div className="space-y-3">
              <div>
                <h3 className="text-[13.5px] font-semibold text-[#F3F4F6]">Python Export</h3>
                <p className="text-[11.5px] text-[#8B949E] mt-1 leading-relaxed">
                  Generate a standalone OpenCV script from the current pipeline.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={onExportPython}
                  className="px-3.5 py-1.5 rounded-[6px] bg-[#191D24] hover:bg-[#232933] border border-[#2D333E] text-[12px] font-medium text-[#E6EDF3] transition-colors cursor-pointer"
                >
                  Export Python
                </button>
                <button
                  type="button"
                  onClick={() => setShowCodePreview((v) => !v)}
                  className="px-2.5 py-1.5 rounded-[6px] bg-[#14171C] hover:bg-[#1E232B] border border-[#252A33] text-[11.5px] text-[#9CA3AF] hover:text-white transition-colors cursor-pointer"
                >
                  {showCodePreview ? 'Hide Code' : 'Preview'}
                </button>
              </div>

              {showCodePreview && (
                <div className="relative rounded-[6px] bg-[#0B0D10] border border-[#22262C] p-2.5">
                  <button
                    type="button"
                    onClick={handleCopyCode}
                    className="absolute right-2 top-2 px-1.5 py-0.5 rounded bg-[#181C22] border border-[#2A303B] text-[10px] text-[#C9D1D9] flex items-center gap-1 cursor-pointer"
                  >
                    {copied ? <Check className="w-3 h-3 text-[#4ADE80]" /> : <Copy className="w-3 h-3" />}
                    <span>{copied ? 'Copied' : 'Copy'}</span>
                  </button>
                  <pre className="text-[10px] font-mono-tabular text-[#A5D6FF] overflow-x-auto max-h-36 leading-relaxed pr-12">
                    {pythonCode}
                  </pre>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {selectedResult ? (
                <>
                  <div className="rounded-[6px] overflow-hidden border border-[#242932] bg-checkerboard flex items-center justify-center h-32">
                    <img
                      src={selectedResult.previewDataUrl}
                      alt="Selected node output"
                      className="max-w-full max-h-full object-contain"
                    />
                  </div>
                  <div className="flex items-center justify-between text-[11px] font-mono-tabular text-[#8B949E]">
                    <span>
                      {selectedResult.width}×{selectedResult.height} · {selectedResult.channels}ch
                    </span>
                    <span className="text-[#4ADE80]">{selectedResult.durationMs.toFixed(1)}ms</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => selectedNode && onExportNodeImage(selectedNode.id)}
                    className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-[6px] bg-[#191D24] hover:bg-[#232933] border border-[#2D333E] text-[12px] font-medium text-[#E6EDF3] cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Save Output PNG</span>
                  </button>
                </>
              ) : (
                <p className="text-[12px] text-[#7D8590]">
                  Select an executed node to inspect its full-resolution output mat.
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};
