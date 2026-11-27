import React, { useEffect, useRef, useState } from 'react';
import {
  Camera,
  ChevronDown,
  Download,
  FileCode2,
  FolderOpen,
  Grid,
  Image as ImageIcon,
  Info,
  Layers,
  PanelLeft,
  PanelRight,
  Play,
  Redo2,
  SlidersHorizontal,
  Sparkles,
  Undo2,
  Wand2,
  Workflow,
} from 'lucide-react';
import { PIPELINE_TEMPLATES } from '../data/templates';
import { PipelineTemplate } from '../types/pipeline';

interface TopBarProps {
  nodeCount: number;
  linkCount: number;
  seed: number;
  samples: number;
  onSeedChange: (val: number) => void;
  onSamplesChange: (val: number) => void;
  onRun: () => void;
  onRunToSelected: () => void;
  hasSelectedNode: boolean;
  isRunning: boolean;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onAutoLayout: () => void;
  onOpenBatchModal: () => void;
  webcamActive: boolean;
  webcamFps: number;
  onSelectTemplate: (template: PipelineTemplate) => void;
  onLoadJsonFile: (file: File) => void;
  onExportPython: () => void;
  onExportJson: () => void;
  onExportImage: () => void;
  onOpenOSDEditorQuick: () => void;
  onOpenAIGenerate: () => void;
  wasmReady: boolean;
  leftDrawerOpen: boolean;
  rightDrawerOpen: boolean;
  onToggleLeftDrawer: () => void;
  onToggleRightDrawer: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  nodeCount,
  linkCount,
  seed,
  samples,
  onSeedChange,
  onSamplesChange,
  onRun,
  onRunToSelected,
  hasSelectedNode,
  isRunning,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onAutoLayout,
  onOpenBatchModal,
  webcamActive,
  webcamFps,
  onSelectTemplate,
  onLoadJsonFile,
  onExportPython,
  onExportJson,
  onExportImage,
  onOpenOSDEditorQuick,
  onOpenAIGenerate,
  wasmReady,
  leftDrawerOpen,
  rightDrawerOpen,
  onToggleLeftDrawer,
  onToggleRightDrawer,
}) => {
  const [openMenu, setOpenMenu] = useState<'info' | 'templates' | 'export' | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpenMenu(null);
      }
    };
    window.addEventListener('mousedown', handleOutside);
    return () => window.removeEventListener('mousedown', handleOutside);
  }, []);

  return (
    <header
      ref={menuRef}
      className="relative z-40 flex items-center justify-between gap-2 sm:gap-3 px-3 sm:px-4 h-14 bg-[#121417] border-b border-[#22262C] select-none shrink-0"
    >
      {/* Left Zone: Brand Lockup + Info + NODES/LINKS Counters + Graph History/Layout */}
      <div className="flex items-center gap-2 sm:gap-2.5 min-w-0">
        {/* Mobile/Tablet Left Sidebar Toggle */}
        <button
          type="button"
          onClick={onToggleLeftDrawer}
          title="Toggle Nodes Library"
          className={`lg:hidden flex items-center justify-center w-8 h-8 rounded-[6px] border transition-colors shrink-0 ${
            leftDrawerOpen
              ? 'bg-[#222730] border-[#3A414D] text-white'
              : 'bg-[#171A1F] border-[#262A31] text-[#9CA3AF] hover:text-white'
          }`}
        >
          <PanelLeft className="w-4 h-4" />
        </button>

        {/* Brand Logo & Title */}
        <div className="flex items-center gap-2.5 shrink-0">
          <div className="w-8 h-8 rounded-[7px] bg-[#181C22] border border-[#2B313B] flex items-center justify-center shadow-inner">
            <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none">
              <rect x="2.5" y="4" width="6.5" height="5.5" rx="1.5" fill="#F58220" />
              <rect x="2.5" y="14.5" width="6.5" height="5.5" rx="1.5" fill="#38BDF8" />
              <rect x="15" y="9.25" width="6.5" height="5.5" rx="1.5" fill="#22C55E" />
              <path
                d="M9 6.75H11.5C12.3 6.75 12.8 7.3 12.8 8.1V11.2C12.8 11.7 13.2 12 13.7 12H15M9 17.25H11.5C12.3 17.25 12.8 16.7 12.8 15.9V12.8C12.8 12.3 13.2 12 13.7 12H15"
                stroke="#7AB89E"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
          </div>
          <div className="flex flex-col leading-tight">
            <span className="text-[15px] font-bold tracking-[-0.01em] text-[#F3F4F6] whitespace-nowrap">
              Vision Studio
            </span>
            <span className="text-[11px] text-[#7D8590] whitespace-nowrap hidden sm:block">
              OpenCV pipeline playground
            </span>
          </div>
        </div>

        {/* Info Icon Button */}
        <div className="relative hidden sm:block">
          <button
            type="button"
            onClick={() => setOpenMenu(openMenu === 'info' ? null : 'info')}
            aria-label="Pipeline info"
            className="w-7 h-7 rounded-[6px] border border-[#262A31] bg-[#171A1F] hover:bg-[#1E232B] text-[#8B949E] hover:text-white flex items-center justify-center transition-colors cursor-pointer"
          >
            <Info className="w-3.5 h-3.5" />
          </button>

          {openMenu === 'info' && (
            <div className="absolute left-0 top-10 w-80 rounded-lg bg-[#16191E] border border-[#2C313A] shadow-2xl p-4 z-50 text-xs text-[#C9D1D9] space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-white text-sm">Vision Studio</span>
                <span className="font-mono-tabular text-[11px] text-[#7AB89E]">
                  {wasmReady ? 'OpenCV.js WASM Active' : 'Initializing WASM...'}
                </span>
              </div>
              <p className="text-[#8B949E] leading-relaxed">
                Drag nodes from the left library onto the canvas, connect ports to build an OpenCV DAG pipeline, and edit On-Screen Display overlays with the interactive drag-and-drop OSD Canvas Editor.
              </p>
              <div className="pt-2 border-t border-[#242932] grid grid-cols-2 gap-1.5 text-[11px] text-[#8B949E]">
                <span>Ctrl+Z / Ctrl+Y: Undo/Redo</span>
                <span>Key M: Bypass Node</span>
                <span>Click ×: Unlink Edge</span>
                <span>Double-click: Edit OSD/ROI</span>
              </div>
            </div>
          )}
        </div>

        {/* NODES & LINKS Metric Counters */}
        <div className="hidden md:flex items-center gap-1.5">
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-[6px] bg-[#16191E] border border-[#242830]">
            <span className="text-[10px] font-semibold tracking-wider text-[#7D8590]">NODES</span>
            <span className="text-[12px] font-semibold text-white font-mono-tabular">{nodeCount}</span>
          </div>
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-[6px] bg-[#16191E] border border-[#242830]">
            <span className="text-[10px] font-semibold tracking-wider text-[#7D8590]">LINKS</span>
            <span className="text-[12px] font-semibold text-white font-mono-tabular">{linkCount}</span>
          </div>
        </div>

        {/* Compact Undo / Redo / Auto-Layout Controls */}
        <div className="hidden sm:flex items-center gap-1 ml-0.5">
          <button
            type="button"
            onClick={onUndo}
            disabled={!canUndo}
            title="Undo (Ctrl+Z)"
            className="w-7 h-7 rounded-[6px] border border-[#262A31] bg-[#171A1F] hover:bg-[#1E232B] disabled:opacity-35 text-[#9CA3AF] hover:text-white flex items-center justify-center transition-colors cursor-pointer"
          >
            <Undo2 className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={onRedo}
            disabled={!canRedo}
            title="Redo (Ctrl+Y)"
            className="w-7 h-7 rounded-[6px] border border-[#262A31] bg-[#171A1F] hover:bg-[#1E232B] disabled:opacity-35 text-[#9CA3AF] hover:text-white flex items-center justify-center transition-colors cursor-pointer"
          >
            <Redo2 className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={onAutoLayout}
            disabled={nodeCount === 0}
            title="Auto-Layout DAG"
            className="w-7 h-7 rounded-[6px] border border-[#262A31] bg-[#171A1F] hover:bg-[#1E232B] disabled:opacity-35 text-[#9CA3AF] hover:text-[#389E82] flex items-center justify-center transition-colors cursor-pointer"
          >
            <Workflow className="w-3.5 h-3.5" />
          </button>
        </div>

        {webcamActive && (
          <div className="hidden lg:flex items-center gap-1.5 px-2 py-1 rounded-[6px] bg-[#1B1214] border border-[#F43F5E]/40 text-[#FDA4AF] text-[11px] font-mono-tabular">
            <Camera className="w-3 h-3 text-[#F43F5E]" />
            <span>LIVE {webcamFps} FPS</span>
          </div>
        )}
      </div>

      {/* Right Zone: Seed/Samples + Run Controls + Templates/Load/Export */}
      <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
        {/* Seed & Samples Inputs */}
        <div className="hidden xl:flex items-center gap-1.5 mr-0.5">
          <label className="flex flex-col bg-[#16191E] border border-[#262A31] rounded-[6px] px-2.5 py-0.5 w-[72px] focus-within:border-[#3A424E]">
            <span className="text-[9px] text-[#7D8590] leading-tight">Seed</span>
            <input
              type="number"
              value={seed}
              onChange={(e) => onSeedChange(Number(e.target.value) || 0)}
              className="bg-transparent text-[12px] font-mono-tabular text-white outline-none w-full leading-tight"
            />
          </label>
          <label className="flex flex-col bg-[#16191E] border border-[#262A31] rounded-[6px] px-2.5 py-0.5 w-[72px] focus-within:border-[#3A424E]">
            <span className="text-[9px] text-[#7D8590] leading-tight">Samples</span>
            <input
              type="number"
              min={1}
              max={16}
              value={samples}
              onChange={(e) => onSamplesChange(Math.min(16, Math.max(1, Number(e.target.value) || 1)))}
              className="bg-transparent text-[12px] font-mono-tabular text-white outline-none w-full leading-tight"
            />
          </label>
          {samples > 1 && (
            <button
              type="button"
              onClick={onOpenBatchModal}
              title="Open Multi-Seed Batch Grid"
              className="h-8 px-2 rounded-[6px] bg-[#181B20] hover:bg-[#22262E] border border-[#F58220]/45 text-[#FDBA74] text-[11px] font-medium flex items-center gap-1 cursor-pointer"
            >
              <Grid className="w-3.5 h-3.5 text-[#F58220]" />
              <span>Batch</span>
            </button>
          )}
        </div>

        {/* Primary Run CTA (Vivid Amber-Orange with Dark Label) */}
        <button
          type="button"
          onClick={onRun}
          disabled={isRunning}
          className="flex items-center gap-1.5 px-3 sm:px-3.5 py-1.5 rounded-[6px] bg-[#F58220] hover:bg-[#FF9130] active:bg-[#E07316] disabled:opacity-60 text-[#0B0D10] font-semibold text-[13px] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Play className="w-3.5 h-3.5 fill-[#0B0D10]" />
          <span>{samples > 1 ? `Run ×${samples}` : 'Run'}</span>
        </button>

        {/* Run to selected */}
        <button
          type="button"
          onClick={onRunToSelected}
          disabled={!hasSelectedNode || isRunning}
          className="hidden sm:flex items-center px-3 py-1.5 rounded-[6px] bg-[#181B20] hover:bg-[#22262E] disabled:opacity-45 border border-[#2A2F38] text-[#E6EDF3] text-[13px] font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          Run to selected
        </button>

        {/* Quick OSD Drag & Drop Editor Launcher */}
        <button
          type="button"
          onClick={onOpenOSDEditorQuick}
          title="Open Drag & Drop OpenCV OSD Canvas Editor"
          className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-[6px] bg-[#181B20] hover:bg-[#22262E] border border-[#2E7D67]/60 text-[#86EFAC] text-[12px] font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>OSD Editor</span>
        </button>

        {/* AI Generate Pipeline Button */}
        <button
          type="button"
          onClick={onOpenAIGenerate}
          title="Generate OpenCV Pipeline from Prompt"
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-[6px] bg-[#181B20] hover:bg-[#22262E] border border-[#F58220]/50 text-[#FDBA74] text-[12px] font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Wand2 className="w-3.5 h-3.5 text-[#F58220]" />
          <span className="hidden sm:inline">AI Generate</span>
        </button>

        {/* Templates Dropdown */}
        <div className="relative">
          <button
            type="button"
            onClick={() => setOpenMenu(openMenu === 'templates' ? null : 'templates')}
            className="flex items-center gap-1 px-2.5 sm:px-3 py-1.5 rounded-[6px] bg-[#181B20] hover:bg-[#22262E] border border-[#2A2F38] text-[#E6EDF3] text-[13px] font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <span className="hidden sm:inline">Templates</span>
            <Sparkles className="w-3.5 h-3.5 sm:hidden" />
          </button>

          {openMenu === 'templates' && (
            <div className="absolute right-0 top-10 w-72 rounded-lg bg-[#16191E] border border-[#2C313A] shadow-2xl py-1.5 z-50">
              <div className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-[#7D8590] border-b border-[#22262C]">
                Pipeline Templates
              </div>
              {PIPELINE_TEMPLATES.map((tpl) => (
                <button
                  key={tpl.id}
                  type="button"
                  onClick={() => {
                    onSelectTemplate(tpl);
                    setOpenMenu(null);
                  }}
                  className="w-full text-left px-3 py-2 hover:bg-[#20252D] transition-colors flex flex-col gap-0.5 cursor-pointer"
                >
                  <span className="text-[12px] font-medium text-white">{tpl.name}</span>
                  <span className="text-[11px] text-[#8B949E] leading-snug">{tpl.description}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Load Button */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-[6px] bg-[#181B20] hover:bg-[#22262E] border border-[#2A2F38] text-[#E6EDF3] text-[13px] font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <span>Load</span>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,application/json"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onLoadJsonFile(file);
            e.target.value = '';
          }}
        />

        {/* Export Dropdown */}
        <div className="relative">
          <button
            type="button"
            onClick={() => setOpenMenu(openMenu === 'export' ? null : 'export')}
            className="flex items-center gap-1 px-2.5 sm:px-3 py-1.5 rounded-[6px] bg-[#181B20] hover:bg-[#22262E] border border-[#2A2F38] text-[#E6EDF3] text-[13px] font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <span>Export</span>
            <ChevronDown className="w-3.5 h-3.5 text-[#8B949E]" />
          </button>

          {openMenu === 'export' && (
            <div className="absolute right-0 top-10 w-52 rounded-lg bg-[#16191E] border border-[#2C313A] shadow-2xl py-1 z-50">
              <button
                type="button"
                onClick={() => {
                  onExportPython();
                  setOpenMenu(null);
                }}
                className="w-full px-3 py-2 text-left text-[12px] text-[#E6EDF3] hover:bg-[#20252D] flex items-center gap-2 cursor-pointer"
              >
                <FileCode2 className="w-3.5 h-3.5 text-[#7AB89E]" />
                <span>Export Python (.py)</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  onExportJson();
                  setOpenMenu(null);
                }}
                className="w-full px-3 py-2 text-left text-[12px] text-[#E6EDF3] hover:bg-[#20252D] flex items-center gap-2 cursor-pointer"
              >
                <FolderOpen className="w-3.5 h-3.5 text-[#38BDF8]" />
                <span>Export Pipeline (.json)</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  onExportImage();
                  setOpenMenu(null);
                }}
                className="w-full px-3 py-2 text-left text-[12px] text-[#E6EDF3] hover:bg-[#20252D] flex items-center gap-2 cursor-pointer"
              >
                <ImageIcon className="w-3.5 h-3.5 text-[#F58220]" />
                <span>Download Output (.png)</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  fileInputRef.current?.click();
                  setOpenMenu(null);
                }}
                className="sm:hidden w-full px-3 py-2 text-left text-[12px] text-[#E6EDF3] hover:bg-[#20252D] flex items-center gap-2 cursor-pointer border-t border-[#22262C]"
              >
                <Download className="w-3.5 h-3.5 text-[#9CA3AF]" />
                <span>Load Pipeline JSON</span>
              </button>
            </div>
          )}
        </div>

        {/* Mobile/Tablet Right Inspector Toggle */}
        <button
          type="button"
          onClick={onToggleRightDrawer}
          title="Toggle Inspector Panel"
          className={`lg:hidden flex items-center justify-center w-8 h-8 rounded-[6px] border transition-colors shrink-0 ${
            rightDrawerOpen
              ? 'bg-[#222730] border-[#3A414D] text-white'
              : 'bg-[#171A1F] border-[#262A31] text-[#9CA3AF] hover:text-white'
          }`}
        >
          {rightDrawerOpen ? <PanelRight className="w-4 h-4" /> : <SlidersHorizontal className="w-4 h-4" />}
        </button>
      </div>
    </header>
  );
};
