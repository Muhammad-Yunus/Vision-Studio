import React from 'react';
import { Check, Download, Grid, X } from 'lucide-react';
import { BatchSampleResult } from '../types/pipeline';

interface BatchSamplesModalProps {
  open: boolean;
  targetTitle: string;
  samples: BatchSampleResult[];
  activeSeed: number;
  isLoading: boolean;
  onClose: () => void;
  onAdoptSeed: (seed: number) => void;
}

export const BatchSamplesModal: React.FC<BatchSamplesModalProps> = ({
  open,
  targetTitle,
  samples,
  activeSeed,
  isLoading,
  onClose,
  onAdoptSeed,
}) => {
  if (!open) return null;

  const handleDownloadSample = (item: BatchSampleResult) => {
    const a = document.createElement('a');
    a.href = item.previewDataUrl;
    a.download = `vision_studio_seed_${item.seed}.png`;
    a.click();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-[2px] flex items-center justify-center p-3 sm:p-5">
      <div className="w-full max-w-4xl max-h-[88vh] bg-[#121417] border border-[#272C35] rounded-xl shadow-2xl flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#22262C] bg-[#15181D]">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-md bg-[#F58220]/15 border border-[#F58220]/40 flex items-center justify-center text-[#F58220]">
              <Grid className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-[14px] font-semibold text-white">
                Batch Seed Sweep — {targetTitle}
              </h2>
              <p className="text-[11px] text-[#7D8590]">
                {samples.length} seeded variations executed across OpenCV WASM. Click any card to adopt its Seed.
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

        <div className="flex-1 overflow-y-auto p-4 bg-[#0B0D10]">
          {isLoading ? (
            <div className="py-16 text-center text-[13px] text-[#9CA3AF]">
              Generating multi-seed batch previews...
            </div>
          ) : samples.length === 0 ? (
            <div className="py-16 text-center text-[13px] text-[#7D8590]">
              No batch samples generated yet. Set Samples &gt; 1 in the top bar and click Run.
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {samples.map((item) => {
                const isCurrent = item.seed === activeSeed;
                return (
                  <div
                    key={item.seed}
                    className={`rounded-lg border overflow-hidden bg-[#14171D] flex flex-col transition-colors ${
                      isCurrent ? 'border-[#389E82] ring-1 ring-[#389E82]/40' : 'border-[#262B34] hover:border-[#3A414E]'
                    }`}
                  >
                    <div className="h-36 bg-[#080A0D] flex items-center justify-center p-1.5 relative">
                      <img
                        src={item.previewDataUrl}
                        alt={`Seed ${item.seed}`}
                        className="max-w-full max-h-full object-contain"
                      />
                      <span className="absolute top-1.5 right-1.5 px-1.5 py-0.5 rounded bg-[#0B0D10]/85 border border-[#242933] text-[10px] font-mono-tabular text-[#4ADE80]">
                        {item.durationMs.toFixed(0)}ms
                      </span>
                    </div>
                    <div className="px-2.5 py-2 flex items-center justify-between gap-1.5 border-t border-[#22262C]">
                      <span className="text-[11px] font-mono-tabular text-[#E6EDF3]">
                        Seed {item.seed}
                      </span>
                      <div className="flex items-center gap-1">
                        <button
                          type="button"
                          title="Download PNG"
                          onClick={() => handleDownloadSample(item)}
                          className="p-1 rounded bg-[#1B1F27] hover:bg-[#252A35] text-[#9CA3AF] hover:text-white cursor-pointer"
                        >
                          <Download className="w-3 h-3" />
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            onAdoptSeed(item.seed);
                            onClose();
                          }}
                          className={`px-2 py-0.5 rounded text-[10.5px] font-medium flex items-center gap-1 cursor-pointer ${
                            isCurrent
                              ? 'bg-[#389E82] text-[#0B0D10]'
                              : 'bg-[#F58220] hover:bg-[#FF9233] text-[#0B0D10]'
                          }`}
                        >
                          {isCurrent && <Check className="w-2.5 h-2.5" />}
                          <span>{isCurrent ? 'Active' : 'Use'}</span>
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
