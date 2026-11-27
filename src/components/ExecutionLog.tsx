import React, { useEffect, useRef } from 'react';
import { ChevronDown, ChevronUp, Trash2 } from 'lucide-react';
import { ExecutionLogEntry } from '../types/pipeline';

interface ExecutionLogProps {
  logs: ExecutionLogEntry[];
  nodeCount: number;
  totalMs: number;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onClear: () => void;
}

export const ExecutionLog: React.FC<ExecutionLogProps> = ({
  logs,
  nodeCount,
  totalMs,
  collapsed,
  onToggleCollapse,
  onClear,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current && !collapsed) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, collapsed]);

  return (
    <section className="bg-[#0E1013] border-t border-[#22262C] flex flex-col shrink-0">
      {/* Header Bar */}
      <div className="flex items-center justify-between px-3.5 h-8 bg-[#121418] border-b border-[#1D2127] select-none">
        <button
          type="button"
          onClick={onToggleCollapse}
          className="flex items-center gap-2.5 text-left cursor-pointer"
        >
          <span className="text-[12.5px] font-semibold text-[#E6EDF3]">Execution Log</span>
          <span className="text-[11px] font-mono-tabular text-[#7D8590]">
            {nodeCount} nodes · {totalMs.toFixed(1)}ms
          </span>
          {collapsed ? (
            <ChevronUp className="w-3.5 h-3.5 text-[#7D8590]" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-[#7D8590]" />
          )}
        </button>

        <button
          type="button"
          onClick={onClear}
          title="Clear execution log"
          className="p-1 text-[#6E7681] hover:text-[#C9D1D9] transition-colors cursor-pointer"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Scrollable Log Body */}
      {!collapsed && (
        <div
          ref={scrollRef}
          className="h-[112px] sm:h-[124px] overflow-y-auto px-3.5 py-2 font-mono-tabular text-[11.5px] leading-[1.55] text-[#C9D1D9]"
        >
          {logs.length === 0 ? (
            <div className="text-[#6E7681]">Ready. Click Run to execute pipeline.</div>
          ) : (
            logs.map((entry) => (
              <div
                key={entry.id}
                className={
                  entry.kind === 'error'
                    ? 'text-[#F87171]'
                    : entry.kind === 'warn'
                      ? 'text-[#FBBF24]'
                      : 'text-[#D1D5DB]'
                }
              >
                {entry.message}
              </div>
            ))
          )}
        </div>
      )}
    </section>
  );
};
