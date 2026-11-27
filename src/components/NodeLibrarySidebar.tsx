import React, { useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, Plus, Search, X } from 'lucide-react';
import { CATEGORY_META, CATEGORY_ORDER, NODE_CATALOG } from '../data/nodeCatalog';
import { NodeCatalogItem, NodeCategory } from '../types/pipeline';

interface NodeLibrarySidebarProps {
  onAddNode: (nodeType: string) => void;
  className?: string;
}

export const NodeLibrarySidebar: React.FC<NodeLibrarySidebarProps> = ({
  onAddNode,
  className = '',
}) => {
  const [query, setQuery] = useState('');
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const grouped = useMemo(() => {
    const q = query.trim().toLowerCase();
    const map = new Map<NodeCategory, NodeCatalogItem[]>();
    CATEGORY_ORDER.forEach((c) => map.set(c, []));

    NODE_CATALOG.forEach((item) => {
      if (
        !q ||
        item.title.toLowerCase().includes(q) ||
        item.type.toLowerCase().includes(q) ||
        item.description.toLowerCase().includes(q) ||
        item.category.toLowerCase().includes(q)
      ) {
        map.get(item.category)?.push(item);
      }
    });
    return map;
  }, [query]);

  const toggleCategory = (cat: NodeCategory) => {
    setCollapsed((prev) => ({ ...prev, [cat]: !prev[cat] }));
  };

  return (
    <aside
      className={`w-[252px] bg-[#121417] border-r border-[#22262C] flex flex-col h-full shrink-0 select-none ${className}`}
    >
      {/* Header & Helper Text */}
      <div className="px-4 pt-3.5 pb-3 border-b border-[#1D2127]">
        <h2 className="text-[17px] font-semibold tracking-[-0.01em] text-[#F3F4F6] leading-snug">
          Nodes
        </h2>
        <p className="text-[11px] text-[#7D8590] mt-1 leading-snug">
          Drag onto the canvas, or press + / double-click to drop
        </p>

        {/* Search Input with Emerald/Teal Focus Ring matching mockup */}
        <div className="mt-3 relative flex items-center rounded-[6px] bg-[#0D0F13] border border-[#2D6E5E] focus-within:border-[#389E82] transition-colors">
          <Search className="w-3.5 h-3.5 text-[#7D8590] ml-2.5 shrink-0" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search nodes..."
            className="w-full bg-transparent px-2 py-1.5 text-[12px] text-[#E6EDF3] placeholder:text-[#636B76] outline-none"
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery('')}
              aria-label="Clear search"
              className="mr-2 text-[#7D8590] hover:text-white cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Category Accordions */}
      <div className="flex-1 overflow-y-auto px-3 py-2.5 space-y-3">
        {CATEGORY_ORDER.map((category) => {
          const items = grouped.get(category) || [];
          if (items.length === 0) return null;
          const meta = CATEGORY_META[category];
          const isCollapsed = Boolean(collapsed[category]) && !query.trim();

          return (
            <div key={category} className="space-y-1.5">
              {/* Category Header */}
              <button
                type="button"
                onClick={() => toggleCategory(category)}
                className="w-full flex items-center justify-between px-1 py-1 text-left group cursor-pointer"
              >
                <div className="flex items-center gap-2">
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: meta.dotColor }}
                  />
                  <span className="text-[10.5px] font-semibold tracking-[0.06em] text-[#9CA3AF] group-hover:text-white transition-colors">
                    {meta.label}
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="px-1.5 py-0.5 rounded bg-[#1B1F26] text-[10px] font-mono-tabular text-[#8B949E]">
                    {items.length}
                  </span>
                  {isCollapsed ? (
                    <ChevronRight className="w-3 h-3 text-[#6E7681]" />
                  ) : (
                    <ChevronDown className="w-3 h-3 text-[#6E7681]" />
                  )}
                </div>
              </button>

              {/* Node Cards */}
              {!isCollapsed && (
                <div className="space-y-2">
                  {items.map((item) => (
                    <div
                      key={item.type}
                      draggable
                      onDragStart={(e) => {
                        e.dataTransfer.setData('application/vision-node', item.type);
                        e.dataTransfer.effectAllowed = 'copy';
                      }}
                      onDoubleClick={() => onAddNode(item.type)}
                      className="rounded-[7px] bg-[#16191E] hover:bg-[#1A1E25] border border-[#242932] hover:border-[#343B47] p-2.5 transition-colors cursor-grab active:cursor-grabbing"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[12.5px] font-semibold text-[#F3F4F6] truncate">
                          {item.title}
                        </span>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onAddNode(item.type);
                          }}
                          title={`Add ${item.title}`}
                          className="w-5 h-5 rounded-[4px] bg-[#20252E] hover:bg-[#2B323E] border border-[#2E3440] text-[#C9D1D9] hover:text-white flex items-center justify-center shrink-0 cursor-pointer"
                        >
                          <Plus className="w-3 h-3" />
                        </button>
                      </div>

                      <p className="text-[11px] text-[#8B949E] mt-1 leading-[1.35] line-clamp-2">
                        {item.description}
                      </p>

                      {/* Bottom Port & Attribute Chips */}
                      <div className="flex items-center flex-wrap gap-1.5 mt-2">
                        {item.inputs.length > 0 && (
                          <span className="px-1.5 py-0.5 rounded-[4px] bg-[#1A2922] text-[#7FD19E] text-[10px] font-medium font-mono-tabular">
                            {item.inputs.length} in
                          </span>
                        )}
                        <span className="px-1.5 py-0.5 rounded-[4px] bg-[#2B1F17] text-[#E59F67] text-[10px] font-medium font-mono-tabular">
                          {item.outputs.length} out
                        </span>
                        {item.isRandom && (
                          <span className="px-1.5 py-0.5 rounded-[4px] bg-[#292317] text-[#D8B26E] text-[10px] font-medium">
                            random
                          </span>
                        )}
                        {item.category === 'osd' && (
                          <span className="px-1.5 py-0.5 rounded-[4px] bg-[#2D1F14] text-[#F58220] text-[10px] font-medium">
                            drag &amp; drop
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
};
