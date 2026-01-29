import { useState } from "react";
import { format } from "date-fns";
import { SignalItem, CATEGORY_GROUPS } from "@/app/data";
import { Badge, cn } from "@/app/components/ui/common";
import { Globe, Radio, MessageSquare, Twitter, Newspaper, Phone, Filter, X } from "lucide-react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { Checkbox } from "@/app/components/ui/checkbox";

interface FeedPanelProps {
  signals: SignalItem[];
  selectedFilters: string[];
  onToggleFilter: (category: string) => void;
  onClearFilters: () => void;
}

const SourceIcon = ({ source }: { source: string }) => {
  const s = source.toLowerCase();
  if (s.includes("twitter") || s.includes("facebook")) return <Twitter className="w-3 h-3" />;
  if (s.includes("news")) return <Newspaper className="w-3 h-3" />;
  if (s.includes("direct") || s.includes("sms")) return <Phone className="w-3 h-3" />;
  return <MessageSquare className="w-3 h-3" />;
};

export const FeedPanel = ({ signals, selectedFilters, onToggleFilter, onClearFilters }: FeedPanelProps) => {
  const [isFiltersOpen, setIsFiltersOpen] = useState(false);

  return (
    <div className="h-full flex flex-col bg-slate-50">
      {/* Header */}
      <div className="p-3 border-b border-slate-200 bg-white flex justify-between items-center sticky top-0 z-20">
        <h2 className="text-sm font-bold text-slate-800 flex items-center gap-2 uppercase tracking-wide">
          <Radio className="w-4 h-4 text-slate-500" />
          Live Feed 
          <span className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full text-xs font-mono">{signals.length}</span>
        </h2>
        <button 
          onClick={() => setIsFiltersOpen(!isFiltersOpen)}
          className={cn(
            "p-1.5 rounded transition-colors border",
            isFiltersOpen || selectedFilters.length > 0 
              ? "bg-slate-100 text-blue-600 border-blue-200" 
              : "text-slate-400 border-transparent hover:bg-slate-50"
          )}
        >
          <Filter className="w-4 h-4" />
        </button>
      </div>
      
      {/* Collapsible Filters */}
      <Collapsible.Root open={isFiltersOpen} onOpenChange={setIsFiltersOpen}>
        <Collapsible.Content className="border-b border-slate-200 bg-slate-50/50 animate-in slide-in-from-top-2 overflow-hidden">
          <div className="p-3">
            <div className="flex justify-between items-center mb-3">
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Active Filters</span>
              {selectedFilters.length > 0 && (
                <button 
                  onClick={onClearFilters}
                  className="text-[10px] text-red-500 hover:text-red-600 font-medium flex items-center gap-1"
                >
                  Clear All <X className="w-3 h-3" />
                </button>
              )}
            </div>
            
            <div className="grid grid-cols-2 gap-x-4 gap-y-6">
              {Object.entries(CATEGORY_GROUPS).map(([groupName, categories]) => (
                <div key={groupName}>
                  <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2 border-b border-slate-200/50 pb-1">
                    {groupName}
                  </h4>
                  <div className="space-y-1.5">
                    {categories.map(cat => (
                      <div key={cat} className="flex items-center gap-2">
                        <Checkbox 
                          id={`filter-${cat}`}
                          checked={selectedFilters.includes(cat)}
                          onCheckedChange={() => onToggleFilter(cat)}
                          className="w-3.5 h-3.5 rounded-sm border-slate-300 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600"
                        />
                        <label 
                          htmlFor={`filter-${cat}`}
                          className="text-xs text-slate-600 cursor-pointer select-none leading-none"
                        >
                          {cat}
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Collapsible.Content>
      </Collapsible.Root>

      {/* Feed List */}
      <div className="flex-1 overflow-y-auto p-0">
        {signals.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-slate-400 text-sm">
            <span>No signals found</span>
            <button onClick={onClearFilters} className="text-blue-500 hover:underline mt-1">Clear filters</button>
          </div>
        ) : (
          signals.map((signal) => {
            const maxConf = Math.max(...signal.classifications.map(c => c.confidence));
            
            return (
              <div 
                key={signal.id}
                className="p-3 border-b border-slate-200 hover:bg-white transition-colors cursor-default bg-slate-50/30 group"
              >
                {/* Top Row: Time, Severity, Confidence */}
                <div className="flex justify-between items-center mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-slate-500 group-hover:text-slate-700">
                      {format(signal.timestamp, "HH:mm:ss")}
                    </span>
                    <Badge 
                      variant={
                        signal.riskLevel === "HIGH" ? "danger" : 
                        signal.riskLevel === "MEDIUM" ? "warning" : "neutral"
                      }
                      className="text-[10px] py-0 px-1.5"
                    >
                      {signal.riskLevel}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-1.5 bg-white px-1.5 py-0.5 rounded border border-slate-200 shadow-sm">
                    <span className={cn("text-[10px] font-bold font-mono", maxConf > 0.8 ? "text-slate-900" : "text-slate-500")}>
                      {(maxConf * 100).toFixed(0)}%
                    </span>
                    <span className="text-[10px] text-slate-400 font-medium uppercase border-l border-slate-200 pl-1.5">
                      {signal.language}
                    </span>
                    {signal.isTranslated && <Globe className="w-2.5 h-2.5 text-blue-400" />}
                  </div>
                </div>

                {/* Message Content */}
                <p className="text-xs text-slate-800 leading-relaxed mb-2 font-medium line-clamp-2">
                  {signal.content.length > 140 ? signal.content.substring(0, 140) + "..." : signal.content}
                </p>

                {/* Tags */}
                <div className="flex flex-wrap gap-1 mb-2">
                  {signal.categories.map(cat => (
                    <span key={cat} className="text-[10px] px-1.5 py-0.5 bg-blue-50/50 text-blue-700 border border-blue-100 rounded">
                      {cat}
                    </span>
                  ))}
                </div>

                {/* Footer: Source + ID */}
                <div className="flex justify-between items-center text-[10px] text-slate-400">
                  <div className="flex items-center gap-1.5">
                    <SourceIcon source={signal.source} />
                    <span className="font-medium text-slate-500">{signal.source}</span>
                  </div>
                  <span className="font-mono opacity-60 group-hover:opacity-100 transition-opacity">{signal.id}</span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
