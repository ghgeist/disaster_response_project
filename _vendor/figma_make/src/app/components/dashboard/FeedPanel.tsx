import { useState } from "react";
import { format } from "date-fns";
import { SignalItem, CATEGORY_GROUPS } from "@/app/data";
import { Badge, cn } from "@/app/components/ui/common";
import { Globe, Radio, MessageSquare, Twitter, Newspaper, Phone, Filter, X, EyeOff, Search } from "lucide-react";
import { Checkbox } from "@/app/components/ui/checkbox";
import { Popover, PopoverContent, PopoverTrigger } from "@/app/components/ui/popover";
import { Input } from "@/app/components/ui/input";

interface FeedPanelProps {
  signals: SignalItem[];
  selectedFilters: string[];
  onToggleFilter: (category: string) => void;
  onClearFilters: () => void;
  loading?: boolean;
  error?: string | null;
}

const SourceIcon = ({ source }: { source: string }) => {
  const s = source.toLowerCase();
  if (s.includes("twitter") || s.includes("facebook") || s.includes("x") || s.includes("social")) return <Twitter className="w-3 h-3" />;
  if (s.includes("news")) return <Newspaper className="w-3 h-3" />;
  if (s.includes("direct") || s.includes("sms")) return <Phone className="w-3 h-3" />;
  return <MessageSquare className="w-3 h-3" />;
};

export const FeedPanel = ({ signals, selectedFilters, onToggleFilter, onClearFilters, loading, error }: FeedPanelProps) => {
  const [hideLowConfidence, setHideLowConfidence] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");

  const displaySignals = hideLowConfidence 
    ? signals.filter(s => {
        const confs = s.classifications?.map(c => c.confidence) ?? [];
        const max = confs.length > 0 ? Math.max(...confs) : 0;
        return max >= 0.5;
      })
    : signals;

  // Filter categories based on search
  const getFilteredCategories = () => {
    if (!searchTerm) return CATEGORY_GROUPS;
    const result: Record<string, string[]> = {};
    Object.entries(CATEGORY_GROUPS).forEach(([group, cats]) => {
      const filtered = cats.filter(c => c.toLowerCase().includes(searchTerm.toLowerCase()));
      if (filtered.length > 0) result[group] = filtered;
    });
    return result;
  };

  const filteredCategories = getFilteredCategories();

  return (
    <div className="h-full flex flex-col bg-slate-50">
      {/* Header */}
      <div className="p-3 border-b border-slate-200 bg-white flex justify-between items-center sticky top-0 z-20">
        <h2 className="text-sm font-bold text-slate-800 flex items-center gap-2 uppercase tracking-wide">
          <Radio className="w-4 h-4 text-slate-500" />
          Live Feed 
          <span className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full text-xs font-mono">{displaySignals.length}</span>
        </h2>
        
        <Popover>
          <PopoverTrigger asChild>
            <button 
              className={cn(
                "p-1.5 rounded transition-colors border",
                selectedFilters.length > 0 || hideLowConfidence
                  ? "bg-slate-100 text-blue-600 border-blue-200" 
                  : "text-slate-400 border-transparent hover:bg-slate-50"
              )}
            >
              <Filter className="w-4 h-4" />
            </button>
          </PopoverTrigger>
          <PopoverContent align="end" className="w-[300px] p-0 overflow-hidden" sideOffset={8}>
            <div className="p-3 border-b border-slate-100 bg-slate-50/50">
              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 w-3.5 h-3.5 text-slate-400" />
                <Input
                  placeholder="Filter categories..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="h-9 text-xs pl-8 bg-white"
                />
              </div>
            </div>

            <div className="p-0 max-h-[400px] overflow-y-auto">
              <div className="p-3">
                <div className="flex justify-between items-center mb-3">
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Active Filters</span>
                  {(selectedFilters.length > 0 || hideLowConfidence) && (
                    <button 
                      onClick={() => {
                        onClearFilters();
                        setHideLowConfidence(false);
                      }}
                      className="text-[10px] text-red-500 hover:text-red-600 font-medium flex items-center gap-1"
                    >
                      Clear All <X className="w-3 h-3" />
                    </button>
                  )}
                </div>
                
                <div className="mb-4 pb-4 border-b border-slate-200/50">
                   <div className="flex items-center gap-2">
                    <Checkbox 
                      id="filter-low-conf"
                      checked={hideLowConfidence}
                      onCheckedChange={(checked) => setHideLowConfidence(!!checked)}
                      className="w-3.5 h-3.5 rounded-sm border-slate-300 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600"
                    />
                    <label 
                      htmlFor="filter-low-conf"
                      className="text-xs text-slate-600 cursor-pointer select-none flex items-center gap-1.5"
                    >
                      <EyeOff className="w-3 h-3 text-slate-400" />
                      Hide Low Confidence
                    </label>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-x-4 gap-y-6">
                  {Object.entries(filteredCategories).map(([groupName, categories]) => (
                    <div key={groupName} className="col-span-2 sm:col-span-1">
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
                  {Object.keys(filteredCategories).length === 0 && (
                     <div className="col-span-2 text-center py-4 text-xs text-slate-400">
                        No categories found matching "{searchTerm}"
                     </div>
                  )}
                </div>
              </div>
            </div>
          </PopoverContent>
        </Popover>
      </div>

      {/* Filter Chips Area */}
      {selectedFilters.length > 0 && (
        <div className="px-3 py-2 bg-slate-50 border-b border-slate-200 flex flex-wrap gap-1.5 animate-in slide-in-from-top-1">
          {selectedFilters.map(filter => (
            <Badge key={filter} variant="neutral" className="bg-white border-slate-200 text-slate-600 pl-2 pr-1 py-0.5 h-6 gap-1 flex items-center shadow-sm">
              {filter}
              <button 
                onClick={() => onToggleFilter(filter)} 
                className="hover:bg-slate-100 rounded-full p-0.5 text-slate-400 hover:text-slate-600 transition-colors"
              >
                <X className="w-3 h-3" />
              </button>
            </Badge>
          ))}
          <button onClick={onClearFilters} className="text-[10px] text-blue-600 hover:text-blue-700 hover:underline ml-1 font-medium">
            Clear all
          </button>
        </div>
      )}
      
      {/* Feed List */}
      <div className="flex-1 overflow-y-auto p-0">
        {loading ? (
          <div className="flex flex-col items-center justify-center h-48 text-slate-500 text-sm">
            <span>Loading feed…</span>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-48 text-red-600 text-sm px-4 text-center">
            <span>{error}</span>
          </div>
        ) : displaySignals.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-slate-400 text-sm px-6 text-center">
            <Filter className="w-8 h-8 mb-2 opacity-20" />
            <span className="font-medium text-slate-600">No signals found</span>
            <span className="text-xs mt-1 mb-3 opacity-70">Try adjusting your filters</span>
            <button onClick={() => { onClearFilters(); setHideLowConfidence(false); }} className="text-blue-500 hover:underline text-xs bg-blue-50 px-3 py-1.5 rounded-full border border-blue-100">
              Clear all filters
            </button>
          </div>
        ) : (
          displaySignals.map((signal, index) => {
            const confidences = signal.classifications?.map((c) => c.confidence) ?? [];
            const rawMax = confidences.length > 0 ? Math.max(...confidences) : 0;
            const maxConf = Number.isFinite(rawMax) ? rawMax : 0;
            const isHighRisk = signal.riskLevel === "HIGH";

            return (
              <div 
                key={`${signal.id}-${signal.timestamp.getTime()}-${index}`}
                className={cn(
                  "p-2 border-b border-slate-200 hover:bg-white transition-colors cursor-default group",
                  isHighRisk ? "bg-red-50/40" : "bg-slate-50/30"
                )}
              >
                {/* Compact Header: Time • Source • Priority • Conf */}
                <div className="flex justify-between items-center mb-1.5">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-slate-500 group-hover:text-slate-700">
                      {format(signal.timestamp, "HH:mm:ss")}
                    </span>
                    
                    <div className="flex items-center gap-1 text-[10px] text-slate-500">
                      <SourceIcon source={signal.source} />
                      <span className="font-medium">{signal.source}</span>
                    </div>

                    <Badge 
                      variant={
                        signal.riskLevel === "HIGH" ? "danger" : 
                        signal.riskLevel === "MEDIUM" ? "warning" : "neutral"
                      }
                      className="text-[10px] py-0 px-1.5 h-4"
                    >
                      {signal.riskLevel}
                    </Badge>
                  </div>
                  
                  <div className="flex items-center gap-1.5">
                    <span className={cn("text-[10px] font-bold font-mono", maxConf > 0.8 ? "text-slate-900" : "text-slate-400")}>
                      {(maxConf * 100).toFixed(0)}%
                    </span>
                    <span className="text-[10px] text-slate-400 font-medium uppercase border-l border-slate-200 pl-1.5">
                      {signal.language}
                    </span>
                    {signal.isTranslated && <Globe className="w-2.5 h-2.5 text-blue-400" />}
                  </div>
                </div>

                {/* Message Content */}
                <p className="text-xs text-slate-800 leading-snug mb-1.5 font-medium line-clamp-2">
                  {signal.content.length > 140 ? signal.content.substring(0, 140) + "..." : signal.content}
                </p>

                {/* Tags */}
                <div className="flex flex-wrap gap-1">
                  {signal.categories
                    .filter(cat => cat.toLowerCase() !== "related") // Filter out 'related' tag
                    .map(cat => (
                    <span key={cat} className="text-[10px] px-1.5 py-0 bg-blue-50/50 text-blue-700 border border-blue-100 rounded">
                      {cat}
                    </span>
                  ))}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
