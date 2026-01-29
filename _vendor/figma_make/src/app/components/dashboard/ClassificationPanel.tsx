import { useState } from "react";
import { Badge, cn } from "@/app/components/ui/common";
import { getCategoryVolume } from "@/app/data";
import { Send, Sparkles, BarChart2, Ambulance, AlertOctagon, Trash2 } from "lucide-react";

export const ClassificationPanel = () => {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState<{ categories: {name: string, conf: number, vol: number}[], severity: string } | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [hasNoCategories, setHasNoCategories] = useState(false);

  const handleClassify = () => {
    if (!inputText.trim()) return;
    setIsClassifying(true);
    setHasNoCategories(false);
    setResult(null);
    
    // Simulate API call
    setTimeout(() => {
      // Mock result logic based on keywords or random
      const mockCats = [];
      const lower = inputText.toLowerCase();
      
      if (lower.includes("water") || lower.includes("flood")) mockCats.push({name: "Water", conf: 0.92});
      if (lower.includes("hurt") || lower.includes("injured")) mockCats.push({name: "Medical Help", conf: 0.88});
      if (lower.includes("food") || lower.includes("hungry")) mockCats.push({name: "Food", conf: 0.85});
      if (lower.includes("rubble") || lower.includes("trapped")) mockCats.push({name: "Search & Rescue", conf: 0.94});
      if (lower.includes("fire")) mockCats.push({name: "Fire", conf: 0.97});

      // Special case for empty
      if (mockCats.length === 0) {
        setHasNoCategories(true);
      } else {
        // Add volume stats
        const enrichedCats = mockCats.map(c => ({
          ...c,
          vol: getCategoryVolume(c.name)
        }));

        setResult({
          categories: enrichedCats,
          severity: enrichedCats.some(c => ["Medical Help", "Water", "Food", "Search & Rescue", "Fire"].includes(c.name)) ? "HIGH" : "LOW"
        });
      }
      
      setIsClassifying(false);
    }, 800);
  };

  return (
    <div className="h-full flex flex-col bg-white overflow-y-auto">
      {/* Header */}
      <div className="p-3 border-b border-slate-200 bg-white sticky top-0">
        <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-2">
          <Sparkles className="w-3 h-3 text-blue-500" /> Classify Message
        </h3>
      </div>

      <div className="p-4 flex-1 flex flex-col">
        <div className="space-y-3 mb-6">
          <textarea
            className="w-full text-xs p-3 rounded border border-slate-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none min-h-[120px] resize-none bg-slate-50 placeholder:text-slate-400"
            placeholder="Paste a raw message to test classification model...
e.g. 'Flash flood in sector 7, need immediate evacuation support...'"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                handleClassify();
              }
            }}
          />
          <button
            onClick={handleClassify}
            disabled={!inputText.trim() || isClassifying}
            className="w-full bg-slate-900 text-white text-xs font-medium py-2.5 rounded shadow-sm flex items-center justify-center gap-2 hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isClassifying ? "Processing..." : (
              <>
                <Send className="w-3 h-3" /> Run Classification
              </>
            )}
          </button>
        </div>

        {/* Results */}
        {hasNoCategories && (
          <div className="animate-in fade-in slide-in-from-top-2 duration-300 border-t border-slate-100 pt-4 flex-1 flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Detection Results</span>
              <Badge variant="neutral" className="text-[10px] py-0 px-2 opacity-50">
                NO MATCH
              </Badge>
            </div>
            
            <div className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-slate-100 rounded bg-slate-50 text-center mb-4 flex-1 min-h-[120px]">
              <AlertOctagon className="w-8 h-8 text-slate-300 mb-2" />
              <p className="text-xs text-slate-500 font-medium">No emergency categories detected.</p>
              <p className="text-[10px] text-slate-400 mt-1">This message likely does not contain actionable signal.</p>
            </div>
            
            <button className="w-full bg-slate-100 text-slate-600 text-xs font-medium py-2.5 rounded shadow-sm flex items-center justify-center gap-2 hover:bg-slate-200 border border-slate-200 transition-all mt-auto">
              <Trash2 className="w-3 h-3" /> Mark as Irrelevant
            </button>
          </div>
        )}

        {result && (
          <div className="animate-in fade-in slide-in-from-top-2 duration-300 border-t border-slate-100 pt-4 flex-1 flex flex-col">
            <div className="flex justify-between items-center mb-3">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Detection Results</span>
              <Badge variant={result.severity === "HIGH" ? "danger" : "neutral"} className="text-[10px] py-0 px-2">
                {result.severity} SEVERITY
              </Badge>
            </div>
            
            <div className="space-y-2 mb-6">
              {result.categories.map((cat, i) => (
                <div key={i} className="bg-white border border-slate-200 p-2.5 rounded shadow-sm">
                  <div className="flex justify-between items-center mb-1.5">
                    <span className="text-xs font-bold text-slate-700">{cat.name}</span>
                    <span className="text-xs font-mono font-bold text-slate-900">{(cat.conf * 100).toFixed(0)}%</span>
                  </div>
                  
                  {/* Confidence Bar */}
                  <div className="w-full h-1 bg-slate-100 rounded-full overflow-hidden mb-2">
                    <div className="h-full bg-blue-500 rounded-full" style={{ width: `${cat.conf * 100}%` }} />
                  </div>

                  {/* Operational Context Stats */}
                  <div className="flex items-center gap-1.5 text-[10px] text-slate-500 bg-slate-50 px-2 py-1 rounded border border-slate-100">
                    <BarChart2 className="w-3 h-3 text-slate-400" />
                    <span>Volume Today:</span>
                    <span className="font-mono font-medium text-slate-700">{cat.vol.toLocaleString()}</span>
                  </div>
                </div>
              ))}
            </div>
            
            {/* Primary Action */}
            <div className="mt-auto pt-4 border-t border-slate-100">
               <button className="w-full bg-blue-600 text-white text-xs font-bold py-3 rounded shadow-sm flex items-center justify-center gap-2 hover:bg-blue-700 transition-all active:scale-[0.98]">
                <Ambulance className="w-4 h-4" /> Dispatch Assistance
              </button>
              <div className="mt-2 p-2 rounded text-[10px] text-center text-slate-400">
                Action logs will be recorded
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
