import { useState } from "react";
import { Badge } from "@/app/components/ui/common";
import { Send, Sparkles, BarChart2, Ambulance, AlertOctagon, Trash2, Loader2 } from "lucide-react";
import { CRITICAL_CATEGORIES } from "@/app/data";

export interface ClassificationCategory {
  name: string;
  conf: number;
  vol: number;
  threshold?: number;
  meetsThreshold?: boolean;
}

export interface ClassificationResult {
  categories: ClassificationCategory[];
  severity: string;
}

interface ClassificationPanelProps {
  onDispatch?: (message: string, results: ClassificationResult) => void;
}

export const ClassificationPanel = ({ onDispatch }: ClassificationPanelProps) => {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [hasNoCategories, setHasNoCategories] = useState(false);
  const [classifyError, setClassifyError] = useState<string | null>(null);

  const handleClassify = () => {
    if (!inputText.trim()) return;
    setIsClassifying(true);
    setHasNoCategories(false);
    setResult(null);
    setClassifyError(null);

    fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: inputText.trim() }),
    })
      .then((res) => {
        if (!res.ok) throw new Error(res.status === 400 ? "Invalid message" : `Classification ${res.status}`);
        return res.json();
      })
      .then((data: {
        categories?: {
          name: string;
          confidence: number;
          volume: number;
          threshold?: number;
          meetsThreshold?: boolean;
        }[];
        severity?: string;
      }) => {
        const cats = Array.isArray(data?.categories) ? data.categories : [];
        if (cats.length === 0) {
          setHasNoCategories(true);
        } else {
          setResult({
            categories: cats.map((c) => ({
              name: c.name,
              conf: c.confidence ?? 0,
              vol: c.volume ?? 0,
              threshold: c.threshold,
              meetsThreshold: c.meetsThreshold,
            })),
            severity: (data.severity as string) ?? "LOW",
          });
        }
      })
      .catch((err) => setClassifyError(err?.message ?? "Classification failed"))
      .finally(() => setIsClassifying(false));
  };

  const handleDispatchClick = () => {
    if (onDispatch && result) {
      // Dispatch the message
      onDispatch(inputText, result);
      
      // Reset the sandbox
      setInputText("");
      setResult(null);
      setHasNoCategories(false);
      setClassifyError(null);
    }
  };

  // Helper to determine severity display with overrides
  const getSeverityDisplay = (res: ClassificationResult) => {
    const hasCritical = res.categories.some(
      (c: ClassificationCategory) => CRITICAL_CATEGORIES.includes(c.name) && c.conf > 0.4
    );
    
    if (hasCritical) return { label: "CRITICAL SEVERITY", variant: "danger" as const };
    if (res.severity === "HIGH") return { label: `${res.severity} SEVERITY`, variant: "danger" as const };
    if (res.severity === "MEDIUM") return { label: `${res.severity} SEVERITY`, variant: "warning" as const };
    return { label: `${res.severity} SEVERITY`, variant: "neutral" as const };
  };

  const severityDisplay = result ? getSeverityDisplay(result) : null;
  
  // Split categories for display using per-category threshold when available
  const isAboveThreshold = (cat: ClassificationCategory) =>
    cat.meetsThreshold !== undefined ? !!cat.meetsThreshold : cat.conf > 0.2;
  const highConfCats = result?.categories.filter((c: ClassificationCategory) => isAboveThreshold(c)) || [];
  const lowConfCats = result?.categories.filter((c: ClassificationCategory) => !isAboveThreshold(c)) || [];

  return (
    <div className="h-full flex flex-col bg-white relative min-w-[260px] overflow-x-auto overflow-y-hidden scrollbar-thin">
      {/* Header */}
      <div className="p-3 border-b border-slate-200 bg-white flex-shrink-0 sticky top-0 z-10 pb-0">
        <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-2">
          <Sparkles className="w-3 h-3 text-blue-500" /> Classify Message
        </h3>
      </div>

      {/* Input Section - Fixed Top */}
      <div className="flex-shrink-0 px-3 pb-3 pt-2 border-b border-slate-100 bg-white z-0">
        <div className="space-y-2">
          <textarea
            className="w-full text-xs p-3 rounded border border-slate-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none min-h-[80px] resize-none bg-slate-50 text-slate-700 placeholder:text-slate-500"
            placeholder="Paste a raw message to test classification model...
e.g. 'please help me I need food and water'"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !isClassifying) {
                e.preventDefault();
                handleClassify();
              }
            }}
          />
          <button
            onClick={handleClassify}
            disabled={!inputText.trim() || isClassifying}
            className="w-full bg-slate-900 text-white text-xs font-medium py-2 rounded shadow-sm flex items-center justify-center gap-2 hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isClassifying ? (
              <>
                <Loader2 className="w-3 h-3 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Send className="w-3 h-3" /> Run Classification
              </>
            )}
          </button>
          {classifyError && (
            <p className="text-xs text-red-600 mt-1">{classifyError}</p>
          )}
        </div>
      </div>

      {/* Main Scrollable Content - Results Only */}
      <div className="flex-1 overflow-y-auto p-4 pb-24 scrollbar-thin">
        {/* Results */}
        {hasNoCategories && (
          <div className="animate-in fade-in slide-in-from-top-2 duration-300 flex-1 flex flex-col">
            <div className="flex justify-between items-center mb-3">
              <span className="text-[10px] font-bold text-slate-500 uppercase">Detection Results</span>
              <Badge variant="neutral" className="text-[10px] py-0 px-2 opacity-70">
                NO MATCH
              </Badge>
            </div>
            
            <div className="flex flex-col items-center justify-center p-4 border-2 border-dashed border-slate-100 rounded bg-slate-50 text-center mb-3 flex-1 min-h-[100px]">
              <AlertOctagon className="w-6 h-6 text-slate-400 mb-2" />
              <p className="text-xs text-slate-600 font-medium">No emergency categories detected.</p>
              <p className="text-[10px] text-slate-500 mt-1">This message likely does not contain actionable signal.</p>
            </div>
            
            <button 
              onClick={() => {
                setInputText("");
                setHasNoCategories(false);
              }}
              className="w-full bg-slate-100 text-slate-700 text-xs font-medium py-2 rounded shadow-sm flex items-center justify-center gap-2 hover:bg-slate-200 border border-slate-200 transition-all mt-auto"
            >
              <Trash2 className="w-3 h-3" /> Clear & Reset
            </button>
          </div>
        )}

        {result && (
          <div className="animate-in fade-in slide-in-from-top-2 duration-300 flex-1 flex flex-col">
            <div className="flex justify-between items-center mb-2">
              <span className="text-[10px] font-bold text-slate-500 uppercase">Detection Results</span>
              {severityDisplay && (
                <Badge variant={severityDisplay.variant} className="text-[10px] py-0 px-2">
                  {severityDisplay.label}
                </Badge>
              )}
            </div>
            
            <div className="space-y-1.5 mb-3">
              {highConfCats.map((cat, i) => (
                <div key={i} className="bg-white border border-slate-200 p-2 rounded shadow-sm">
                  <div className="flex justify-between items-center mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-slate-700">{cat.name}</span>
                      <span className="text-[10px] text-slate-400 flex items-center gap-0.5 bg-slate-50 px-1.5 py-0.5 rounded">
                        <BarChart2 className="w-2.5 h-2.5" />
                        {cat.vol.toLocaleString()}
                      </span>
                    </div>
                    <span className="text-xs font-mono font-bold text-slate-900">{(cat.conf * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full h-1 bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500 rounded-full" style={{ width: `${cat.conf * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
            
            {lowConfCats.length > 0 && (
              <div className="mt-3 pt-3 border-t border-slate-50">
                <p className="text-[10px] text-slate-400 mb-1.5 uppercase tracking-wide font-medium">Low Confidence / Noise</p>
                <div className="flex flex-wrap gap-1.5">
                  {lowConfCats.map((cat, i) => (
                    <span key={i} className="text-[10px] px-2 py-1 bg-slate-50 text-slate-500 border border-slate-100 rounded flex items-center gap-1.5">
                      {cat.name} <span className="text-slate-300">|</span> <span className="opacity-70 font-mono">{(cat.conf * 100).toFixed(0)}%</span>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Sticky Dispatch Action */}
      {result && (
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-white/95 backdrop-blur border-t border-slate-100 z-20 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
           <button 
             onClick={handleDispatchClick}
             className="w-full bg-blue-600 text-white text-xs font-bold py-3 rounded shadow-sm flex items-center justify-center gap-2 hover:bg-blue-700 transition-all active:scale-[0.98]"
           >
            <Ambulance className="w-4 h-4" /> Dispatch Assistance
          </button>
          <div className="mt-2 text-[10px] text-center text-slate-400">
            Dispatched items appear in Live Feed
          </div>
        </div>
      )}
    </div>
  );
};
