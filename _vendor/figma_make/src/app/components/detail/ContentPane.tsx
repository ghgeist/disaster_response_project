import { SignalItem } from "@/app/data";
import { Badge, cn } from "@/app/components/ui/common";
import { Calendar, Globe, User, Share2, Languages, Sparkles } from "lucide-react";
import { format } from "date-fns";
import { useState } from "react";

interface ContentPaneProps {
  item: SignalItem;
}

export const ContentPane = ({ item }: ContentPaneProps) => {
  const [viewMode, setViewMode] = useState<"translated" | "original">("translated");

  const highlightText = (text: string, keywords: string[]) => {
    if (!keywords || keywords.length === 0) return text;
    
    const parts = text.split(new RegExp(`(${keywords.join("|")})`, "gi"));
    return parts.map((part, i) => 
      keywords.some(k => k.toLowerCase() === part.toLowerCase()) ? (
        <span key={i} className="bg-yellow-100 text-yellow-800 font-semibold px-0.5 rounded border-b-2 border-yellow-300">
          {part}
        </span>
      ) : (
        part
      )
    );
  };

  return (
    <div className="flex-1 flex flex-col h-auto lg:h-full bg-white lg:overflow-y-auto">
      {/* Header Metadata */}
      <div className="p-6 border-b border-slate-100 pb-4">
        <div className="flex items-center gap-3 mb-4">
          <Badge 
            variant={item.riskLevel === 'critical' ? 'critical' : item.riskLevel === 'high' ? 'danger' : 'neutral'} 
            className="text-sm px-2.5 py-0.5"
          >
            {item.riskLevel.toUpperCase()} RISK
          </Badge>
          <span className="text-sm text-slate-400 font-mono">{item.id}</span>
        </div>

        <div className="flex flex-wrap gap-4 text-xs text-slate-500 font-medium">
           <div className="flex items-center gap-1.5">
             <Calendar className="w-3.5 h-3.5 text-slate-400" />
             {format(item.timestamp, "MMM dd, yyyy • HH:mm:ss z")}
           </div>
           <div className="flex items-center gap-1.5">
             <Globe className="w-3.5 h-3.5 text-slate-400" />
             <span className="uppercase">{item.language}</span>
           </div>
           <div className="flex items-center gap-1.5">
             <User className="w-3.5 h-3.5 text-slate-400" />
             {item.authorHandle}
           </div>
           <div className="flex items-center gap-1.5">
             <Share2 className="w-3.5 h-3.5 text-slate-400" />
             {item.source}
           </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        <div className="flex justify-between items-center mb-4">
           <h2 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-2">
             <Languages className="w-3 h-3" /> Signal Content
           </h2>
           
           {/* Tabbed View Control */}
           {item.translatedContent && (
             <div className="flex bg-slate-100 p-0.5 rounded-lg border border-slate-200">
               <button
                 onClick={() => setViewMode("translated")}
                 className={cn(
                   "px-3 py-1 text-xs font-medium rounded-md transition-all",
                   viewMode === "translated" 
                     ? "bg-white text-slate-900 shadow-sm border border-slate-200/50" 
                     : "text-slate-500 hover:text-slate-700"
                 )}
               >
                 English (Translated)
               </button>
               <button
                 onClick={() => setViewMode("original")}
                 className={cn(
                   "px-3 py-1 text-xs font-medium rounded-md transition-all",
                   viewMode === "original" 
                     ? "bg-white text-slate-900 shadow-sm border border-slate-200/50" 
                     : "text-slate-500 hover:text-slate-700"
                 )}
               >
                 Original ({item.language.toUpperCase()})
               </button>
             </div>
           )}
        </div>

        <div className="p-6 bg-slate-50 border border-slate-200 rounded-lg shadow-sm min-h-[120px]">
          <p className="text-lg leading-relaxed text-slate-800 font-serif">
            {viewMode === "translated" && item.translatedContent 
              ? highlightText(item.translatedContent, item.keywords)
              : highlightText(item.content, item.keywords)
            }
          </p>
        </div>

        {/* Why this category? */}
        <div className="mt-6">
          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Sparkles className="w-3 h-3 text-blue-500" /> Extraction Logic
          </h4>
          <div className="bg-blue-50/50 border border-blue-100 rounded p-3">
            <p className="text-xs text-blue-800 leading-normal">
              <span className="font-semibold">Key triggers: </span>
              The model identified <span className="font-mono bg-blue-100 px-1 rounded text-blue-900 mx-0.5">"{item.keywords[0]}"</span> 
              and <span className="font-mono bg-blue-100 px-1 rounded text-blue-900 mx-0.5">"{item.keywords[1]}"</span> 
              as high-confidence indicators for the primary classification.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
