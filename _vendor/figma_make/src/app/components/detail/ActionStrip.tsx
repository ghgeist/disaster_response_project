import { Archive, CheckCircle, Edit3, Flag, MessageSquarePlus, AlertOctagon, XCircle } from "lucide-react";
import { useState } from "react";
import { format } from "date-fns";

interface ActionStripProps {
  onAction?: (action: string) => void;
}

export const ActionStrip = ({ onAction }: ActionStripProps) => {
  const [actions, setActions] = useState<{ type: string; time: Date }[]>([]);

  const handleAction = (type: string) => {
    setActions(prev => [{ type, time: new Date() }, ...prev]);
    if (onAction) onAction(type);
  };

  return (
    <div className="border-t border-slate-200 bg-white p-4">
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        <div className="flex items-center gap-3 w-full lg:w-auto">
          {/* Primary Action */}
          <button 
            onClick={() => handleAction("Escalated")}
            className="flex-1 lg:flex-none flex items-center justify-center gap-2 px-4 py-2 bg-slate-900 border border-transparent rounded text-sm font-medium text-white hover:bg-slate-800 transition-colors shadow-sm"
          >
            <Flag className="w-4 h-4" />
            Escalate Signal
          </button>

          {/* Secondary Action */}
          <button 
            onClick={() => handleAction("Marked Irrelevant")}
            className="flex-1 lg:flex-none flex items-center justify-center gap-2 px-4 py-2 bg-white border border-slate-300 rounded text-sm font-medium text-slate-700 hover:bg-slate-50 hover:text-slate-900 transition-colors"
          >
            <Archive className="w-4 h-4" />
            Mark Irrelevant
          </button>
          
          <div className="h-6 w-px bg-slate-200 mx-1 hidden lg:block" />

          {/* Tertiary Action */}
          <button 
            onClick={() => handleAction("Annotating")}
            className="flex-none flex items-center gap-2 px-3 py-2 bg-transparent text-sm font-medium text-slate-500 hover:text-blue-600 transition-colors"
          >
            <Edit3 className="w-4 h-4" />
            Annotate
          </button>
        </div>

        {/* Mini Action Log */}
        <div className="hidden lg:flex items-center gap-4 text-xs text-slate-400">
           <span className="uppercase tracking-wider font-semibold text-[10px]">Operator Log</span>
           <div className="flex gap-3">
             {actions.length === 0 && <span className="opacity-50">Ready for triage...</span>}
             {actions.slice(0, 2).map((action, i) => (
               <div key={i} className="flex items-center gap-1.5 bg-slate-50 px-2 py-1 rounded text-slate-600 animate-in fade-in slide-in-from-right-4 duration-300 border border-slate-100">
                 <CheckCircle className="w-3 h-3 text-emerald-500" />
                 <span>{action.type}</span>
                 <span className="opacity-50 text-[10px] font-mono">{format(action.time, "HH:mm")}</span>
               </div>
             ))}
           </div>
        </div>
      </div>
    </div>
  );
};
