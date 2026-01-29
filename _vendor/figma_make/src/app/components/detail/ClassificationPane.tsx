import { Classification } from "@/app/data";
import { ConfidenceBar, Badge } from "@/app/components/ui/common";
import { Info, HelpCircle } from "lucide-react";

interface ClassificationPaneProps {
  classifications: Classification[];
  globalThreshold: number;
}

export const ClassificationPane = ({ classifications, globalThreshold }: ClassificationPaneProps) => {
  return (
    <div className="bg-white h-auto lg:h-full flex flex-col border-l border-slate-200 lg:overflow-y-auto">
      <div className="p-4 border-b border-slate-200 bg-slate-50/50">
        <h3 className="text-sm font-semibold text-slate-800 flex items-center justify-between">
          Model Certainty
          <span className="text-[10px] font-mono font-normal text-slate-500 bg-white border border-slate-200 px-2 py-0.5 rounded-full">
            Transformer-v4
          </span>
        </h3>
      </div>
      
      <div className="p-4 space-y-6">
        {classifications.map((cls, idx) => (
          <div key={idx} className="group">
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-sm font-medium text-slate-700">{cls.category}</span>
              <div className="flex items-center gap-2">
                {cls.confidence >= (globalThreshold / 100) && (
                  <Badge variant="neutral" className="py-0 px-1.5 text-[10px] uppercase tracking-wide bg-slate-800 text-white border-slate-900">
                    Match
                  </Badge>
                )}
              </div>
            </div>
            
            <ConfidenceBar 
              confidence={cls.confidence} 
              threshold={globalThreshold / 100} // Use global threshold
              uncertaintyRange={cls.uncertaintyRange}
            />
            
            <div className="mt-1 flex justify-between text-[10px] text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity">
               <span>Uncertainty: ±{((cls.uncertaintyRange[1] - cls.uncertaintyRange[0]) * 50).toFixed(0)}%</span>
               <span className="font-mono">Threshold: {globalThreshold}%</span>
            </div>
          </div>
        ))}

        <div className="mt-8 p-3 bg-blue-50 border border-blue-100 rounded text-xs text-blue-700 flex gap-2">
          <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <p className="leading-snug">
            Gray bands indicate model uncertainty. The vertical line represents the current global recall sensitivity ({globalThreshold}%).
          </p>
        </div>
      </div>
    </div>
  );
};
