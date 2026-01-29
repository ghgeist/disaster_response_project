import { Clock, Copy, GitBranch, TrendingUp } from "lucide-react";

export const ContextSidebar = () => {
  return (
    <div className="h-auto lg:h-full bg-slate-50 border-l border-slate-200 w-full lg:w-72 flex flex-col lg:overflow-y-auto">
      <div className="p-4 border-b border-slate-200">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Decision Context</h3>
      </div>

      <div className="p-4 space-y-6">
        {/* Similar Items */}
        <section>
          <h4 className="text-xs font-semibold text-slate-700 mb-3 flex items-center gap-1.5">
            <Copy className="w-3 h-3 text-slate-400" /> Similar Items
          </h4>
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="p-2 bg-white border border-slate-200 rounded text-xs hover:border-blue-300 cursor-pointer transition-colors">
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>SIG-10{80+i}</span>
                  <span>9{i}% match</span>
                </div>
                <div className="text-slate-600 line-clamp-2">
                  Previous report of disturbance in sector {i + 4}...
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Temporal Neighbors */}
        <section>
          <h4 className="text-xs font-semibold text-slate-700 mb-3 flex items-center gap-1.5">
            <Clock className="w-3 h-3 text-slate-400" /> Temporal Neighbors
          </h4>
          <div className="bg-white border border-slate-200 rounded p-3">
             <div className="text-xs text-slate-500 mb-2">Signals in last 30m</div>
             <div className="flex items-end gap-1 h-12 w-full">
                {[4, 7, 3, 8, 12, 5, 9, 15, 6, 4].map((h, i) => (
                  <div key={i} className="flex-1 bg-blue-100 rounded-sm hover:bg-blue-300 transition-colors" style={{ height: `${h * 5}%` }} title={`${h} signals`} />
                ))}
             </div>
          </div>
        </section>

        {/* Rarity */}
        <section>
          <h4 className="text-xs font-semibold text-slate-700 mb-3 flex items-center gap-1.5">
             <TrendingUp className="w-3 h-3 text-slate-400" /> Rarity Baseline
          </h4>
          <div className="bg-white border border-slate-200 rounded p-3 flex items-center justify-between">
            <div className="text-xs text-slate-500">Anomaly Score</div>
            <div className="text-lg font-bold text-purple-600">92/100</div>
          </div>
          <p className="text-[10px] text-slate-400 mt-2">
            This signal pattern is highly unusual for this time of day (top 8th percentile).
          </p>
        </section>
      </div>
    </div>
  );
};
