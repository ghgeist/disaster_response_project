import { SYSTEM_METRICS } from "@/app/data";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Activity, BarChart3, TrendingUp, AlertCircle } from "lucide-react";

export const MetricsPanel = () => {
  return (
    <div className="h-full flex flex-col bg-slate-50 overflow-y-auto">
      {/* Header with Simulated Badge */}
      <div className="p-4 flex justify-end">
        <div className="inline-flex items-center gap-1.5 px-2 py-1 bg-slate-100 border border-slate-200 rounded text-[10px] font-medium text-slate-500">
          <AlertCircle className="w-3 h-3 text-slate-400" />
          METRICS SIMULATED
        </div>
      </div>

      <div className="px-6 pb-6 space-y-6">
        {/* KPI Cards */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm">
            <div className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-1">Vol Today</div>
            <div className="text-2xl font-bold text-slate-900 flex items-baseline gap-2">
              {SYSTEM_METRICS.volumeToday.toLocaleString()}
              <span className="text-[10px] text-emerald-600 font-medium bg-emerald-50 px-1.5 rounded flex items-center">
                <TrendingUp className="w-2.5 h-2.5 mr-0.5" /> +5%
              </span>
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm">
            <div className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-1">Flagged</div>
            <div className="text-2xl font-bold text-slate-900 flex items-baseline gap-2">
              <span className="text-red-600">{SYSTEM_METRICS.flaggedRate}%</span>
              <span className="text-xs text-slate-400 font-normal">of total</span>
            </div>
          </div>
        </div>

        {/* Chart */}
        <div className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm">
          <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-4 flex items-center gap-2">
            <Activity className="w-3 h-3 text-slate-400" /> Flagged Signals (6H)
          </h3>
          <div className="h-[150px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={SYSTEM_METRICS.flaggedHistory}>
                <defs>
                  <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.1}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis 
                  dataKey="time" 
                  axisLine={false} 
                  tickLine={false} 
                  tick={{fontSize: 10, fill: '#94a3b8'}} 
                  interval="preserveStartEnd"
                />
                <YAxis hide domain={['dataMin - 10', 'dataMax + 10']} />
                <Tooltip 
                  contentStyle={{background: '#1e293b', border: 'none', borderRadius: '4px', fontSize: '10px', color: 'white'}}
                  itemStyle={{color: 'white'}}
                />
                <Area 
                  type="monotone" 
                  dataKey="count" 
                  stroke="#ef4444" 
                  strokeWidth={2}
                  fillOpacity={1} 
                  fill="url(#colorCount)" 
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Categories */}
        <div className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm">
          <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-4 flex items-center gap-2">
            <BarChart3 className="w-3 h-3 text-slate-400" /> Top Categories
          </h3>
          <div className="space-y-3">
            {SYSTEM_METRICS.topCategories.map((cat, i) => (
              <div key={cat.name} className="flex items-center gap-3">
                <div className="w-6 text-[10px] font-mono text-slate-400 text-right">0{i+1}</div>
                <div className="flex-1">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="font-medium text-slate-700">{cat.name}</span>
                    <span className="text-slate-500">{cat.count}</span>
                  </div>
                  <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-slate-800 rounded-full" 
                      style={{ width: `${(cat.count / SYSTEM_METRICS.topCategories[0].count) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="text-[10px] text-slate-400 text-center pt-4 border-t border-slate-200 border-dashed">
          System Status: <span className="text-emerald-600 font-medium">Operational</span> • Last sync: 12s ago
        </div>
      </div>
    </div>
  );
};
