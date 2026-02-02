import { useState, useEffect } from "react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Activity, BarChart3, TrendingUp } from "lucide-react";
import { Skeleton } from "@/app/components/ui/skeleton";

interface MetricsData {
  volumeToday: number;
  flaggedRate: number;
  flaggedHistory: { time: string; count: number }[];
  topCategories: { name: string; count: number }[];
}

const defaultMetrics: MetricsData = {
  volumeToday: 0,
  flaggedRate: 0,
  flaggedHistory: [],
  topCategories: [],
};

interface MetricsPanelProps {
  onCategoryClick?: (category: string) => void;
}

export const MetricsPanel = ({ onCategoryClick }: MetricsPanelProps) => {
  const [metrics, setMetrics] = useState<MetricsData>(defaultMetrics);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch("/api/metrics")
      .then((res) => {
        if (!res.ok) throw new Error(`Metrics ${res.status}`);
        return res.json();
      })
      .then((data: { volToday?: number; flaggedRate?: number; trendData?: { time: string; count: number }[]; topCategories?: { name: string; count: number }[] }) => {
        setMetrics({
          volumeToday: data.volToday ?? 0,
          flaggedRate: data.flaggedRate ?? 0,
          flaggedHistory: Array.isArray(data.trendData) ? data.trendData : [],
          topCategories: Array.isArray(data.topCategories) ? data.topCategories : [],
        });
      })
      .catch((err) => setError(err?.message ?? "Failed to load metrics"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="h-full flex flex-col bg-slate-50 overflow-y-auto" aria-busy="true" aria-live="polite">
        <span className="sr-only">Loading metrics…</span>
        <div className="pt-4 px-6 pb-6 space-y-6">
          <div className="grid grid-cols-2 gap-4">
            {Array.from({ length: 2 }).map((_, index) => (
              <div key={`metric-skeleton-${index}`} className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm space-y-3">
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-7 w-24" />
              </div>
            ))}
          </div>

          <div className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm space-y-3">
            <Skeleton className="h-3 w-40" />
            <Skeleton className="h-[150px] w-full" />
          </div>

          <div className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm space-y-4">
            <Skeleton className="h-3 w-32" />
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, index) => (
                <div key={`category-skeleton-${index}`} className="flex items-center gap-3">
                  <Skeleton className="h-3 w-6" />
                  <div className="flex-1 space-y-2">
                    <Skeleton className="h-3 w-32" />
                    <Skeleton className="h-2 w-full" />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="text-[10px] text-slate-400 text-center pt-4 border-t border-slate-200 border-dashed">
            <Skeleton className="h-3 w-40 mx-auto" />
          </div>
        </div>
      </div>
    );
  }
  if (error) {
    return (
      <div className="h-full flex flex-col bg-slate-50 overflow-y-auto items-center justify-center text-red-600 text-sm px-4 text-center">
        {error}
      </div>
    );
  }

  // Calculate global max if list is not empty
  const maxCatCount = metrics.topCategories.length > 0
    ? Math.max(...metrics.topCategories.map(c => c.count), 1)
    : 1;

  return (
    <div className="h-full flex flex-col bg-slate-50 overflow-y-auto">
      <div className="pt-4 px-6 pb-6 space-y-6">
        {/* KPI Cards */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm">
            <div className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-1">Vol Today</div>
            <div className="text-2xl font-bold text-slate-900 flex items-baseline gap-2">
              {metrics.volumeToday.toLocaleString()}
              <span className="text-[10px] text-emerald-600 font-medium bg-emerald-50 px-1.5 rounded flex items-center">
                <TrendingUp className="w-2.5 h-2.5 mr-0.5" /> +5%
              </span>
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm">
            <div className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-1">Flagged</div>
            <div className="text-2xl font-bold text-slate-900 flex items-baseline gap-2">
              <span className="text-red-600">{metrics.flaggedRate}%</span>
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
              <AreaChart data={metrics.flaggedHistory}>
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
                <YAxis 
                  width={30}
                  tick={{fontSize: 10, fill: '#94a3b8'}}
                  axisLine={false}
                  tickLine={false}
                />
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
            {metrics.topCategories.map((cat, i) => (
              <div 
                key={cat.name} 
                className="flex items-center gap-3 cursor-pointer hover:bg-slate-50 rounded p-1 -mx-1 transition-colors"
                onClick={() => onCategoryClick?.(cat.name)}
              >
                <div className="w-6 text-[10px] font-mono text-slate-400 text-right">0{i+1}</div>
                <div className="flex-1">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="font-medium text-slate-700">{cat.name}</span>
                    <span className="text-slate-500">{cat.count}</span>
                  </div>
                  <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-slate-800 rounded-full transition-all duration-500" 
                      style={{ width: `${(cat.count / maxCatCount) * 100}%` }}
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
