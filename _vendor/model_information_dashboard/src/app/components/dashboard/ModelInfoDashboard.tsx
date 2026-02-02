import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';
import { MODEL_METRICS, CATEGORIES, SYSTEM_BOUNDARIES } from "@/app/data/modelData";
import { Activity, AlertTriangle, CheckCircle, XCircle, BarChart3, ShieldAlert, Search, Bell, Settings, User, Zap, Menu } from 'lucide-react';
import { cn } from "@/app/components/ui/utils";
import exampleImage from 'figma:asset/fbc113be3c107da77627f09b70f5f3c9dfcfad28.png';

// --- Styled Components for "Storm Signal" Look ---

const StormHeader = () => (
  <header className="h-16 bg-white border-b border-gray-200 px-4 flex items-center justify-between sticky top-0 z-50 shadow-sm">
    <div className="flex items-center gap-4">
      <button className="p-2 hover:bg-gray-100 rounded-md text-gray-500">
        <Menu className="h-5 w-5" />
      </button>
      <div className="flex items-center gap-2">
        <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold">
          <Zap className="h-5 w-5 fill-current" />
        </div>
        <span className="text-lg font-bold tracking-tight text-gray-900">STORM SIGNAL</span>
      </div>
    </div>
    
    <div className="flex items-center gap-6">
      <div className="hidden md:flex items-center gap-2 bg-emerald-50 px-3 py-1.5 rounded-full border border-emerald-100">
        <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
        <span className="text-xs font-bold text-emerald-700 uppercase tracking-wide">System: Operational</span>
      </div>
      
      <div className="flex items-center gap-3 text-gray-400">
        <button className="p-2 hover:bg-gray-100 rounded-full transition-colors"><Settings className="h-5 w-5" /></button>
        <button className="p-2 hover:bg-gray-100 rounded-full transition-colors relative">
           <Bell className="h-5 w-5" />
           <span className="absolute top-2 right-2 h-2 w-2 bg-red-500 rounded-full border-2 border-white"></span>
        </button>
      </div>
      
      <div className="flex items-center gap-3 border-l border-gray-200 pl-6">
        <div className="text-right hidden md:block">
          <div className="text-sm font-bold text-gray-900 leading-none">Operator_7</div>
          <div className="text-xs text-gray-500 mt-1">Level 3 Clearance</div>
        </div>
        <div className="h-10 w-10 bg-gray-100 rounded-full flex items-center justify-center text-gray-500 border border-gray-200">
          <User className="h-5 w-5" />
        </div>
      </div>
    </div>
  </header>
);

const MetricCard = ({ title, value, subtext, trend, trendUp }: { title: string, value: string, subtext?: string, trend?: string, trendUp?: boolean }) => (
  <Card className="shadow-sm border-gray-200 rounded-xl overflow-hidden bg-white">
    <CardContent className="p-6">
      <div className="flex flex-col gap-1">
        <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">{title}</h3>
        <div className="flex items-baseline gap-3 mt-1">
          <span className="text-3xl font-bold text-gray-900 tracking-tight">{value}</span>
          {trend && (
             <span className={cn("text-xs font-bold px-1.5 py-0.5 rounded", trendUp ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700")}>
               {trend}
             </span>
          )}
        </div>
        {subtext && <p className="text-sm text-gray-500 font-medium mt-1">{subtext}</p>}
      </div>
    </CardContent>
  </Card>
);

const HealthDial = ({ value, label, color }: { value: number; label: string; color: string }) => {
  const data = [{ name: label, value: value, fill: color }];
  
  return (
    <div className="flex flex-col items-center justify-center">
      <div className="h-[120px] w-[120px] relative">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart 
            cx="50%" 
            cy="50%" 
            innerRadius="70%" 
            outerRadius="100%" 
            barSize={10} 
            data={data} 
            startAngle={90} 
            endAngle={-270}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
            <RadialBar background dataKey="value" cornerRadius={30} />
          </RadialBarChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex items-center justify-center flex-col">
            <span className="text-2xl font-bold text-gray-900">{value}%</span>
        </div>
      </div>
      <span className="text-xs font-bold text-gray-500 uppercase tracking-wider mt-2">{label}</span>
    </div>
  );
};

const HeatmapCell = ({ category }: { category: { name: string; recall: number, type: string } }) => {
  // Style matching the "Live Feed" items or just clean cards
  let colorClass = "bg-red-50 text-red-700 border-red-100";
  let barColor = "bg-red-500";
  
  if (category.recall >= 90) {
    colorClass = "bg-white text-gray-900 border-gray-100 hover:border-emerald-200";
    barColor = "bg-emerald-500";
  } else if (category.recall >= 80) {
    colorClass = "bg-white text-gray-900 border-gray-100 hover:border-amber-200";
    barColor = "bg-amber-500";
  }
  
  return (
    <div 
        className={cn(
            "flex flex-col p-3 border rounded-lg transition-all hover:shadow-md cursor-default group",
            colorClass
        )}
    >
      <div className="flex justify-between items-start mb-2">
         <span className="text-xs font-bold uppercase tracking-tight text-gray-500">{category.type.substring(0,3)}</span>
         <span className={cn("text-xs font-mono font-bold", category.recall < 80 ? "text-red-600" : "text-gray-400")}>{category.recall}%</span>
      </div>
      <span className="font-semibold text-sm truncate mb-2 leading-tight">{category.name}</span>
      <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden">
        <div className={cn("h-full rounded-full transition-all", barColor)} style={{ width: `${category.recall}%` }} />
      </div>
    </div>
  );
};

// Mock data for the chart to match the "FLAGGED SIGNALS (6H)" look
const CHART_DATA = [
  { time: '6h ago', value: 45 },
  { time: '5h ago', value: 90 },
  { time: '4h ago', value: 110 },
  { time: '3h ago', value: 80 },
  { time: '2h ago', value: 160 }, // Peak
  { time: '1h ago', value: 130 },
  { time: 'Now', value: 95 },
];

export function ModelInfoDashboard() {
  return (
    <div className="min-h-screen bg-[#F3F4F6] font-sans text-gray-900">
      <StormHeader />

      <main className="p-6 max-w-[1600px] mx-auto grid grid-cols-12 gap-6">
        
        {/* Left Column: Context / Narrative (Replacing "Live Feed" structure but using same width) */}
        <section className="col-span-12 lg:col-span-3 flex flex-col gap-6">
             <Card className="bg-white border-gray-200 shadow-sm rounded-xl overflow-hidden h-full">
                <CardHeader className="border-b border-gray-100 py-4">
                    <div className="flex items-center justify-between">
                        <CardTitle className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2">
                            <Activity className="h-4 w-4" />
                            Recall Recovery
                        </CardTitle>
                        <Badge variant="outline" className="text-xs font-mono">PHASE-6</Badge>
                    </div>
                </CardHeader>
                <CardContent className="p-0">
                    <div className="p-6 border-b border-gray-100">
                        <p className="text-sm text-gray-600 leading-relaxed mb-4">
                            {MODEL_METRICS.description}
                        </p>
                        
                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between mb-1.5">
                                    <span className="text-xs font-bold text-gray-400 uppercase">Baseline</span>
                                    <span className="text-xs font-mono font-bold text-red-500">0.0%</span>
                                </div>
                                <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                                    <div className="h-full bg-red-400 w-[2%]"></div>
                                </div>
                            </div>
                            
                            <div>
                                <div className="flex justify-between mb-1.5">
                                    <span className="text-xs font-bold text-indigo-500 uppercase">Current Weighted</span>
                                    <span className="text-xs font-mono font-bold text-indigo-600">93.5%</span>
                                </div>
                                <div className="h-2 w-full bg-indigo-50 rounded-full overflow-hidden">
                                    <div className="h-full bg-indigo-600 w-[93.5%] shadow-[0_0_10px_rgba(79,70,229,0.3)]"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    {/* System Boundaries List style */}
                    <div className="bg-gray-50/50 p-6">
                         <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                            <ShieldAlert className="h-4 w-4" />
                            Operational Scope
                         </h4>
                         <div className="space-y-4">
                             <div className="space-y-2">
                                 <span className="text-xs font-bold text-emerald-600 flex items-center gap-1.5">
                                     <CheckCircle className="h-3 w-3" /> IN SCOPE
                                 </span>
                                 <ul className="text-xs text-gray-500 space-y-1 pl-4 border-l-2 border-emerald-100">
                                     {SYSTEM_BOUNDARIES.does.slice(0,3).map((item, i) => (
                                         <li key={i}>{item}</li>
                                     ))}
                                 </ul>
                             </div>
                             <div className="space-y-2">
                                 <span className="text-xs font-bold text-red-600 flex items-center gap-1.5">
                                     <XCircle className="h-3 w-3" /> OUT OF SCOPE
                                 </span>
                                 <ul className="text-xs text-gray-500 space-y-1 pl-4 border-l-2 border-red-100">
                                     {SYSTEM_BOUNDARIES.doesNot.slice(0,3).map((item, i) => (
                                         <li key={i}>{item}</li>
                                     ))}
                                 </ul>
                             </div>
                         </div>
                    </div>
                </CardContent>
             </Card>
        </section>

        {/* Center Column: Metrics & Main Grid */}
        <section className="col-span-12 lg:col-span-9 flex flex-col gap-6">
            
            {/* Top Metrics Row */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <MetricCard 
                    title="F1 Score" 
                    value={`${MODEL_METRICS.f1Score}%`} 
                    trend="+1.2%" 
                    trendUp={true} 
                />
                <MetricCard 
                    title="Precision" 
                    value={`${MODEL_METRICS.precision}%`} 
                    subtext="High confidence thresholding active"
                />
                <Card className="shadow-sm border-gray-200 rounded-xl overflow-hidden bg-white">
                    <CardContent className="p-4 flex items-center justify-around h-full">
                        <HealthDial value={MODEL_METRICS.recall} label="Recall" color="#3b82f6" />
                        <div className="h-12 w-px bg-gray-100 mx-2"></div>
                         {/* Mini Trend Chart mimicking 'Flagged Signals' */}
                        <div className="flex-1 h-[80px]">
                            <div className="text-xs font-bold text-gray-400 uppercase mb-2">Recovery Trend</div>
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={CHART_DATA}>
                                    <defs>
                                        <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.1}/>
                                            <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
                                        </linearGradient>
                                    </defs>
                                    <Area type="monotone" dataKey="value" stroke="#f43f5e" strokeWidth={2} fillOpacity={1} fill="url(#colorValue)" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Main Content: Category Heatmap */}
            <Card className="flex-1 shadow-sm border-gray-200 rounded-xl overflow-hidden bg-white flex flex-col">
                <CardHeader className="border-b border-gray-100 py-4 px-6">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <BarChart3 className="h-5 w-5 text-gray-400" />
                            <CardTitle className="text-sm font-bold text-gray-700 uppercase tracking-wider">Category Performance Matrix</CardTitle>
                        </div>
                        <div className="flex gap-4">
                            <div className="flex items-center gap-2 text-xs font-medium text-gray-500">
                                <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
                                High Performance
                            </div>
                            <div className="flex items-center gap-2 text-xs font-medium text-gray-500">
                                <span className="h-2 w-2 rounded-full bg-amber-500"></span>
                                Degrading
                            </div>
                            <div className="flex items-center gap-2 text-xs font-medium text-gray-500">
                                <span className="h-2 w-2 rounded-full bg-red-500"></span>
                                Critical
                            </div>
                        </div>
                    </div>
                </CardHeader>
                <CardContent className="p-6 bg-gray-50/30 flex-1">
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-6 gap-3">
                        {CATEGORIES.map((cat, idx) => (
                            <HeatmapCell key={idx} category={cat} />
                        ))}
                    </div>
                </CardContent>
            </Card>

        </section>

      </main>
    </div>
  );
}
