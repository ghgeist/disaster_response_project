import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { 
  MODEL_METRICS, 
  CATEGORIES, 
  REGISTRY_FILES, 
  CRITICAL_THRESHOLDS, 
  HIERARCHY_SPECS,
  THRESHOLD_MAP,
  SYSTEM_BOUNDARIES
} from "@/app/data/modelData";
import { 
  FileText, 
  Database, 
  Activity, 
  Layers, 
  GitCommit, 
  Server, 
  Clock, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle,
  FolderOpen,
  FileJson,
  FileSpreadsheet,
  FileCode
} from 'lucide-react';
import { cn } from "@/app/components/ui/utils";

// --- Helpers ---

const getStatus = (categoryName: string, recall: number) => {
  // Check strict threshold first
  const strictThreshold = THRESHOLD_MAP[categoryName];
  if (strictThreshold !== undefined) {
    if (recall < strictThreshold * 100) return 'DEGRADED';
  }

  // Fallback to standard logic
  if (recall >= 90) return 'OPTIMAL';
  if (recall >= 80) return 'MONITOR';
  return 'CRITICAL';
};

const getFileIcon = (type?: string) => {
  switch(type) {
    case 'json': return <FileJson className="h-3 w-3 text-yellow-600" />;
    case 'csv': return <FileSpreadsheet className="h-3 w-3 text-emerald-600" />;
    case 'md': return <FileCode className="h-3 w-3 text-blue-600" />;
    default: return <FileText className="h-3 w-3 text-gray-400" />;
  }
};

// --- Sub-components ---

const MetricCard = ({ label, value }: { label: string; value: string }) => (
  <div className="bg-white border border-gray-200 p-4 flex flex-col justify-between h-24 hover:border-gray-400 transition-colors">
    <span className="text-xs font-bold text-gray-500 uppercase tracking-widest">{label}</span>
    <span className="text-3xl font-mono font-bold text-gray-900 tracking-tighter">{value}</span>
  </div>
);

const CategoryCard = ({ category }: { category: { name: string; recall: number; f1: number } }) => {
  const status = getStatus(category.name, category.recall);
  const strictThreshold = THRESHOLD_MAP[category.name];
  
  let statusColor = "bg-red-50 border-red-200 text-red-700";
  let barColor = "bg-red-500";
  let statusLabel = null;
  
  if (status === 'DEGRADED') {
     statusColor = "bg-red-50 border-red-400 text-red-800 shadow-[inset_0_0_0_1px_rgba(248,113,113,0.4)]";
     barColor = "bg-red-600";
     statusLabel = (
        <span className="absolute top-1 right-1 text-[8px] font-bold bg-red-100 text-red-700 px-1 py-0.5 border border-red-200 uppercase tracking-tight z-10">
            Degraded
        </span>
     );
  } else if (status === 'OPTIMAL') {
    statusColor = "bg-emerald-50 border-emerald-200 text-emerald-700";
    barColor = "bg-emerald-500";
  } else if (status === 'MONITOR') {
    statusColor = "bg-amber-50 border-amber-200 text-amber-700";
    barColor = "bg-amber-500";
  }

  return (
    <div className={cn("relative p-2 border text-xs flex flex-col justify-between h-20 transition-all hover:bg-opacity-80", statusColor)}>
      {statusLabel}
      <div className="flex justify-between items-start pt-1">
        <span className="font-bold truncate max-w-[70%] uppercase tracking-tight" title={category.name}>{category.name}</span>
        <span className="font-mono font-bold">{category.recall}%</span>
      </div>
      <div className="mt-auto space-y-1">
        <div className="flex justify-between items-center text-[10px] opacity-80">
            <span className="font-medium">F1: {category.f1}%</span>
            {strictThreshold && (
                 <span className="font-mono text-[9px] opacity-60">THR: {strictThreshold}</span>
            )}
        </div>
        <div className="h-1 w-full bg-black/5">
            <div className="h-full" style={{ width: `${category.recall}%` }} className={barColor} />
        </div>
      </div>
    </div>
  );
};

const FileItem = ({ file }: { file: typeof REGISTRY_FILES[0] }) => (
  <div className={cn(
    "flex items-center justify-between p-2 text-xs border-b border-gray-100 last:border-0 hover:bg-gray-50 cursor-pointer group transition-colors",
    file.highlight && "bg-blue-50/30"
  )}>
    <div className="flex items-center gap-2">
      {getFileIcon(file.type)}
      <span className={cn("font-mono text-gray-600 group-hover:text-gray-900 transition-colors", file.highlight && "font-bold text-blue-700")}>
        {file.name}
      </span>
    </div>
    <span className="text-gray-400 font-mono text-[10px]">{file.size}</span>
  </div>
);

export function ModelRegistryDashboard() {
  return (
    <div className="min-h-screen bg-gray-50 font-sans text-gray-900 flex">
      
      {/* Left Sidebar: Registry Assets */}
      <aside className="w-80 bg-white border-r border-gray-200 flex flex-col h-screen sticky top-0 overflow-y-auto shrink-0 z-10">
        <div className="p-4 border-b border-gray-200 bg-gray-50/80 backdrop-blur-sm sticky top-0 z-20">
          <div className="flex items-center gap-2 text-sm font-bold text-gray-700 uppercase tracking-wider">
            <FolderOpen className="h-4 w-4" />
            Registry Assets
          </div>
          <div className="text-[10px] font-mono text-gray-500 mt-1 flex items-center gap-1">
             <span className="text-emerald-600">●</span> @model/active/production
          </div>
        </div>

        <div className="flex-1">
          {/* File Manifest */}
          <div className="border-b border-gray-200">
             <div className="px-4 py-2 bg-gray-50 text-[10px] font-bold text-gray-400 uppercase tracking-wider flex justify-between items-center">
                 <span>Manifest</span>
                 <span className="text-emerald-600 font-normal normal-case">Synced</span>
             </div>
             {REGISTRY_FILES.map((file, idx) => (
               <FileItem key={idx} file={file} />
             ))}
          </div>

          {/* Critical Thresholds */}
          <div className="p-4 border-b border-gray-200">
             <div className="flex items-center gap-2 mb-3">
                 <Activity className="h-4 w-4 text-gray-400" />
                 <span className="text-xs font-bold text-gray-700 uppercase tracking-wider">Optimization Rules</span>
             </div>
             <div className="bg-gray-50 border border-gray-200 p-3 space-y-2">
                <div className="flex justify-between items-center text-[10px] text-gray-400 uppercase border-b border-gray-200 pb-1 mb-1">
                    <span>Category</span>
                    <span>Min Score</span>
                </div>
                {CRITICAL_THRESHOLDS.map((t, idx) => (
                    <div key={idx} className="flex justify-between items-center text-xs">
                        <span className="text-gray-600 font-medium truncate max-w-[140px]" title={t.label}>{t.label}</span>
                        <span className={cn("font-mono font-bold", t.value > 0.85 ? "text-red-600" : "text-gray-900")}>
                            {t.value.toFixed(2)}
                        </span>
                    </div>
                ))}
             </div>
             <div className="mt-2 text-[10px] text-gray-400 italic">
                 Values source: optimized_critical_thresholds.json
             </div>
          </div>
          
           {/* Boundaries Sidebar Check */}
           <div className="p-4">
                <div className="flex items-center gap-2 mb-3">
                    <Layers className="h-4 w-4 text-gray-400" />
                    <span className="text-xs font-bold text-gray-700 uppercase tracking-wider">Scope Validation</span>
                </div>
                <div className="space-y-3">
                    <div>
                        <span className="text-[10px] font-bold text-emerald-600 uppercase mb-1 block">In Scope</span>
                        <ul className="text-[10px] text-gray-500 space-y-1 list-disc pl-3">
                            {SYSTEM_BOUNDARIES.does.slice(0,3).map((item, i) => (
                                <li key={i}>{item}</li>
                            ))}
                        </ul>
                    </div>
                     <div>
                        <span className="text-[10px] font-bold text-red-600 uppercase mb-1 block">Out of Scope</span>
                        <ul className="text-[10px] text-gray-500 space-y-1 list-disc pl-3">
                            {SYSTEM_BOUNDARIES.doesNot.slice(0,3).map((item, i) => (
                                <li key={i}>{item}</li>
                            ))}
                        </ul>
                    </div>
                </div>
           </div>

        </div>
        
        {/* Footer */}
        <div className="p-3 border-t border-gray-200 bg-gray-50 text-[10px] font-mono text-gray-400 text-center">
            7f8a9b2c3d... committed by operator_7
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0">
        
        {/* Header */}
        <header className="bg-white border-b border-gray-200 h-16 px-6 flex items-center justify-between shrink-0 sticky top-0 z-30">
            <div className="flex items-center gap-4">
                <div className="h-8 w-8 bg-gray-900 text-white flex items-center justify-center font-bold rounded-none">
                    AI
                </div>
                <div>
                    <h1 className="text-sm font-bold text-gray-900 uppercase tracking-widest flex items-center gap-2">
                        {MODEL_METRICS.id}
                    </h1>
                    <div className="text-[10px] font-mono text-gray-500 flex items-center gap-2">
                        <Badge variant="secondary" className="bg-gray-100 text-gray-600 hover:bg-gray-200 font-mono rounded-none text-[10px] h-4 px-1">v{MODEL_METRICS.version}</Badge>
                        <span>feature/phase-6-polish-testing</span>
                    </div>
                </div>
            </div>
            <div className="flex items-center gap-6">
                <div className="flex flex-col items-end">
                    <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Last Sync</span>
                    <span className="text-xs font-mono text-gray-700 flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {new Date(MODEL_METRICS.lastUpdated).toISOString().replace('T', ' ').substring(0, 19)}
                    </span>
                </div>
                <div className="h-8 w-px bg-gray-200"></div>
                <Badge className="bg-emerald-600 hover:bg-emerald-700 text-white border-0 rounded-none px-3 py-1 text-xs uppercase tracking-widest font-bold shadow-sm">
                    Production
                </Badge>
            </div>
        </header>

        <div className="flex-1 overflow-y-auto p-8 space-y-8">
            
            {/* Top Metrics Row */}
            <section className="grid grid-cols-1 md:grid-cols-3 gap-0 border border-gray-200 bg-white shadow-sm">
                <div className="border-r border-gray-200 last:border-0 p-6 hover:bg-gray-50 transition-colors">
                    <span className="text-xs font-bold text-gray-500 uppercase tracking-widest block mb-2">Global Precision</span>
                    <div className="flex items-baseline gap-2">
                        <span className="text-4xl font-mono font-bold text-gray-900">{MODEL_METRICS.precision}%</span>
                        <span className="text-xs text-emerald-600 font-medium bg-emerald-50 px-1 py-0.5 rounded">+0.4%</span>
                    </div>
                </div>
                <div className="border-r border-gray-200 last:border-0 p-6 hover:bg-gray-50 transition-colors">
                    <span className="text-xs font-bold text-gray-500 uppercase tracking-widest block mb-2">Global Recall</span>
                    <div className="flex items-baseline gap-2">
                        <span className="text-4xl font-mono font-bold text-gray-900">{MODEL_METRICS.recall}%</span>
                        <span className="text-xs text-emerald-600 font-medium bg-emerald-50 px-1 py-0.5 rounded">+12.5%</span>
                    </div>
                </div>
                <div className="p-6 hover:bg-gray-50 transition-colors">
                    <span className="text-xs font-bold text-gray-500 uppercase tracking-widest block mb-2">F1 Weighted</span>
                    <div className="flex items-baseline gap-2">
                        <span className="text-4xl font-mono font-bold text-gray-900">{MODEL_METRICS.f1Score}%</span>
                        <span className="text-xs text-emerald-600 font-medium bg-emerald-50 px-1 py-0.5 rounded">+3.2%</span>
                    </div>
                </div>
            </section>

            {/* Main Category Matrix */}
            <section>
                <div className="flex flex-col md:flex-row md:items-center justify-between mb-4 gap-4">
                    <h2 className="text-sm font-bold text-gray-900 uppercase tracking-wider flex items-center gap-2">
                        <Database className="h-4 w-4" />
                        Performance Metrics
                        <span className="text-gray-400 font-normal normal-case text-xs ml-2">source: performance_metrics.csv</span>
                    </h2>
                    <div className="flex gap-4 text-[10px] font-mono uppercase font-bold text-gray-500">
                        <span className="flex items-center gap-1"><div className="w-2 h-2 bg-emerald-500"></div> Optimal</span>
                        <span className="flex items-center gap-1"><div className="w-2 h-2 bg-amber-500"></div> Monitor</span>
                        <span className="flex items-center gap-1"><div className="w-2 h-2 bg-red-500"></div> Critical</span>
                        <span className="flex items-center gap-1"><div className="w-2 h-2 bg-red-600 border border-red-800"></div> Degraded (Below Threshold)</span>
                    </div>
                </div>
                
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-px bg-gray-200 border border-gray-200 shadow-sm">
                    {CATEGORIES.map((cat, idx) => (
                        <CategoryCard key={idx} category={cat} />
                    ))}
                </div>
            </section>

            {/* Hierarchy Specs */}
            <section className="bg-white border border-gray-200 p-6 shadow-sm">
                <h2 className="text-sm font-bold text-gray-900 uppercase tracking-wider mb-6 flex items-center gap-2">
                    <GitCommit className="h-4 w-4" />
                    Taxonomy Hierarchy
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-8">
                    {HIERARCHY_SPECS.map((group, idx) => (
                        <div key={idx} className="space-y-3">
                            <h3 className="text-xs font-bold text-gray-900 border-b border-gray-100 pb-2 uppercase">{group.parent}</h3>
                            <ul className="space-y-1.5">
                                {group.children.map((child, i) => (
                                    <li key={i} className="text-xs text-gray-500 font-mono flex items-center gap-2">
                                        <span className="w-1 h-1 bg-gray-300 rounded-none"></span>
                                        {child}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </div>
            </section>

        </div>
      </main>
    </div>
  );
}
