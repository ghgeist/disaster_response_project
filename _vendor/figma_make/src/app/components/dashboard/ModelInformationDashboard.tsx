import React, { useEffect, useState } from 'react';
import { Badge } from "@/app/components/ui/badge";
import { MODEL_METRICS, CATEGORIES } from "@/app/data/model_info_fallbacks";
import { fetchDashboard } from "@/app/data/model_info_api";
import { Info } from 'lucide-react';
import { cn } from "@/app/components/ui/utils";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/app/components/ui/tooltip";
import { Skeleton } from "@/app/components/ui/skeleton";
import { DashboardSidebar } from './DashboardSidebar';
import { StormHeader } from './StormHeader';
import { Footer } from './Footer';

const MetricCard = ({ label, value, tooltip }: { label: string; value: string; tooltip: string }) => (
  <div className="bg-white border border-slate-200 p-5 flex flex-col justify-between h-24 hover:border-slate-400 transition-colors rounded-none shadow-sm">
    <Tooltip>
        <TooltipTrigger asChild>
            <div className="flex items-center gap-1.5 w-fit cursor-help group">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-wider group-hover:text-slate-600 transition-colors">
                    {label}
                </span>
                <Info className="w-3.5 h-3.5 text-slate-400 group-hover:text-slate-600 transition-colors" />
            </div>
        </TooltipTrigger>
        <TooltipContent className="max-w-[200px] bg-slate-900 text-slate-50 border-slate-800">
            <p className="font-sans normal-case tracking-normal">{tooltip}</p>
        </TooltipContent>
    </Tooltip>
    <span className="text-4xl font-mono font-bold text-slate-900 tracking-tighter">{value}</span>
  </div>
);

type MatrixCategory = {
  name: string;
  f1: number;
  adjustedF1: number;
  count: number;
  hierarchyParentKey?: string;
  hierarchyParentLabel?: string;
};
type MatrixCategoryGroup = {
  key: string;
  label: string;
  items: MatrixCategory[];
  totalSupport: number;
  weightedF1Pct: number;
};

const MatrixCell = ({ category }: { category: MatrixCategory }) => {
    const rawPercentage = Math.round(category.f1 * 100);

    let barColor = "bg-blue-200";

    if (category.f1 < 0.20) {
        barColor = "bg-red-500";
    } else {
        if (category.f1 >= 0.75) {
            barColor = "bg-blue-700";
        } else if (category.f1 >= 0.50) {
            barColor = "bg-blue-500";
        } else {
            barColor = "bg-blue-200";
        }
    }

    return (
        <div className="flex flex-col p-3 border border-slate-200 h-24 bg-white hover:border-slate-400 transition-colors rounded-none relative overflow-hidden group">
             <span className="text-xs font-bold uppercase truncate w-full leading-tight text-slate-700 mb-1" title={category.name}>
                 {category.name}
             </span>

             <div className="w-full bg-slate-100 h-2 mb-1 rounded-full overflow-hidden">
                 <div className={cn("h-full transition-all rounded-full", barColor)} style={{ width: `${rawPercentage}%` }}></div>
             </div>

             <div className="flex flex-col mt-auto">
                <span className="text-lg font-mono font-bold text-slate-900 leading-tight">
                    {rawPercentage}%
                </span>
             </div>
        </div>
    );
};

const k = 200;
const UNGROUPED_KEY = "ungrouped";
const UNGROUPED_LABEL = "Ungrouped";
const HIERARCHY_GROUP_ORDER = [
  "aid_related",
  "infrastructure_related",
  "weather_related",
  "related",
  UNGROUPED_KEY,
];

function useDashboardPayload() {
  const [payload, setPayload] = useState<Awaited<ReturnType<typeof fetchDashboard>> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchDashboard()
      .then((data) => {
        if (!cancelled) setPayload(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Model info unavailable");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { payload, loading, error };
}

export function ModelInformationDashboard() {
  const { payload, loading, error } = useDashboardPayload();

  const modelId = payload?.model?.id ?? MODEL_METRICS.id;
  const modelVersion = payload?.model?.version ?? MODEL_METRICS.version;
  const lastSynced = payload?.model?.generatedAt ?? payload?.model?.lastUpdated ?? MODEL_METRICS.lastUpdated;
  const algorithmCode = payload?.model?.algorithm ?? "unknown";
  const algorithmName = payload?.model?.algorithmName ?? "Unknown";
  const f1Pct = payload?.metrics != null ? Math.round(payload.metrics.f1 * 100) : MODEL_METRICS.f1Score;
  const precisionPct = payload?.metrics != null ? Math.round(payload.metrics.precision * 100) : MODEL_METRICS.precision;
  const recallPct = payload?.metrics != null ? Math.round(payload.metrics.recall * 100) : MODEL_METRICS.recall;

  const categoriesFromApi = payload?.categories ?? [];
  const processedCategories = categoriesFromApi.length > 0
    ? categoriesFromApi.map((cat) => ({
        name: cat.label,
        f1: cat.f1,
        count: cat.support,
        adjustedF1: cat.f1 * (cat.support / (cat.support + k)),
        hierarchyParentKey: cat.hierarchyParentKey ?? UNGROUPED_KEY,
        hierarchyParentLabel: cat.hierarchyParentLabel ?? UNGROUPED_LABEL,
      }))
    : CATEGORIES.map((cat) => ({
        ...cat,
        adjustedF1: cat.f1 * (cat.count / (cat.count + k)),
        hierarchyParentKey: UNGROUPED_KEY,
        hierarchyParentLabel: UNGROUPED_LABEL,
      }));
  const sortedCategories = [...processedCategories].sort((a, b) => b.adjustedF1 - a.adjustedF1);
  const groupedCategories = sortedCategories.reduce<Record<string, { label: string; items: MatrixCategory[] }>>(
    (acc, category) => {
      const groupKey = category.hierarchyParentKey ?? UNGROUPED_KEY;
      const groupLabel = category.hierarchyParentLabel ?? UNGROUPED_LABEL;
      if (!acc[groupKey]) {
        acc[groupKey] = { label: groupLabel, items: [] };
      }
      acc[groupKey].items.push(category);
      return acc;
    },
    {},
  );
  const groupedCategoryEntries: MatrixCategoryGroup[] = Object.entries(groupedCategories).map(([groupKey, group]) => {
    const totalSupport = group.items.reduce((sum, category) => sum + category.count, 0);
    const weightedF1 = totalSupport > 0
      ? group.items.reduce((sum, category) => sum + category.f1 * category.count, 0) / totalSupport
      : 0;
    return {
      key: groupKey,
      label: group.label,
      items: group.items,
      totalSupport,
      weightedF1Pct: Math.round(weightedF1 * 100),
    };
  }).sort((left, right) => {
    const leftOrder = HIERARCHY_GROUP_ORDER.indexOf(left.key);
    const rightOrder = HIERARCHY_GROUP_ORDER.indexOf(right.key);
    const leftRank = leftOrder === -1 ? Number.MAX_SAFE_INTEGER : leftOrder;
    const rightRank = rightOrder === -1 ? Number.MAX_SAFE_INTEGER : rightOrder;
    if (leftRank !== rightRank) return leftRank - rightRank;
    return left.label.localeCompare(right.label);
  });

  const appContent = loading ? (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 flex flex-col">
      <StormHeader />
      <div className="flex-1 overflow-y-auto p-8" aria-busy="true" aria-live="polite">
        <span className="sr-only">Loading model information…</span>
        <div className="max-w-[1400px] mx-auto space-y-8">
          <div className="flex justify-between items-end border-b border-slate-200 pb-4">
            <div className="space-y-2">
              <Skeleton className="h-6 w-48" />
              <Skeleton className="h-4 w-64" />
            </div>
            <Skeleton className="h-8 w-24" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Array.from({ length: 3 }).map((_, index) => (
              <div key={`metric-card-skeleton-${index}`} className="bg-white border border-slate-200 p-5 h-24 rounded-none shadow-sm space-y-3">
                <Skeleton className="h-3 w-24" />
                <Skeleton className="h-8 w-20" />
              </div>
            ))}
          </div>

          <div className="space-y-4">
            <Skeleton className="h-4 w-72" />
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {Array.from({ length: 8 }).map((_, index) => (
                <div key={`matrix-skeleton-${index}`} className="border border-slate-200 h-24 bg-white rounded-none shadow-sm p-3 space-y-3">
                  <Skeleton className="h-3 w-24" />
                  <Skeleton className="h-2 w-full" />
                  <Skeleton className="h-5 w-12" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  ) : (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 flex flex-col">
      <StormHeader />

      <div className="flex flex-1 overflow-hidden">

          <main className="flex-1 overflow-y-auto p-8">
              {error && (
                <div className="max-w-[1400px] mx-auto mb-4 px-4 py-2 bg-amber-50 border border-amber-200 text-amber-800 text-sm rounded">
                  Model info unavailable. Showing fallback data.
                </div>
              )}
              <div className="max-w-[1400px] mx-auto space-y-8">
                  <div className="flex justify-between items-end border-b border-slate-200 pb-4">
                      <div>
                          <div className="flex items-center gap-3 mb-1">
                              <h1 className="text-2xl font-bold text-slate-900 tracking-tight uppercase">{modelId}</h1>
                              <Badge className="rounded-none bg-slate-900 text-white hover:bg-slate-800 font-mono text-xs">v{modelVersion}</Badge>
                              {algorithmCode !== "unknown" && (
                                <Badge className="rounded-none bg-blue-600 text-white hover:bg-blue-700 font-mono text-xs">
                                  {algorithmName}
                                </Badge>
                              )}
                          </div>
                          <p className="text-sm text-slate-500 font-mono">Last Synced: {lastSynced ? new Date(lastSynced).toISOString() : "—"}</p>
                      </div>
                      <div className="text-right">
                      </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <MetricCard
                        label="F1 Score"
                        value={`${f1Pct}%`}
                        tooltip="A single score that balances accuracy and coverage. It reflects how well the system finds important signals without generating too many mistakes."
                      />
                      <MetricCard
                        label="Precision"
                        value={`${precisionPct}%`}
                        tooltip="When the system flags something, how often it's actually correct. Higher precision means fewer false alarms."
                      />
                      <MetricCard
                        label="Recall"
                        value={`${recallPct}%`}
                        tooltip="How many real signals the system successfully catches. Higher recall means fewer missed events."
                      />
                  </div>

                  <div>
                      <div className="flex items-center justify-between mb-4">
                          <Tooltip>
                              <TooltipTrigger asChild>
                                  <div className="flex items-center gap-1.5 w-fit cursor-help group">
                                      <h2 className="text-xs font-bold text-slate-700 uppercase tracking-wider">PERFORMANCE MATRIX (F1 BY CATEGORY)</h2>
                                      <Info className="w-3.5 h-3.5 text-slate-400 group-hover:text-slate-600 transition-colors" />
                                  </div>
                              </TooltipTrigger>
                              <TooltipContent className="max-w-[260px] bg-slate-900 text-slate-50 border-slate-800">
                                  <p className="font-sans normal-case tracking-normal">
                                      Per-category model performance using F1 score.
                                      Lower scores often reflect rare or harder-to-detect categories.
                                  </p>
                              </TooltipContent>
                          </Tooltip>
                      </div>

                      <div className="space-y-6">
                          {groupedCategoryEntries.map((group) => (
                            <section key={group.key} className="space-y-3">
                              <div className="bg-white border border-slate-200 px-4 py-3 rounded-none">
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                  <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                                    {group.label}
                                  </h3>
                                  <div className="flex items-center gap-4 text-xs font-mono text-slate-600">
                                    <span>
                                      GROUP F1:{" "}
                                      <strong className="text-slate-900">
                                        {group.weightedF1Pct}%
                                      </strong>
                                    </span>
                                    <span>
                                      SUPPORT:{" "}
                                      <strong className="text-slate-900">
                                        {group.totalSupport}
                                      </strong>
                                    </span>
                                    <span>
                                      CATEGORIES:{" "}
                                      <strong className="text-slate-900">{group.items.length}</strong>
                                    </span>
                                  </div>
                                </div>
                              </div>
                              <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6">
                                {group.items.map((cat, idx) => (
                                  <MatrixCell key={`${group.key}-${cat.name}-${idx}`} category={cat} />
                                ))}
                              </div>
                            </section>
                          ))}
                      </div>
                  </div>

              </div>
          </main>
      </div>
      <Footer />
    </div>
  );

  return (
    <DashboardSidebar activePage="production-model">
      {appContent}
    </DashboardSidebar>
  );
}
