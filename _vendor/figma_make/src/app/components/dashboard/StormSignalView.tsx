import { useState, useMemo, useEffect, useRef } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import type { SignalItem, ModelInfo, CategoryGroups } from '@/app/data';
import { DEFAULT_CATEGORY_GROUPS } from '@/app/data';
import { FeedPanel } from '@/app/components/dashboard/FeedPanel';
import { MetricsPanel } from '@/app/components/dashboard/MetricsPanel';
import { ClassificationPanel } from '@/app/components/dashboard/ClassificationPanel';
import { Radar, Bell, Settings, UserCircle, Menu, LayoutDashboard, FileBarChart, Info } from 'lucide-react';
import { toApiName, getCategories } from '@/app/utils/api';
import { Tooltip, TooltipTrigger, TooltipContent } from '@/app/components/ui/tooltip';
import {
  SidebarProvider,
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  useSidebar,
} from '@/app/components/ui/sidebar';

function mapFeedItem(item: { timestamp: string; [k: string]: unknown }): SignalItem {
  return {
    ...item,
    timestamp: new Date(item.timestamp),
  } as SignalItem;
}

function DashboardHeader({ modelInfo }: { modelInfo: ModelInfo | null }) {
  const { toggleSidebar } = useSidebar();
  return (
    <header className="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-4 flex-shrink-0 z-50">
      <div className="flex items-center gap-4">
        <button
          type="button"
          aria-label="Open menu"
          className="p-1.5 hover:bg-slate-100 rounded text-slate-500 hover:text-slate-900 transition-colors"
          onClick={toggleSidebar}
        >
          <Menu className="w-5 h-5" />
        </button>
        <div className="flex items-center gap-2">
          <div className="bg-blue-600 p-1 rounded-sm">
            <Radar className="w-4 h-4 text-white" />
          </div>
          <h1 className="text-lg font-bold tracking-tight text-slate-900">STORM SIGNAL</h1>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <Tooltip delayDuration={300}>
          <TooltipTrigger asChild>
            <button
              type="button"
              className="hidden lg:flex items-center gap-2 text-xs text-slate-600 bg-slate-100 px-3 py-1.5 rounded-full border border-slate-200 font-medium tracking-wide cursor-pointer hover:bg-slate-200 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              aria-label="System status - hover for model details"
            >
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              SYSTEM: OPERATIONAL
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom" sideOffset={8} className="bg-slate-900 text-white text-xs max-w-xs z-[100]">
            {modelInfo ? (
              <>
                Model version: {modelInfo.version} | {modelInfo.f1_score !== null ? `${Math.round(modelInfo.f1_score * 100)}%` : 'N/A'} F1-score | {Math.round(modelInfo.hierarchy_violations)}% Hierarchy Violations
              </>
            ) : (
              <>Loading model info...</>
            )}
          </TooltipContent>
        </Tooltip>
        <button className="p-1.5 hover:bg-slate-100 rounded-full text-slate-400 hover:text-slate-900 transition-colors">
          <Settings className="w-5 h-5" />
        </button>
        <button className="p-1.5 hover:bg-slate-100 rounded-full text-slate-400 hover:text-slate-900 transition-colors">
          <Bell className="w-5 h-5" />
        </button>
        <div className="h-6 w-px bg-slate-200 mx-1"></div>
        <button className="flex items-center gap-2 pl-1 pr-2 py-1 hover:bg-slate-100 rounded-full transition-colors group">
          <UserCircle className="w-7 h-7 text-slate-300 group-hover:text-slate-400 transition-colors" />
          <div className="flex flex-col items-start leading-none">
            <span className="text-xs font-bold text-slate-700">Operator_7</span>
            <span className="text-[10px] text-slate-400">Level 3 Clearance</span>
          </div>
        </button>
      </div>
    </header>
  );
}

export function StormSignalView() {
  const [signals, setSignals] = useState<SignalItem[]>([]);
  const [feedLoading, setFeedLoading] = useState(true);
  const [feedError, setFeedError] = useState<string | null>(null);
  const [selectedFilters, setSelectedFilters] = useState<string[]>([]);
  const [showMobileBanner, setShowMobileBanner] = useState(true);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [categoryGroups, setCategoryGroups] = useState<CategoryGroups>(DEFAULT_CATEGORY_GROUPS);
  const justDispatchedRef = useRef(false);
  const [isDesktop, setIsDesktop] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia("(min-width: 1024px)").matches;
  });

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const mediaQuery = window.matchMedia("(min-width: 1024px)");
    const handleChange = (event: MediaQueryListEvent) => {
      setIsDesktop(event.matches);
    };

    mediaQuery.addEventListener("change", handleChange);

    return () => {
      mediaQuery.removeEventListener("change", handleChange);
    };
  }, []);

  useEffect(() => {
    getCategories()
      .then((data) => setCategoryGroups(data.groups))
      .catch(() => { /* keep DEFAULT_CATEGORY_GROUPS */ });
  }, []);

  useEffect(() => {
    fetch("/api/model-info")
      .then((res) => {
        if (!res.ok) throw new Error(`Model info ${res.status}`);
        return res.json();
      })
      .then((data: ModelInfo) => {
        setModelInfo(data);
      })
      .catch(() => {
        setModelInfo({
          version: "unavailable",
          f1_score: null,
          status: "unknown",
          hierarchy_violations: 0,
        });
      });
  }, []);

  useEffect(() => {
    setFeedLoading(true);
    setFeedError(null);

    const params = new URLSearchParams({ limit: '15' });
    selectedFilters.forEach(cat => {
      params.append('categories[]', toApiName(cat));
    });

    fetch(`/api/feed?${params.toString()}`)
      .then((res) => {
        if (!res.ok) throw new Error(`Feed ${res.status}`);
        return res.json();
      })
      .then((data: { items?: unknown[] }) => {
        const items = Array.isArray(data?.items) ? data.items : [];
        const serverSignals = items.map((i) => mapFeedItem(i as { timestamp: string; [k: string]: unknown }));

        setSignals(prev => {
          const manualDispatches = prev.filter(s => s.id.startsWith("MANUAL"));
          return [...manualDispatches, ...serverSignals];
        });
      })
      .catch((err) => setFeedError(err?.message ?? 'Failed to load feed'))
      .finally(() => setFeedLoading(false));
  }, [selectedFilters]);

  useEffect(() => {
    if (justDispatchedRef.current) {
      const timeoutId = setTimeout(() => {
        const feedPanel = document.querySelector('[data-feed-panel]');
        if (feedPanel) {
          feedPanel.scrollTop = 0;
        }
        justDispatchedRef.current = false;
      }, 100);
      return () => clearTimeout(timeoutId);
    }
  }, [signals]);

  const handleToggleFilter = (category: string) => {
    setSelectedFilters(prev =>
      prev.includes(category)
        ? prev.filter(c => c !== category)
        : [...prev, category]
    );
  };

  const handleAddFilter = (category: string) => {
    setSelectedFilters(prev =>
      prev.includes(category) ? prev : [...prev, category]
    );
  };

  const handleClearFilters = () => {
    setSelectedFilters([]);
  };

  const filteredSignals = useMemo(() => {
    let result = signals;
    if (selectedFilters.length > 0) {
      result = signals.filter(s =>
        s.categories.some(c => selectedFilters.includes(c))
      );
    }

    const manuals = result.filter(s => s.id.startsWith("MANUAL"));
    const others = result.filter(s => !s.id.startsWith("MANUAL"));

    others.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    manuals.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    return [...manuals, ...others].slice(0, 15);
  }, [signals, selectedFilters]);

  const handleDispatch = (message: string, results: { severity: string; categories: { name: string; conf: number }[] }) => {
    const newSignal: SignalItem = {
      id: `MANUAL-${Date.now()}`,
      timestamp: new Date(),
      source: "Manual Dispatch",
      content: message,
      originalContent: null,
      language: "en",
      riskLevel: results.severity as SignalItem['riskLevel'],
      categories: results.categories
        .filter((c) => c.conf > 0.4 && c.name.toLowerCase() !== "related")
        .map((c) => c.name)
        .slice(0, 3),
      classifications: results.categories.map((c) => ({
        category: c.name,
        confidence: c.conf
      })),
      isTranslated: false
    };

    justDispatchedRef.current = true;

    setSignals(prev => {
      let ts = new Date();
      if (prev.length > 0) {
        const latest = prev[0].timestamp;
        if (latest > ts) {
          ts = new Date(latest.getTime() + 1000);
        }
      }

      const newSignalWithFixedTime: SignalItem = {
        ...newSignal,
        timestamp: ts
      };

      return [newSignalWithFixedTime, ...prev];
    });
  };

  const ResizeHandle = () => (
    <PanelResizeHandle className="w-1.5 bg-slate-50 hover:bg-blue-500 transition-colors flex items-center justify-center group focus:outline-none focus:bg-blue-500 border-x border-slate-200">
      <div className="h-8 w-1 rounded-full bg-slate-300 group-hover:bg-white/80 transition-colors" />
    </PanelResizeHandle>
  );

  const sidebarTheme =
    "bg-white border-r border-slate-200 shadow-sm [--sidebar:#ffffff] [--sidebar-foreground:#111827] [--sidebar-accent:#f1f5f9] [--sidebar-accent-foreground:#111827] [--sidebar-border:#e2e8f0] [--sidebar-ring:#94a3b8]";

  return (
    <SidebarProvider className={sidebarTheme}>
      <Sidebar className={sidebarTheme}>
        <SidebarHeader />
        <SidebarContent>
          <SidebarGroup className="p-4">
            <SidebarGroupContent>
              <SidebarMenu className="space-y-0.5">
                <SidebarMenuItem>
                  <SidebarMenuButton asChild className="rounded-md text-slate-900 hover:bg-slate-100 hover:text-slate-900 data-[active=true]:bg-slate-100">
                    <a href="/api/dashboard">
                      <LayoutDashboard className="h-4 w-4 text-slate-600" />
                      <span>Overview</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
                <SidebarMenuItem>
                  <SidebarMenuButton asChild className="rounded-md text-slate-900 hover:bg-slate-100 hover:text-slate-900">
                    <a href="/api/model-info-dashboard">
                      <FileBarChart className="h-4 w-4 text-slate-600" />
                      <span>Production Model</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
                <SidebarMenuItem>
                  <SidebarMenuButton asChild className="rounded-md text-slate-900 hover:bg-slate-100 hover:text-slate-900">
                    <a href="/api/about">
                      <Info className="h-4 w-4 text-slate-600" />
                      <span>About</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
      </Sidebar>
      <SidebarInset>
        <div className="h-screen w-full flex flex-col bg-slate-50 font-sans text-slate-900 overflow-hidden">
          <div className="lg:hidden fixed inset-0 z-[100] bg-slate-900/90 flex items-center justify-center p-6 backdrop-blur-sm">
            <div className="bg-white p-6 rounded-lg shadow-xl max-w-sm text-center">
              <Radar className="w-12 h-12 text-blue-500 mx-auto mb-4" />
              <h2 className="text-lg font-bold text-slate-900 mb-2">Desktop Optimized</h2>
              <p className="text-sm text-slate-600 mb-4">
                Storm Signal is optimized for desktop viewing. Please access from a larger screen for the full experience.
              </p>
              {showMobileBanner && (
                <button
                  onClick={() => setShowMobileBanner(false)}
                  className="text-xs text-slate-400 hover:text-slate-600 underline"
                >
                  Dismiss (View Anyway)
                </button>
              )}
            </div>
          </div>

          <DashboardHeader modelInfo={modelInfo} />

          <main className="flex-1 overflow-hidden relative">
            {isDesktop ? (
              <div className="h-full w-full">
                <PanelGroup direction="horizontal">
                  <Panel defaultSize={38} minSize={25} order={1} className="bg-white min-w-0 overflow-hidden">
                    <FeedPanel
                      signals={filteredSignals}
                      selectedFilters={selectedFilters}
                      onToggleFilter={handleToggleFilter}
                      onClearFilters={handleClearFilters}
                      categoryGroups={categoryGroups}
                      loading={feedLoading}
                      error={feedError}
                    />
                  </Panel>

                  <ResizeHandle />

                  <Panel defaultSize={34} minSize={20} order={2} className="bg-slate-50 min-w-0 overflow-hidden">
                    <MetricsPanel onCategoryClick={handleAddFilter} />
                  </Panel>

                  <ResizeHandle />

                  <Panel defaultSize={28} minSize={22} collapsible order={3} className="bg-white min-w-0 overflow-hidden">
                    <ClassificationPanel onDispatch={handleDispatch} />
                  </Panel>
                </PanelGroup>
              </div>
            ) : (
              <div className="h-full overflow-y-auto">
                <FeedPanel
                  signals={filteredSignals}
                  selectedFilters={selectedFilters}
                  onToggleFilter={handleToggleFilter}
                  onClearFilters={handleClearFilters}
                  categoryGroups={categoryGroups}
                  loading={feedLoading}
                  error={feedError}
                />
              </div>
            )}
          </main>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
