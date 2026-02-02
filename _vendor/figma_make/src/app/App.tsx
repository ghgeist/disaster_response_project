import { useState, useMemo, useEffect } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import type { SignalItem } from '@/app/data';
import { FeedPanel } from '@/app/components/dashboard/FeedPanel';
import { MetricsPanel } from '@/app/components/dashboard/MetricsPanel';
import { ClassificationPanel } from '@/app/components/dashboard/ClassificationPanel';
import { Radar, Bell, Settings, UserCircle, Menu } from 'lucide-react';
import { toApiName } from '@/app/utils/api';

function mapFeedItem(item: { timestamp: string; [k: string]: unknown }): SignalItem {
  return {
    ...item,
    timestamp: new Date(item.timestamp),
  } as SignalItem;
}

export default function App() {
  const [signals, setSignals] = useState<SignalItem[]>([]);
  const [feedLoading, setFeedLoading] = useState(true);
  const [feedError, setFeedError] = useState<string | null>(null);
  const [selectedFilters, setSelectedFilters] = useState<string[]>([]);
  const [showMobileBanner, setShowMobileBanner] = useState(true);
  // Initialize isDesktop correctly to prevent flash on initial render
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

    setIsDesktop(mediaQuery.matches);
    mediaQuery.addEventListener("change", handleChange);

    return () => {
      mediaQuery.removeEventListener("change", handleChange);
    };
  }, []);

  useEffect(() => {
    setFeedLoading(true);
    setFeedError(null);
    
    // Build Query Params with filters
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
        setSignals(items.map((i) => mapFeedItem(i as { timestamp: string; [k: string]: unknown })));
      })
      .catch((err) => setFeedError(err?.message ?? 'Failed to load feed'))
      .finally(() => setFeedLoading(false));
  }, [selectedFilters]); // Trigger re-fetch when filters change

  const handleToggleFilter = (category: string) => {
    setSelectedFilters(prev => 
      prev.includes(category) 
        ? prev.filter(c => c !== category)
        : [...prev, category]
    );
  };

  // Handler for drill-down from metrics (only adds if not present)
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
    // If filters are applied server-side, we can just return the signals directly
    // However, keeping this local filter doesn't hurt and handles any optimistic updates or mixed states
    if (selectedFilters.length > 0) {
      result = signals.filter(s => 
        s.categories.some(c => selectedFilters.includes(c))
      );
    }
    
    // Force strict sort by timestamp descending (Newest First)
    // AND ensure manual dispatches are always pinned to the top for immediate feedback
    const manuals = result.filter(s => s.id.startsWith("MANUAL"));
    const others = result.filter(s => !s.id.startsWith("MANUAL"));
    
    others.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    
    // Sort manuals by time (if multiple)
    manuals.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    return [...manuals, ...others].slice(0, 15);
  }, [signals, selectedFilters]);

  // Handler for dispatching from classification panel
  const handleDispatch = (message: string, results: any) => {
    // Create a new signal item
    const newSignal: SignalItem = {
      id: `MANUAL-${Date.now()}`,
      timestamp: new Date(),
      source: "Manual Dispatch",
      content: message,
      originalContent: null,
      language: "en",
      riskLevel: results.severity,
      categories: results.categories
        .filter((c: any) => c.conf > 0.4 && c.name.toLowerCase() !== "related")
        .map((c: any) => c.name)
        .slice(0, 3), // Top 3
      classifications: results.categories.map((c: any) => ({
        category: c.name,
        confidence: c.conf
      })),
      isTranslated: false
    };

    setSignals(prev => {
      // Ensure the manual dispatch stays at the top by checking if we need to shim the timestamp
      let ts = new Date();
      if (prev.length > 0) {
        const latest = prev[0].timestamp;
        if (latest > ts) {
          // If the latest item is in the future (due to mock data generation), 
          // set our manual item to be 1 second ahead of it.
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

  return (
    <div className="h-screen w-full flex flex-col bg-slate-50 font-sans text-slate-900 overflow-hidden">
      {/* Mobile Warning Banner */}
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

      {/* Global Header - Light Theme */}
      <header className="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-4 flex-shrink-0 z-50">
        <div className="flex items-center gap-4">
          <button className="p-1.5 hover:bg-slate-100 rounded text-slate-500 hover:text-slate-900 transition-colors">
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
          <div className="hidden lg:flex items-center gap-2 text-xs text-slate-600 bg-slate-100 px-3 py-1.5 rounded-full border border-slate-200 font-medium tracking-wide">
             <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
             SYSTEM: OPERATIONAL
          </div>
          <button className="p-1.5 hover:bg-slate-100 rounded-full text-slate-400 hover:text-slate-900 transition-colors">
            <Settings className="w-5 h-5" />
          </button>
          <button className="p-1.5 hover:bg-slate-100 rounded-full text-slate-400 hover:text-slate-900 transition-colors relative">
            <Bell className="w-5 h-5" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full border border-white"></span>
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

      {/* 3-Panel Resizable Layout */}
      <main className="flex-1 overflow-hidden relative">
        {isDesktop ? (
          <div className="h-full w-full">
            <PanelGroup direction="horizontal">
              {/* Left Panel: Feed & Filters */}
              <Panel defaultSize={40} minSize={25} order={1} className="bg-white">
              <FeedPanel 
                signals={filteredSignals} 
                selectedFilters={selectedFilters}
                onToggleFilter={handleToggleFilter}
                onClearFilters={handleClearFilters}
                loading={feedLoading}
                error={feedError}
              />
              </Panel>
              
              <ResizeHandle />
              
              {/* Center Panel: Metrics */}
              <Panel defaultSize={35} minSize={20} order={2} className="bg-slate-50">
                <MetricsPanel onCategoryClick={handleAddFilter} />
              </Panel>
              
              <ResizeHandle />
              
              {/* Right Panel: Classification */}
              <Panel defaultSize={25} minSize={15} collapsible order={3} className="bg-white">
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
              loading={feedLoading}
              error={feedError}
            />
          </div>
        )}
      </main>
    </div>
  );
}
