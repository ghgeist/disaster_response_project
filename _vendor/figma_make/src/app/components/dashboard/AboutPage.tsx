import React from 'react';
import {
  Menu,
  Settings,
  Bell,
  User,
  Radar,
  LayoutDashboard,
  FileBarChart,
  Info,
} from 'lucide-react';
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

function StormHeader() {
  const { toggleSidebar } = useSidebar();
  return (
    <header className="h-16 bg-white border-b border-slate-200 px-4 flex items-center justify-between sticky top-0 z-50 shadow-sm">
      <div className="flex items-center gap-4">
        <button
          type="button"
          aria-label="Open menu"
          className="p-2 hover:bg-slate-100 rounded-md text-slate-500 transition-colors"
          onClick={toggleSidebar}
        >
          <Menu className="h-5 w-5" />
        </button>
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 bg-blue-600 rounded-lg flex items-center justify-center text-white shadow-sm">
            <Radar className="h-5 w-5" />
          </div>
          <span className="text-lg font-bold text-slate-900 tracking-tight">STORM SIGNAL</span>
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-4">
        <div className="hidden md:flex items-center gap-2 bg-slate-50 px-3 py-1.5 rounded-full border border-slate-100">
          <div className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_4px_2px_rgba(16,185,129,0.2)]" />
          <span className="text-[11px] font-bold text-slate-600 uppercase tracking-wide">System: Operational</span>
        </div>

        <div className="flex items-center text-slate-400 gap-1">
          <button className="p-2 hover:bg-slate-100 rounded-full transition-colors hover:text-slate-600"><Settings className="h-5 w-5" /></button>
          <button className="p-2 hover:bg-slate-100 rounded-full transition-colors hover:text-slate-600">
             <Bell className="h-5 w-5" />
          </button>
        </div>

        <div className="h-8 w-px bg-slate-200 mx-2 hidden md:block"></div>

        <div className="flex items-center gap-3 hidden md:flex">
          <div className="h-9 w-9 bg-slate-50 rounded-full flex items-center justify-center text-slate-400 border border-slate-200">
            <User className="h-5 w-5" />
          </div>
          <div className="text-left">
            <div className="text-sm font-bold text-slate-900 leading-none">Operator_7</div>
            <div className="text-[10px] text-slate-500 mt-1 font-medium">Level 3 Clearance</div>
          </div>
        </div>
      </div>
    </header>
  );
}

export function AboutPage() {
  const sidebarTheme =
    "bg-white border-r border-slate-200 shadow-sm [--sidebar:#ffffff] [--sidebar-foreground:#111827] [--sidebar-accent:#f1f5f9] [--sidebar-accent-foreground:#111827] [--sidebar-border:#e2e8f0] [--sidebar-ring:#94a3b8]";

  return (
    <SidebarProvider className={sidebarTheme}>
      <Sidebar className={sidebarTheme}>
        <SidebarHeader />
        <SidebarContent className="bg-white">
          <SidebarGroup className="p-4">
            <SidebarGroupContent>
              <SidebarMenu className="space-y-0.5">
                <SidebarMenuItem>
                  <SidebarMenuButton asChild className="rounded-md text-slate-900 hover:bg-slate-100 hover:text-slate-900">
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
                  <SidebarMenuButton asChild isActive className="rounded-md text-slate-900 hover:bg-slate-100 hover:text-slate-900 data-[active=true]:bg-slate-100 data-[active=true]:font-medium">
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
        <div className="min-h-screen bg-slate-50 font-sans text-slate-900 flex flex-col">
          <StormHeader />

          <div className="flex flex-1 overflow-hidden">
            <main className="flex-1 overflow-y-auto p-8">
              <div className="max-w-[1400px] mx-auto space-y-8">
                <div className="border-b border-slate-200 pb-4">
                  <h1 className="text-2xl font-bold text-slate-900 tracking-tight uppercase">Storm Signal</h1>
                </div>

                <section className="space-y-4">
                  <h2 className="text-xs font-bold text-slate-700 uppercase tracking-wider">Platform Overview</h2>
                  <p className="text-sm text-slate-900 leading-relaxed max-w-3xl">
                    Storm Signal is a working prototype that simulates real-time disaster intelligence delivery from high-volume public communication streams. It monitors incoming messages, classifies them across disaster-relevant categories, and surfaces signals that may warrant attention when volume, speed, and uncertainty exceed what manual review can handle.
                  </p>
                </section>

                <section className="space-y-4">
                  <h2 className="text-xs font-bold text-slate-700 uppercase tracking-wider">Designed for Operations</h2>
                  <p className="text-sm text-slate-900 leading-relaxed max-w-3xl">
                    Storm Signal is designed for people whose job is to continuously monitor, triage, and interpret large volumes of incoming information under time pressure. This includes:
                  </p>
                  <ul className="text-sm text-slate-900 leading-relaxed max-w-3xl space-y-2 list-none pl-0">
                    <li className="flex items-start gap-2">
                      <span className="text-slate-400 mt-0.5">•</span>
                      <span>Analysts responsible for scanning and classifying crisis-related information</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-slate-400 mt-0.5">•</span>
                      <span>Operations teams maintaining situational awareness and prioritization</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-slate-400 mt-0.5">•</span>
                      <span>Intelligence or risk functions supporting downstream response and decision-making</span>
                    </li>
                  </ul>
                </section>

                <section className="space-y-4">
                  <h2 className="text-xs font-bold text-slate-700 uppercase tracking-wider">Intelligence, Not Conclusions</h2>
                  <p className="text-sm text-slate-900 leading-relaxed max-w-3xl">
                    Signals surfaced by the platform support human judgment and require context, corroboration, and domain expertise to interpret.
                  </p>
                  <ul className="text-sm text-slate-900 leading-relaxed max-w-3xl space-y-2 list-none pl-0">
                    <li className="flex items-start gap-2">
                      <span className="text-slate-400 mt-0.5">•</span>
                      <span>Flagged signals indicate potential relevance, not verified facts</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-slate-400 mt-0.5">•</span>
                      <span>Confidence reflects model certainty, not real-world accuracy</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-slate-400 mt-0.5">•</span>
                      <span>Categories describe signal type, not urgency or priority</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-slate-400 mt-0.5">•</span>
                      <span>Noise is expected as a tradeoff for broader coverage and earlier visibility</span>
                    </li>
                  </ul>
                </section>
              </div>
            </main>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
