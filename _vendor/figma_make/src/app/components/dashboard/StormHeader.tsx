import React from 'react';
import {
  Menu,
  Settings,
  Bell,
  User,
  Radar,
} from 'lucide-react';
import { useSidebar } from '@/app/components/ui/sidebar';

export function StormHeader() {
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
