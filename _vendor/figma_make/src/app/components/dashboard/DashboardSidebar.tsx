import React from 'react';
import {
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
} from '@/app/components/ui/sidebar';

export type ActivePage = 'overview' | 'production-model' | 'about';

interface DashboardSidebarProps {
  activePage: ActivePage;
  children: React.ReactNode;
}

export function DashboardSidebar({ activePage, children }: DashboardSidebarProps) {
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
                  <SidebarMenuButton
                    asChild
                    isActive={activePage === 'overview'}
                    className="rounded-md text-slate-900 hover:bg-slate-100 hover:text-slate-900 data-[active=true]:bg-slate-100 data-[active=true]:font-medium"
                  >
                    <a href="/dashboard">
                      <LayoutDashboard className="h-4 w-4 text-slate-600" />
                      <span>Overview</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
                <SidebarMenuItem>
                  <SidebarMenuButton
                    asChild
                    isActive={activePage === 'production-model'}
                    className="rounded-md text-slate-900 hover:bg-slate-100 hover:text-slate-900 data-[active=true]:bg-slate-100 data-[active=true]:font-medium"
                  >
                    <a href="/production-model">
                      <FileBarChart className="h-4 w-4 text-slate-600" />
                      <span>Production Model</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
                <SidebarMenuItem>
                  <SidebarMenuButton
                    asChild
                    isActive={activePage === 'about'}
                    className="rounded-md text-slate-900 hover:bg-slate-100 hover:text-slate-900 data-[active=true]:bg-slate-100 data-[active=true]:font-medium"
                  >
                    <a href="/about">
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
        {children}
      </SidebarInset>
    </SidebarProvider>
  );
}
