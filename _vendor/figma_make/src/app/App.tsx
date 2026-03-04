import { lazy, Suspense, useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

/**
 * SPA routing overview:
 * - Flask serves the same dashboard `index.html` shell for all dashboard routes.
 * - React Router maps those SPA paths to their respective dashboard views.
 * - Primary routes: `/dashboard`, `/production-model`, `/about` (legacy `/api/*` routes redirect).
 */

const StormSignalView = lazy(() =>
  import('@/app/components/dashboard/StormSignalView').then((module) => ({
    default: module.StormSignalView,
  })),
);
const ModelInformationDashboard = lazy(() =>
  import('@/app/components/dashboard/ModelInformationDashboard').then((module) => ({
    default: module.ModelInformationDashboard,
  })),
);
const AboutPage = lazy(() =>
  import('@/app/components/dashboard/AboutPage').then((module) => ({
    default: module.AboutPage,
  })),
);

function LoadingFallback() {
  return (
    <div className="flex min-h-screen items-center justify-center text-slate-500">
      Loading dashboard...
    </div>
  );
}

const HEALTH_POLL_INTERVAL_MS = 2_000;
const HEALTH_REQUEST_TIMEOUT_MS = 1_500;
const MAX_STARTUP_WAIT_MS = 12_000;

function shouldUseStartupGate(): boolean {
  const hostname = window.location.hostname;
  const isReplitHost = hostname.endsWith('.replit.dev') || hostname.endsWith('.repl.co');
  return import.meta.env.PROD && isReplitHost;
}

async function checkBackendHealth(): Promise<boolean> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), HEALTH_REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch('/health', {
      cache: 'no-store',
      signal: controller.signal,
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function StartupScreen() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-3 px-6 text-center">
      <p className="text-base font-medium text-slate-800">Starting demo...</p>
      <p className="max-w-md text-sm text-slate-500">
        This deployment may take a few extra seconds on first launch.
      </p>
    </div>
  );
}

export default function App() {
  const startupGateEnabled = shouldUseStartupGate();
  const [isBackendReady, setIsBackendReady] = useState(!startupGateEnabled);

  useEffect(() => {
    if (!startupGateEnabled) {
      return undefined;
    }

    let isActive = true;

    const pollHealth = async () => {
      const isHealthy = await checkBackendHealth();
      if (isActive && isHealthy) {
        setIsBackendReady(true);
        window.clearInterval(intervalId);
      }
    };

    const intervalId = window.setInterval(() => {
      void pollHealth();
    }, HEALTH_POLL_INTERVAL_MS);
    void pollHealth();
    const maxWaitTimeoutId = window.setTimeout(() => {
      if (isActive) {
        setIsBackendReady(true);
        window.clearInterval(intervalId);
      }
    }, MAX_STARTUP_WAIT_MS);

    return () => {
      isActive = false;
      window.clearInterval(intervalId);
      window.clearTimeout(maxWaitTimeoutId);
    };
  }, [startupGateEnabled]);

  if (!isBackendReady) {
    return <StartupScreen />;
  }

  return (
    <BrowserRouter>
      <Suspense fallback={<LoadingFallback />}>
        <Routes>
          <Route path="/production-model" element={<ModelInformationDashboard />} />
          <Route path="/production-model/" element={<ModelInformationDashboard />} />
          <Route path="/dashboard" element={<StormSignalView />} />
          <Route path="/dashboard/" element={<StormSignalView />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/about/" element={<AboutPage />} />
          <Route path="*" element={<StormSignalView />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
