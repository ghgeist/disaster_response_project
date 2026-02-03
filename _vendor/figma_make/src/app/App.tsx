import { lazy, Suspense } from 'react';
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

export default function App() {
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
