import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

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
          <Route path="/api/model-info-dashboard" element={<ModelInformationDashboard />} />
          <Route path="/api/model-info-dashboard/" element={<ModelInformationDashboard />} />
          <Route path="/api/dashboard" element={<StormSignalView />} />
          <Route path="/api/dashboard/" element={<StormSignalView />} />
          <Route path="/api/about" element={<AboutPage />} />
          <Route path="/api/about/" element={<AboutPage />} />
          <Route path="*" element={<StormSignalView />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
