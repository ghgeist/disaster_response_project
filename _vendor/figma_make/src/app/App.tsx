import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { StormSignalView } from '@/app/components/dashboard/StormSignalView';
import { ModelInformationDashboard } from '@/app/components/dashboard/ModelInformationDashboard';
import { AboutPage } from '@/app/components/dashboard/AboutPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/api/model-info-dashboard" element={<ModelInformationDashboard />} />
        <Route path="/api/model-info-dashboard/" element={<ModelInformationDashboard />} />
        <Route path="/api/dashboard" element={<StormSignalView />} />
        <Route path="/api/dashboard/" element={<StormSignalView />} />
        <Route path="/api/about" element={<AboutPage />} />
        <Route path="/api/about/" element={<AboutPage />} />
        <Route path="*" element={<StormSignalView />} />
      </Routes>
    </BrowserRouter>
  );
}
