import { SignalItem } from "@/app/data";
import { ContentPane } from "./ContentPane";
import { ClassificationPane } from "./ClassificationPane";
import { ContextSidebar } from "./ContextSidebar";
import { ActionStrip } from "./ActionStrip";
import { ArrowLeft } from "lucide-react";

interface ItemDetailViewProps {
  item: SignalItem;
  onBack: () => void;
  globalThreshold: number;
  onAction: (action: string, itemId: string) => void;
}

export const ItemDetailView = ({ item, onBack, globalThreshold, onAction }: ItemDetailViewProps) => {
  return (
    <div className="h-full flex flex-col bg-white">
      {/* Top Nav for Detail View */}
      <div className="h-12 border-b border-slate-200 flex items-center px-4 bg-white flex-shrink-0">
        <button 
          onClick={onBack}
          className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-900 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Live Feed
        </button>
      </div>

      <div className="flex-1 flex flex-col lg:flex-row overflow-y-auto lg:overflow-hidden">
        {/* Main Content Area (Split 2-pane) */}
        <div className="flex-1 flex flex-col lg:flex-row min-w-0">
           {/* Left: Text & Metadata */}
           <div className="flex-none lg:flex-[3] min-w-0 h-auto lg:h-full flex flex-col border-b lg:border-b-0">
             <ContentPane item={item} />
             <div className="sticky bottom-0 z-10 lg:static mt-auto">
                <ActionStrip onAction={(type) => onAction(type, item.id)} />
             </div>
           </div>

           {/* Right: Classification Results */}
           <div className="flex-none lg:flex-[2] min-w-0 lg:min-w-[300px] h-auto lg:h-full border-l border-slate-200">
             <ClassificationPane 
               classifications={item.classifications} 
               globalThreshold={globalThreshold}
             />
           </div>
        </div>

        {/* Far Right: Sidebar Context */}
        <div className="flex-none w-full lg:w-72 h-auto lg:h-full border-l border-slate-200 bg-slate-50">
          <ContextSidebar />
        </div>
      </div>
    </div>
  );
};
