import clsx from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

export const Badge = ({ 
  children, 
  variant = "neutral", 
  className 
}: { 
  children: React.ReactNode; 
  variant?: "neutral" | "danger" | "critical" | "warning" | "success" | "info"; 
  className?: string;
}) => {
  const styles = {
    neutral: "bg-slate-100 text-slate-600 border-slate-200",
    danger: "bg-red-50 text-red-700 border-red-200", // High
    critical: "bg-[#FF4500]/10 text-[#FF4500] border-[#FF4500]/20 font-bold", // Critical - Signal Orange
    warning: "bg-amber-50 text-amber-700 border-amber-200", // Medium
    success: "bg-emerald-50 text-emerald-700 border-emerald-200", // Low (Avoid generic green for disaster)
    info: "bg-blue-50 text-blue-700 border-blue-200",
  };

  return (
    <span className={cn("px-2 py-0.5 rounded text-xs font-medium border", styles[variant], className)}>
      {children}
    </span>
  );
};

export const ConfidenceBar = ({ 
  confidence, 
  threshold, 
  uncertaintyRange,
  label,
  className 
}: { 
  confidence: number; 
  threshold: number; 
  uncertaintyRange?: [number, number];
  label?: string;
  className?: string;
}) => {
  return (
    <div className={cn("w-full", className)}>
      {label && (
        <div className="flex justify-between text-xs mb-1">
          <span className="font-medium text-slate-700">{label}</span>
          <span className="text-slate-500 font-mono">{(confidence * 100).toFixed(0)}%</span>
        </div>
      )}
      <div className="relative h-4 bg-slate-200 rounded-sm w-full overflow-hidden border border-slate-300">
        {/* Pass/Fail Threshold Line */}
        <div 
          className="absolute top-0 bottom-0 w-0.5 bg-slate-800 z-30" 
          style={{ left: `${threshold * 100}%` }}
        />
        
        {/* Confidence Fill (Blue) */}
        <div 
          className="absolute top-0 bottom-0 left-0 bg-blue-600 z-10 transition-all duration-500"
          style={{ width: `${confidence * 100}%` }}
        >
          {/* Haze / Uncertainty at the end of the bar */}
          <div className="absolute right-0 top-0 bottom-0 w-4 bg-gradient-to-r from-transparent to-white/40" />
        </div>

        {/* Explicit Uncertainty Range (Gray Band) */}
        {uncertaintyRange && (
          <div 
            className="absolute top-1 bottom-1 bg-slate-400/30 z-20 pointer-events-none"
            style={{ 
              left: `${uncertaintyRange[0] * 100}%`, 
              width: `${(uncertaintyRange[1] - uncertaintyRange[0]) * 100}%` 
            }}
          />
        )}
      </div>
    </div>
  );
};
