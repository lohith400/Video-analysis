import { TrendingUp, TrendingDown, Activity } from "lucide-react";

export default function MetricCard({ label, value, trend, suffix = "" }) {
  // Generates unique, responsive mini SVG lines for the sparkline background
  const getSparklinePath = () => {
    if (trend === "up") {
      return "0,25 12,20 24,24 36,15 48,18 60,10 72,12 84,4";
    }
    if (trend === "down") {
      return "0,6 12,14 24,10 36,18 48,15 60,22 72,20 84,26";
    }
    return "0,15 12,14 24,16 36,15 48,16 60,15 72,16 84,15";
  };

  const getSparklineColor = () => {
    if (trend === "up") return "#10B981"; // Emerald
    if (trend === "down") return "#EF4444"; // Red
    return "#0284C7"; // Sky Blue
  };

  return (
    <div className="glass-card glass-card-hover rounded-2xl p-5 flex flex-col justify-between h-28 shadow-sm">
      {/* Metric Header */}
      <div className="flex items-start justify-between">
        <span className="text-[10px] font-heading font-extrabold text-sky-dark/65 uppercase tracking-wider">
          {label}
        </span>
        {trend && (
          <span
            className={`p-1 rounded-lg text-xs flex items-center justify-center border shadow-sm ${
              trend === "up"
                ? "text-emerald-600 bg-emerald-50/50 border-emerald-100"
                : "text-red-600 bg-red-50/50 border-red-100"
            }`}
          >
            {trend === "up" ? (
              <TrendingUp className="w-3.5 h-3.5" />
            ) : (
              <TrendingDown className="w-3.5 h-3.5" />
            )}
          </span>
        )}
      </div>

      {/* Metric Value & Odometer */}
      <div className="flex items-end justify-between mt-2.5">
        <div className="font-heading font-extrabold text-2.5xl sm:text-3xl text-sky-dark leading-none">
          {typeof value === "number" ? value.toLocaleString() : value}
          {suffix && (
            <span className="text-xs font-heading font-bold text-sky-default uppercase tracking-wider ml-1">
              {suffix}
            </span>
          )}
        </div>

        {/* Embedded Vector Sparkline */}
        <div className="select-none pointer-events-none opacity-60 mr-1.5">
          <svg className="w-16 h-7 overflow-visible" viewBox="0 0 84 30">
            <polyline
              fill="none"
              stroke={getSparklineColor()}
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={getSparklinePath()}
            />
          </svg>
        </div>
      </div>
    </div>
  );
}
