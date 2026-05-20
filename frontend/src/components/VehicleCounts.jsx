import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Car, Truck, Sparkles } from "lucide-react";

// Sleek Custom Vector Icons for Indian Road vehicles
function MotorcycleIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <circle cx="5" cy="18" r="3" />
      <circle cx="19" cy="18" r="3" />
      <path d="M12 18V9h4l2 3M8 18l3-9M16 9l-1.5-3h-4L9 9" />
    </svg>
  );
}

function BicycleIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <circle cx="5.5" cy="17.5" r="2.5" />
      <circle cx="18.5" cy="17.5" r="2.5" />
      <path d="M15 6a1 1 0 100-2 1 1 0 000 2zm-3 11.5L9 12h5l2.5 3M5.5 17.5L9 12M18.5 17.5L16 11M8.5 7.5h4L15 11" />
    </svg>
  );
}

function AutoRickshawIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <circle cx="12" cy="18" r="2" />
      <circle cx="5" cy="18" r="2" />
      <circle cx="19" cy="18" r="2" />
      <path d="M5 16V9l7-3 7 3v7M5 12h14M12 6v6" />
    </svg>
  );
}

function BusIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <rect x="3" y="4" width="18" height="12" rx="2" />
      <circle cx="7" cy="18" r="2" />
      <circle cx="17" cy="18" r="2" />
      <path d="M3 10h18M8 4v6M16 4v6" />
    </svg>
  );
}

const VEHICLE_TYPES = [
  { key: "car", label: "Car", icon: Car },
  { key: "motorcycle", label: "Motorcycle", icon: MotorcycleIcon },
  { key: "bus", label: "Bus", icon: BusIcon },
  { key: "truck", label: "Truck", icon: Truck },
  { key: "auto-rickshaw", label: "Auto-Rickshaw", icon: AutoRickshawIcon },
  { key: "bicycle", label: "Bicycle", icon: BicycleIcon },
  { key: "others", label: "Others", icon: Sparkles },
];

function OdometerValue({ value }) {
  const [displayValue, setDisplayValue] = useState(value);

  useEffect(() => {
    let start = displayValue;
    const end = value;
    if (start === end) return;

    const duration = 600; // ms
    const startTime = performance.now();

    const animate = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = progress * (2 - progress); // easeOutQuad
      const current = Math.floor(start + (end - start) * ease);
      
      setDisplayValue(current);

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        setDisplayValue(end);
      }
    };

    requestAnimationFrame(animate);
  }, [value]);

  return <span>{displayValue.toLocaleString()}</span>;
}

export default function VehicleCounts({ counts = {} }) {
  const total = counts.total || 0;

  return (
    <div className="select-none">
      {/* Header telemetry tag */}
      <div className="flex items-center justify-between mb-4 border-b border-sky-border/30 pb-2">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full live-pulse bg-sky-default shadow-sm shadow-sky-default/30" />
          <h3 className="font-heading font-extrabold text-sm text-sky-dark uppercase tracking-wider">
            Vehicle Breakdown
          </h3>
        </div>
        <div className="flex items-center gap-1.5 font-mono text-[11px] font-bold text-sky-default bg-sky-surface/30 border border-sky-border/40 px-2 py-0.5 rounded-lg shadow-sm">
          <span>SUM:</span>
          <span className="text-sky-dark font-extrabold">
            <OdometerValue value={total} />
          </span>
        </div>
      </div>

      {/* Grid count cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        {VEHICLE_TYPES.map(({ key, label, icon: IconComponent }) => {
          const count = counts[key] || 0;
          const ratio = total > 0 ? (count / total) * 100 : 0;
          
          return (
            <motion.div
              key={key}
              className="relative overflow-hidden rounded-xl p-3 bg-white/40 border border-sky-border/40 flex flex-col justify-between shadow-sm"
              whileHover={{ 
                y: -2, 
                borderColor: "#38BDF8", 
                backgroundColor: "rgba(255,255,255,0.7)",
                boxShadow: "0 4px 12px -2px rgba(56, 189, 248, 0.08)"
              }}
              transition={{ duration: 0.2 }}
            >
              {/* Card visual background shimmer loader overlay on updates */}
              {count > 0 && (
                <div className="absolute inset-0 bg-gradient-to-r from-sky-surface/5 via-sky-surface/30 to-sky-surface/5 pointer-events-none opacity-20 -translate-x-full animate-[shimmer_1.5s_infinite]" />
              )}

              {/* Icon & Ratio Gauge */}
              <div className="flex items-start justify-between">
                <div className="p-2 rounded-lg bg-sky-surface/60 text-sky-default border border-sky-border/20 shadow-inner">
                  <IconComponent className="w-4 h-4" />
                </div>
                
                {/* Micro linear progress capsule */}
                <div className="flex flex-col items-end gap-1">
                  <span className="font-mono text-[8px] font-bold text-sky-default/70">{ratio.toFixed(0)}%</span>
                  <div className="w-10 h-1 bg-sky-surface rounded-full overflow-hidden border border-sky-border/20">
                    <motion.div 
                      className="h-full bg-sky-default rounded-full" 
                      initial={{ width: 0 }}
                      animate={{ width: `${ratio}%` }}
                      transition={{ duration: 0.8, ease: "easeOut" }}
                    />
                  </div>
                </div>
              </div>

              {/* Numerical Counts & Labels */}
              <div className="mt-3.5 select-none leading-none">
                <div className="font-mono text-lg font-extrabold text-sky-dark select-all">
                  <OdometerValue value={count} />
                </div>
                <div className="text-[9px] font-heading font-extrabold text-sky-dark/55 uppercase tracking-wider mt-1.5">
                  {label}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

