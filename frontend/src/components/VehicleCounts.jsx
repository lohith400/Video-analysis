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

  // Filter to only display categories that are actively detected (> 0)
  const activeVehicles = VEHICLE_TYPES.filter(({ key }) => (counts[key] || 0) > 0);

  return (
    <div className="select-none">
      {/* Header telemetry tag */}
      <div className="flex items-center justify-between mb-4 border-b border-sky-border/30 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-sky-default/10 flex items-center justify-center text-sky-default">
            <Car className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider">
              Vehicle Breakdown
            </h3>
            <p className="text-[9px] text-sky-dark/45 font-mono uppercase">Live crossing segregation</p>
          </div>
        </div>
        <div className="flex items-center gap-1.5 font-mono text-[9.5px] font-bold text-sky-default bg-sky-surface border border-sky-border/40 px-2.5 py-0.5 rounded-lg shadow-sm">
          <span>SUM:</span>
          <span className="text-sky-dark font-extrabold select-all">
            <OdometerValue value={total} />
          </span>
        </div>
      </div>

      {activeVehicles.length === 0 ? (
        <div className="px-4 py-8 text-center text-[9px] font-heading font-extrabold text-sky-dark/40 uppercase tracking-widest border border-dashed border-sky-border/40 rounded-2xl bg-white/10">
          Waiting for vehicle crossings...
        </div>
      ) : (
        /* Sleek horizontal continuous layout inside a single rectangle */
        <div className="rounded-2xl border border-sky-border/30 bg-white/30 p-4 shadow-sm flex flex-wrap gap-x-5 gap-y-4 items-center justify-start">
          {activeVehicles.map(({ key, label, icon: IconComponent }, index) => {
            const count = counts[key] || 0;
            const ratio = total > 0 ? (count / total) * 100 : 0;
            
            return (
              <div 
                key={key} 
                className="flex items-center gap-3.5"
              >
                {/* Inline Icon container */}
                <div className="p-2.5 rounded-xl bg-sky-surface/30 text-sky-default border border-sky-border/20 shadow-inner flex items-center justify-center">
                  <IconComponent className="w-4 h-4" />
                </div>
                
                {/* Counts & Percentage Info */}
                <div className="flex flex-col">
                  <div className="flex items-baseline gap-1.5 leading-none">
                    <span className="font-mono text-base font-extrabold text-sky-dark select-all">
                      <OdometerValue value={count} />
                    </span>
                    <span className="font-mono text-[8px] font-extrabold text-sky-default bg-sky-surface/40 px-1.5 py-0.5 rounded-md border border-sky-border/10 select-none">
                      {ratio.toFixed(0)}%
                    </span>
                  </div>
                  <span className="text-[8.5px] font-heading font-extrabold text-sky-dark/50 uppercase tracking-widest mt-1.5 leading-none">
                    {label}
                  </span>
                </div>
                
                {/* Continuous visual divider line between cards */}
                {index < activeVehicles.length - 1 && (
                  <div className="h-7 w-[1px] bg-sky-border/30 ml-2 hidden sm:block select-none" />
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}





