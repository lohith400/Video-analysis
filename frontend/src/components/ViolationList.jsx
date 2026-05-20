import { motion, AnimatePresence } from "framer-motion";
import { AlertOctagon, CheckCircle2, ShieldAlert, AlertTriangle } from "lucide-react";

// Crop thumbnail viewfinder representing the ANPR focal scope
function CropThumbnail({ type }) {
  return (
    <div className="relative w-12 h-12 bg-amber-100/70 border border-amber-300 rounded-xl flex items-center justify-center overflow-hidden shrink-0 shadow-inner select-none">
      {/* Viewfinder crosshairs */}
      <div className="absolute top-1 left-1 w-1.5 h-1.5 border-t border-l border-amber-500" />
      <div className="absolute top-1 right-1 w-1.5 h-1.5 border-t border-r border-amber-500" />
      <div className="absolute bottom-1 left-1 w-1.5 h-1.5 border-b border-l border-amber-500" />
      <div className="absolute bottom-1 right-1 w-1.5 h-1.5 border-b border-r border-amber-500" />
      
      {/* Scanning scanning overlay line */}
      <div className="absolute inset-x-0 h-0.5 bg-amber-400/35 bottom-1/2 pointer-events-none animate-bounce" />
      
      <span className="absolute bottom-0.5 right-1 text-[5.5px] font-mono font-black text-amber-600/60 tracking-widest uppercase">CROP_SYS</span>
      
      <AlertTriangle className="w-5 h-5 text-amber-600" />
    </div>
  );
}

export default function ViolationList({ violations = [] }) {
  return (
    <div className="select-none">
      {/* Header telemetry */}
      <div className="flex items-center justify-between mb-4 border-b border-amber-200/50 pb-2">
        <div className="flex items-center gap-2">
          <AlertOctagon className="w-4 h-4 text-amber-600 animate-pulse" />
          <h3 className="font-heading font-extrabold text-sm text-amber-800 uppercase tracking-wider">
            Active Violations
          </h3>
        </div>
        <span
          className="font-mono text-[10px] font-bold px-2 py-0.5 rounded-lg bg-amber-100 border border-amber-200 text-amber-800 shadow-sm"
        >
          {violations.length} CAUTIONS
        </span>
      </div>

      <div className="flex flex-col gap-2.5 max-h-[220px] overflow-y-auto pr-1 custom-scrollbar">
        {violations.length === 0 ? (
          <div
            className="flex flex-col items-center justify-center gap-2 py-8 rounded-xl bg-emerald-50/40 border border-emerald-200/50 shadow-sm"
          >
            <CheckCircle2 className="w-7 h-7 text-emerald-500 animate-[bounce_2s_infinite]" />
            <div className="text-center">
              <span className="block font-heading font-extrabold text-xs text-emerald-800 uppercase tracking-wide">Signal Clear</span>
              <span className="text-[9px] font-heading font-bold text-emerald-600/75 uppercase tracking-wider mt-0.5 block">No road violations detected</span>
            </div>
          </div>
        ) : (
          <AnimatePresence initial={false}>
            {violations.map((v, i) => (
              <motion.div
                key={v.timestamp + i}
                className="flex items-center gap-3.5 p-3 rounded-xl border transition-all duration-300 shadow-sm"
                style={{
                  backgroundColor: "#FEF9C3",
                  borderColor: "rgba(245, 158, 11, 0.4)",
                  borderLeft: "4px solid #F59E0B",
                }}
                initial={{ opacity: 0, y: 15, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ type: "spring", stiffness: 350, damping: 25 }}
              >
                {/* Visual crop viewfinder */}
                <CropThumbnail type={v.type} />

                {/* Violation description */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <ShieldAlert className="w-3.5 h-3.5 text-amber-700 shrink-0" />
                    <span className="font-heading font-extrabold text-xs text-amber-900 uppercase tracking-wide leading-none">
                      {v.type}
                    </span>
                  </div>
                  
                  {v.plate ? (
                    <span className="block font-mono text-[10px] font-black text-amber-800 uppercase tracking-widest mt-1.5 select-all">
                      PLATE: {v.plate}
                    </span>
                  ) : (
                    <span className="block font-heading font-bold text-[8.5px] text-amber-700/50 uppercase mt-1 select-all">
                      Scanning ANPR focus target...
                    </span>
                  )}
                </div>

                {/* Elapsed timestamp badge */}
                {v.timestamp && (
                  <span className="font-mono text-[9.5px] font-bold text-amber-800 bg-white/60 border border-amber-200/40 px-2 py-0.5 rounded-lg shadow-inner self-start">
                    {new Date(v.timestamp).toTimeString().slice(0, 8)}
                  </span>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}

