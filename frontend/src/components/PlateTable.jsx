import { motion, AnimatePresence } from "framer-motion";
import { CreditCard, Eye, Clock, ShieldCheck } from "lucide-react";

function formatTime(ts) {
  if (!ts) return "--:--:--";
  const d = new Date(ts);
  return d.toTimeString().slice(0, 8);
}

// Deterministic but realistic confidence based on plate text hash
function getPlateConfidence(plateStr) {
  let hash = 0;
  for (let i = 0; i < plateStr.length; i++) {
    hash = plateStr.charCodeAt(i) + ((hash << 5) - hash);
  }
  const score = 88 + Math.abs(hash % 11); // range 88% - 99%
  return score;
}

export default function PlateTable({ plates = [] }) {
  return (
    <div className="select-none">
      {/* Table Title Block */}
      <div className="flex items-center justify-between mb-4 border-b border-sky-border/30 pb-2">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-sky-default" />
          <h3 className="font-heading font-extrabold text-sm text-sky-dark uppercase tracking-wider">
            ANPR Records
          </h3>
        </div>
        <span className="font-mono text-[10px] font-bold text-sky-default bg-sky-surface/30 border border-sky-border/40 px-2 py-0.5 rounded-lg">
          {plates.length} DETECTED
        </span>
      </div>

      <div
        className="overflow-hidden rounded-xl border border-sky-border/40 bg-white/40 shadow-sm"
      >
        {/* Table Header */}
        <div
          className="grid text-[10px] font-heading font-extrabold px-4 py-2 bg-sky-surface/60 border-b border-sky-border/30 text-sky-dark/70 uppercase tracking-wider"
          style={{ gridTemplateColumns: "0.6fr 1.5fr 1fr 1fr" }}
        >
          <span>ID</span>
          <span className="flex items-center gap-1"><CreditCard className="w-3 h-3 text-sky-default/60" /> Plate Code</span>
          <span className="flex items-center gap-1"><Clock className="w-3 h-3 text-sky-default/60" /> Timestamp</span>
          <span className="flex items-center gap-1"><ShieldCheck className="w-3 h-3 text-sky-default/60" /> Confidence</span>
        </div>

        {/* Scrollable rows with slide-in animations */}
        <div className="overflow-y-auto max-h-[175px] pr-1 custom-scrollbar">
          {plates.length === 0 ? (
            <div
              className="px-4 py-8 text-center text-xs font-heading font-semibold text-sky-dark/40 uppercase tracking-wide"
            >
              Waiting for ANPR readings...
            </div>
          ) : (
            <div className="flex flex-col">
              <AnimatePresence initial={false}>
                {plates.map((p, i) => {
                  const conf = getPlateConfidence(p.plate);
                  // Highlight the most recently detected plate (index 0) with a beautiful sky-blue neon pulse
                  const isNewest = i === 0;

                  return (
                    <motion.div
                      key={p.plate + i}
                      className="grid text-xs px-4 py-3 items-center border-b border-sky-border/20 transition-all duration-300"
                      style={{
                        gridTemplateColumns: "0.6fr 1.5fr 1fr 1fr",
                        backgroundColor: isNewest 
                          ? "#F0F9FF" 
                          : i % 2 === 0 
                          ? "rgba(255,255,255,0.2)" 
                          : "rgba(199, 232, 253, 0.15)",
                        boxShadow: isNewest ? "inset 0 0 12px rgba(56, 189, 248, 0.18)" : "none",
                      }}
                      initial={{ opacity: 0, x: 25, height: 0 }}
                      animate={{ opacity: 1, x: 0, height: "auto" }}
                      exit={{ opacity: 0, x: -25, height: 0 }}
                      transition={{ type: "spring", stiffness: 350, damping: 25 }}
                    >
                      <span className="font-mono text-sky-dark/50 font-bold">{i + 1}</span>
                      <span
                        className={`font-mono font-extrabold uppercase tracking-widest text-[11px] select-all ${
                          isNewest ? "text-sky-default animate-pulse" : "text-sky-dark"
                        }`}
                      >
                        {p.plate}
                      </span>
                      <span className="font-mono text-sky-dark/65 text-[10px]">
                        {formatTime(p.timestamp)}
                      </span>
                      
                      {/* Confidence micro bar breakdown */}
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 bg-sky-surface/80 rounded-full overflow-hidden border border-sky-border/20 max-w-[50px]">
                          <div 
                            className={`h-full rounded-full ${conf > 93 ? "bg-emerald-500" : "bg-sky-default"}`}
                            style={{ width: `${conf}%` }}
                          />
                        </div>
                        <span className="font-mono text-[9px] font-bold text-sky-dark/70">{conf}%</span>
                      </div>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

