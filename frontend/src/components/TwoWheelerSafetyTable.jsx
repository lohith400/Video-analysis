import { motion, AnimatePresence } from "framer-motion";
import { ShieldCheck, ShieldAlert, CheckCircle, AlertTriangle, MinusCircle } from "lucide-react";

export default function TwoWheelerSafetyTable({ statuses = [] }) {
  // Sort from highest track_id (most recent) to lowest so newest is always first
  const sortedStatuses = [...statuses].sort((a, b) => b.track_id - a.track_id);

  return (
    <div className="select-none">
      {/* Table Title Block */}
      <div className="flex items-center justify-between mb-4 border-b border-sky-border/30 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-sky-default/10 flex items-center justify-center text-sky-default">
            <span className="text-sm">🏍️</span>
          </div>
          <div>
            <h3 className="font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider">
              Two-Wheeler Safety Log
            </h3>
            <p className="text-[9px] text-sky-dark/45 font-mono uppercase">Helmet & Passenger Compliance Feed</p>
          </div>
        </div>
        <span className="font-mono text-[10px] font-extrabold text-sky-default bg-sky-surface border border-sky-border/40 px-2.5 py-1 rounded-xl shadow-sm">
          {sortedStatuses.length} TRACKED
        </span>
      </div>

      <div className="overflow-hidden rounded-2xl border border-sky-border/30 bg-white/40 shadow-sm">
        {/* Table Header */}
        <div
          className="grid text-[10px] font-heading font-extrabold px-4 py-2.5 bg-sky-surface border-b border-sky-border/30 text-sky-dark/75 uppercase tracking-wider"
          style={{ gridTemplateColumns: "0.8fr 1.6fr 1.3fr 1.3fr" }}
        >
          <span>Track ID</span>
          <span>Plate Code</span>
          <span className="text-center flex items-center justify-center gap-1"><ShieldCheck className="w-3.5 h-3.5 text-sky-default/60" /> Rider Status</span>
          <span className="text-center flex items-center justify-center gap-1"><ShieldCheck className="w-3.5 h-3.5 text-sky-default/60" /> Pillion Status</span>
        </div>

        {/* Scrollable rows with slide-in animations */}
        <div className="overflow-y-auto max-h-[220px] pr-1 custom-scrollbar">
          {sortedStatuses.length === 0 ? (
            <div
              className="px-4 py-10 text-center text-xs font-heading font-bold text-sky-dark/40 uppercase tracking-wider"
            >
              Waiting for two-wheelers...
            </div>
          ) : (
            <div className="flex flex-col">
              <AnimatePresence initial={false}>
                {sortedStatuses.map((s, i) => {
                  const isNewest = i === 0;

                  // Rider status styling
                  let riderBadgeColor = "bg-gray-50 text-gray-500 border-gray-200";
                  let riderText = "Checking";
                  let RiderIcon = MinusCircle;
                  if (s.rider_helmet === "helmet") {
                    riderBadgeColor = "bg-emerald-50 text-emerald-600 border-emerald-100";
                    riderText = "Helmet";
                    RiderIcon = CheckCircle;
                  } else if (s.rider_helmet === "no_helmet") {
                    riderBadgeColor = "bg-red-50 text-red-600 border-red-100 animate-pulse";
                    riderText = "No Helmet";
                    RiderIcon = AlertTriangle;
                  }

                  // Pillion status styling
                  let pillionBadgeColor = "bg-gray-50 text-gray-500 border-gray-200";
                  let pillionText = "Checking";
                  let PillionIcon = MinusCircle;
                  if (s.pillion_helmet === "helmet") {
                    pillionBadgeColor = "bg-emerald-50 text-emerald-600 border-emerald-100";
                    pillionText = "Helmet";
                    PillionIcon = CheckCircle;
                  } else if (s.pillion_helmet === "no_helmet") {
                    pillionBadgeColor = "bg-red-50 text-red-600 border-red-100 animate-pulse";
                    pillionText = "No Helmet";
                    PillionIcon = AlertTriangle;
                  } else if (s.pillion_helmet === "none") {
                    pillionBadgeColor = "bg-sky-50/50 text-sky-dark/40 border-sky-100";
                    pillionText = "No Pillion";
                    PillionIcon = MinusCircle;
                  }

                  return (
                    <motion.div
                      key={s.track_id}
                      className="grid text-xs px-4 py-3 items-center border-b border-sky-border/20 transition-all duration-300"
                      style={{
                        gridTemplateColumns: "0.8fr 1.6fr 1.3fr 1.3fr",
                        backgroundColor: isNewest 
                          ? "rgba(2, 132, 199, 0.05)" 
                          : i % 2 === 0 
                          ? "rgba(255,255,255,0.25)" 
                          : "rgba(199, 232, 253, 0.1)",
                        boxShadow: isNewest ? "inset 0 0 16px rgba(2, 132, 199, 0.08)" : "none",
                      }}
                      initial={{ opacity: 0, y: 15 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ type: "spring", stiffness: 350, damping: 25 }}
                    >
                      {/* Track ID */}
                      <span className="font-mono text-sky-dark/50 font-bold">#{s.track_id}</span>
                      
                      {/* Miniature Indian License Plate Style */}
                      <div className="flex items-center">
                        {s.plate && s.plate !== "UNKNOWN" ? (
                          <div className="inline-flex items-center overflow-hidden rounded-md border border-gray-300 bg-gradient-to-b from-white to-gray-50 text-[10px] font-mono font-extrabold uppercase shadow-sm select-all">
                            {/* IND Left Bar */}
                            <div className="bg-blue-600 text-white font-extrabold text-[6px] px-1.5 py-1.5 flex flex-col items-center justify-center leading-none select-none border-r border-gray-200">
                              <span>IND</span>
                            </div>
                            {/* Plate String */}
                            <div className="px-2 py-0.5 font-bold text-gray-900 tracking-wider">
                              {s.plate}
                            </div>
                          </div>
                        ) : (
                          <span className="inline-flex items-center font-mono text-[9px] font-extrabold text-sky-dark/30 bg-sky-surface/30 px-2 py-0.5 rounded border border-sky-border/20 uppercase">
                            UNKNOWN
                          </span>
                        )}
                      </div>

                      {/* Rider Badge */}
                      <div className="flex justify-center">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full font-heading font-extrabold text-[8px] uppercase tracking-wide text-center leading-none border ${riderBadgeColor}`}>
                          <RiderIcon className="w-2.5 h-2.5" />
                          {riderText}
                        </span>
                      </div>

                      {/* Pillion Badge */}
                      <div className="flex justify-center">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full font-heading font-extrabold text-[8px] uppercase tracking-wide text-center leading-none border ${pillionBadgeColor}`}>
                          <PillionIcon className="w-2.5 h-2.5" />
                          {pillionText}
                        </span>
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

