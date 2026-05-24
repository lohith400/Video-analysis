import { motion, AnimatePresence } from "framer-motion";

export default function TwoWheelerSafetyTable({ statuses = [] }) {
  // Sort from highest track_id (most recent) to lowest so newest is always first
  const sortedStatuses = [...statuses].sort((a, b) => b.track_id - a.track_id);

  return (
    <div className="select-none">
      {/* Table Title Block */}
      <div className="flex items-center justify-between mb-4 border-b border-sky-border/30 pb-2">
        <div className="flex items-center gap-2">
          <span className="text-base">🏍️</span>
          <h3 className="font-heading font-extrabold text-sm text-sky-dark uppercase tracking-wider">
            Two-Wheeler Safety Log
          </h3>
        </div>
        <span className="font-mono text-[10px] font-bold text-sky-default bg-sky-surface/30 border border-sky-border/40 px-2 py-0.5 rounded-lg">
          {sortedStatuses.length} LOGGED
        </span>
      </div>

      <div className="overflow-hidden rounded-xl border border-sky-border/40 bg-white/40 shadow-sm">
        {/* Table Header */}
        <div
          className="grid text-[10px] font-heading font-extrabold px-4 py-2 bg-sky-surface/60 border-b border-sky-border/30 text-sky-dark/70 uppercase tracking-wider"
          style={{ gridTemplateColumns: "0.8fr 1.4fr 1.2fr 1.2fr" }}
        >
          <span>Track ID</span>
          <span>Plate</span>
          <span className="text-center">Rider Status</span>
          <span className="text-center">Pillion Status</span>
        </div>

        {/* Scrollable rows with slide-in animations */}
        <div className="overflow-y-auto max-h-[200px] pr-1 custom-scrollbar">
          {sortedStatuses.length === 0 ? (
            <div
              className="px-4 py-8 text-center text-xs font-heading font-semibold text-sky-dark/40 uppercase tracking-wide"
            >
              Waiting for two-wheelers...
            </div>
          ) : (
            <div className="flex flex-col">
              <AnimatePresence initial={false}>
                {sortedStatuses.map((s, i) => {
                  const isNewest = i === 0;

                  // Rider status styling
                  let riderBadgeColor = "bg-gray-100 text-gray-600 border-gray-200/50";
                  let riderText = "Unknown";
                  if (s.rider_helmet === "helmet") {
                    riderBadgeColor = "bg-emerald-500/10 text-emerald-600 border border-emerald-500/20";
                    riderText = "Helmet";
                  } else if (s.rider_helmet === "no_helmet") {
                    riderBadgeColor = "bg-red-500/10 text-red-600 border border-red-500/20 animate-pulse";
                    riderText = "No Helmet";
                  }

                  // Pillion status styling
                  let pillionBadgeColor = "bg-gray-100 text-gray-600 border-gray-200/50";
                  let pillionText = "None";
                  if (s.pillion_helmet === "helmet") {
                    pillionBadgeColor = "bg-emerald-500/10 text-emerald-600 border border-emerald-500/20";
                    pillionText = "Helmet";
                  } else if (s.pillion_helmet === "no_helmet") {
                    pillionBadgeColor = "bg-red-500/10 text-red-600 border border-red-500/20 animate-pulse";
                    pillionText = "No Helmet";
                  } else if (s.pillion_helmet === "none") {
                    pillionBadgeColor = "bg-sky-surface/30 text-sky-dark/40 border border-sky-border/20";
                    pillionText = "No Pillion";
                  }

                  return (
                    <motion.div
                      key={s.track_id}
                      className="grid text-xs px-4 py-2.5 items-center border-b border-sky-border/20 transition-all duration-300"
                      style={{
                        gridTemplateColumns: "0.8fr 1.4fr 1.2fr 1.2fr",
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
                      <span className="font-mono text-sky-dark/50 font-bold">#{s.track_id}</span>
                      
                      <span className="font-mono font-extrabold uppercase tracking-widest text-[11px] select-all text-sky-dark">
                        {s.plate && s.plate !== "UNKNOWN" ? s.plate : (
                          <span className="text-[10px] text-sky-dark/30 font-sans tracking-normal font-bold">UNKNOWN</span>
                        )}
                      </span>

                      {/* Rider Badge */}
                      <div className="flex justify-center">
                        <span className={`px-2 py-0.5 rounded-full font-heading font-extrabold text-[8px] uppercase tracking-wide text-center leading-none ${riderBadgeColor}`}>
                          {riderText}
                        </span>
                      </div>

                      {/* Pillion Badge */}
                      <div className="flex justify-center">
                        <span className={`px-2 py-0.5 rounded-full font-heading font-extrabold text-[8px] uppercase tracking-wide text-center leading-none ${pillionBadgeColor}`}>
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
