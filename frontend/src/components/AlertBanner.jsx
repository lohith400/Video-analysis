import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertOctagon, ShieldAlert, X } from "lucide-react";

const KANNADA_MAP = {
  "No Helmet": "ಹೆಲ್ಮೆಟ್ ಇಲ್ಲ",
  "Wrong Way": "ತಪ್ಪು ದಿಕ್ಕು",
  "Speeding": "ವೇಗ ಮೀರಿದೆ",
  "Triple Riding": "ಮೂವರು ಸವಾರಿ",
};

export default function AlertBanner({ violation, onDismiss }) {
  useEffect(() => {
    if (!violation) return;
    const t = setTimeout(onDismiss, 6000);
    return () => clearTimeout(t);
  }, [violation, onDismiss]);

  if (!violation) return null;

  const kannada = KANNADA_MAP[violation.type] || violation.type;

  return (
    <AnimatePresence>
      <motion.div
        className="flex items-start gap-4 p-4 rounded-2xl shadow-xl select-none relative overflow-hidden max-w-sm w-full"
        style={{
          backgroundColor: "rgba(254, 249, 195, 0.95)",
          backdropFilter: "blur(12px)",
          border: "1px solid rgba(245, 158, 11, 0.5)",
          boxShadow: "0 20px 25px -5px rgba(245, 158, 11, 0.15), 0 10px 10px -5px rgba(245, 158, 11, 0.1)",
        }}
        initial={{ opacity: 0, y: 50, scale: 0.9 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 20, scale: 0.95 }}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
      >
        {/* Subtle decorative hazard background stripes */}
        <div className="absolute top-0 right-0 w-24 h-full opacity-[0.03] pointer-events-none bg-[repeating-linear-gradient(45deg,#000,#000_10px,#fff_10px,#fff_20px)]" />

        {/* Warning Icon Badge */}
        <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center text-amber-600 border border-amber-500/20 shrink-0">
          <AlertOctagon className="w-5.5 h-5.5 animate-pulse" />
        </div>

        {/* Alert Content */}
        <div className="flex-1 min-w-0 pr-4">
          <div className="flex items-center gap-1.5">
            <ShieldAlert className="w-3.5 h-3.5 text-amber-700" />
            <span className="font-heading font-extrabold text-[11px] text-amber-800 uppercase tracking-widest leading-none">
              Traffic Infraction Alert
            </span>
          </div>

          <h4 className="font-heading font-extrabold text-sm text-amber-950 mt-2 uppercase tracking-wide leading-tight flex items-center gap-2 flex-wrap">
            {violation.type}
            {violation.plate && (
              <span className="font-mono text-xs font-black bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded text-amber-800 tracking-wider">
                {violation.plate}
              </span>
            )}
          </h4>
          
          <p className="text-xs font-heading font-semibold text-amber-800/80 mt-1 pb-1 border-b border-amber-900/10 leading-relaxed">
            {kannada} · ಉಲ್ಲಂಘನೆ ಪತ್ತೆಯಾಗಿದೆ
          </p>

          <span className="block font-mono text-[8px] text-amber-700/60 uppercase tracking-wider mt-1.5">
            IRIS_COPS TELEMETRY FEED // ACTIVE RECORDED
          </span>
        </div>

        {/* Dismiss Button */}
        <button
          onClick={onDismiss}
          className="text-amber-800 hover:text-amber-950 hover:bg-amber-500/15 p-1.5 rounded-lg transition-all duration-200 shrink-0 self-start"
        >
          <X className="w-4 h-4" />
        </button>
      </motion.div>
    </AnimatePresence>
  );
}

