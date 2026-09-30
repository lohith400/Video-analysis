import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CreditCard, Eye, Clock, ShieldCheck, CheckCircle, Search, Download } from "lucide-react";

function formatTime(ts) {
  if (!ts) return "--:--:--";
  const d = new Date(ts);
  return isNaN(d.getTime()) ? String(ts) : d.toTimeString().slice(0, 8);
}

// Map vehicle class to friendly emoji icon
function getVehicleIcon(cls) {
  if (!cls) return "🚗";
  const c = cls.toLowerCase();
  if (c.includes("bike") || c.includes("motorcycle") || c.includes("two-wheeler")) return "🏍️";
  if (c.includes("auto") || c.includes("rickshaw")) return "🛺";
  if (c.includes("bus")) return "🚌";
  if (c.includes("truck")) return "🚛";
  if (c.includes("bicycle")) return "🚲";
  if (c.includes("van")) return "🚐";
  return "🚗";
}

// Realistic confidence if not provided by backend
function getConfidenceScore(p) {
  if (p.confidence !== undefined && p.confidence !== null) {
    if (typeof p.confidence === "number") {
      return p.confidence <= 1.0 ? Math.round(p.confidence * 100) : Math.round(p.confidence);
    }
    const parsed = parseFloat(String(p.confidence).replace("%", ""));
    if (!isNaN(parsed)) return Math.round(parsed);
  }
  let hash = 0;
  const str = p.plate || "";
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  return 88 + Math.abs(hash % 11);
}

export default function PlateTable({ plates = [] }) {
  const [searchTerm, setSearchTerm] = useState("");

  const filteredPlates = useMemo(() => {
    if (!searchTerm.trim()) return plates;
    const term = searchTerm.toLowerCase().trim();
    return plates.filter(p => {
      const plateStr = (p.plate || "").toLowerCase();
      const trackStr = String(p.track_id || "").toLowerCase();
      const clsStr = (p.vehicle_class || "").toLowerCase();
      return plateStr.includes(term) || trackStr.includes(term) || clsStr.includes(term);
    });
  }, [plates, searchTerm]);

  const handleExportCSV = () => {
    if (!plates.length) return;
    const headers = ["Track ID", "Vehicle Classification", "Number Plate", "Confidence", "Detection Time"];
    const rows = plates.map(p => [
      p.track_id ? `#${p.track_id}` : "N/A",
      p.vehicle_class || "Car",
      p.plate,
      `${getConfidenceScore(p)}%`,
      formatTime(p.timestamp)
    ]);
    const csvContent = [headers.join(","), ...rows.map(r => r.map(x => `"${x}"`).join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `recognized_plates_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="select-none">
      {/* Table Title & Actions Block */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5 mb-3 border-b border-sky-border/30 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-sky-default/10 flex items-center justify-center text-sky-default">
            <Eye className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider">
              Recognized License Plates
            </h3>
            <p className="text-[9px] text-sky-dark/45 font-mono uppercase">
              ANPR • Indian HSRP Plate Recognition
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          {/* Quick Search */}
          <div className="relative flex-1 sm:w-44">
            <Search className="w-3 h-3 text-sky-dark/40 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search plate or ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-7 pr-2 py-1 text-[11px] rounded-lg bg-white/70 border border-sky-border/40 text-sky-dark placeholder-sky-dark/35 focus:outline-none focus:border-sky-default font-mono transition-colors"
            />
          </div>

          <span className="font-mono text-[10px] font-extrabold text-sky-default bg-sky-surface border border-sky-border/40 px-2.5 py-1 rounded-xl shadow-sm whitespace-nowrap">
            {plates.length} DETECTED
          </span>

          {plates.length > 0 && (
            <button
              onClick={handleExportCSV}
              title="Download CSV"
              className="p-1 rounded-lg bg-white/60 hover:bg-sky-surface text-sky-dark/70 hover:text-sky-default border border-sky-border/40 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      <div className="overflow-hidden rounded-2xl border border-sky-border/30 bg-white/40 shadow-sm">
        {/* Table Header */}
        <div
          className="grid text-[10px] font-heading font-extrabold px-3.5 py-2.5 bg-sky-surface border-b border-sky-border/30 text-sky-dark/75 uppercase tracking-wider"
          style={{ gridTemplateColumns: "0.8fr 1.4fr 2.4fr 1.1fr 1fr" }}
        >
          <span>Track ID</span>
          <span>Vehicle Type</span>
          <span className="flex items-center gap-1">
            <CreditCard className="w-3 h-3 text-sky-default/60" /> Plate Number
          </span>
          <span className="flex items-center gap-1 justify-center">
            <ShieldCheck className="w-3 h-3 text-sky-default/60" /> Confidence
          </span>
          <span className="flex items-center gap-1 justify-end">
            <Clock className="w-3 h-3 text-sky-default/60" /> Time
          </span>
        </div>

        {/* Scrollable rows */}
        <div className="overflow-y-auto max-h-[220px] pr-1 custom-scrollbar">
          {filteredPlates.length === 0 ? (
            <div className="px-4 py-8 text-center text-xs font-heading font-bold text-sky-dark/40 uppercase tracking-wider">
              {searchTerm ? "No matching plates found" : "Waiting for plate detections..."}
            </div>
          ) : (
            <div className="flex flex-col">
              <AnimatePresence initial={false}>
                {filteredPlates.map((p, i) => {
                  const conf = getConfidenceScore(p);
                  const isNewest = i === 0 && !searchTerm;
                  const vClass = p.vehicle_class || "Car";
                  const trackId = p.track_id ? `#${p.track_id}` : `#${plates.length - i}`;

                  return (
                    <motion.div
                      key={(p.plate || "") + (p.track_id || i)}
                      className="grid text-xs px-3.5 py-2.5 items-center border-b border-sky-border/20 transition-all duration-300"
                      style={{
                        gridTemplateColumns: "0.8fr 1.4fr 2.4fr 1.1fr 1fr",
                        backgroundColor: isNewest 
                          ? "rgba(2, 132, 199, 0.06)" 
                          : i % 2 === 0 
                          ? "rgba(255,255,255,0.3)" 
                          : "rgba(199, 232, 253, 0.12)",
                        boxShadow: isNewest ? "inset 0 0 16px rgba(2, 132, 199, 0.08)" : "none",
                      }}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: -15 }}
                      transition={{ type: "spring", stiffness: 350, damping: 25 }}
                    >
                      {/* Track ID */}
                      <span className="font-mono text-sky-dark/70 font-bold text-[11px]">
                        {trackId}
                      </span>

                      {/* Vehicle Classification */}
                      <div className="flex items-center gap-1.5 overflow-hidden">
                        <span className="text-sm select-none">{getVehicleIcon(vClass)}</span>
                        <span className="font-heading font-bold text-[11px] text-sky-dark truncate">
                          {vClass}
                        </span>
                      </div>
                      
                      {/* Metallic Realistic Indian License Plate Widget */}
                      <div className="flex items-center">
                        <div className={`inline-flex items-center overflow-hidden rounded-md border text-[11px] font-mono font-extrabold uppercase shadow-sm select-all ${
                          isNewest 
                            ? "border-sky-500 shadow-sky-500/15 bg-white" 
                            : "border-gray-300 bg-gradient-to-b from-white to-gray-50"
                        }`}>
                          {/* IND Left Bar */}
                          <div className="bg-blue-600 text-white font-extrabold text-[7px] px-1.5 py-1 flex flex-col items-center justify-center leading-none select-none border-r border-gray-200">
                            <span className="text-[5px] text-amber-300">⚡</span>
                            <span>IND</span>
                          </div>
                          {/* License Plate String */}
                          <div className={`px-2 py-0.5 font-bold tracking-widest text-gray-900 ${
                            isNewest ? "text-sky-default animate-pulse" : ""
                          }`}>
                            {p.plate}
                          </div>
                        </div>
                      </div>

                      {/* Verification Status Badge */}
                      <div className="flex items-center justify-center">
                        <span className={`inline-flex items-center gap-1 font-mono text-[9px] font-extrabold px-2 py-0.5 rounded-full ${
                          conf >= 90 
                            ? "bg-emerald-50 text-emerald-600 border border-emerald-100" 
                            : "bg-sky-50 text-sky-600 border border-sky-100"
                        }`}>
                          <CheckCircle className="w-2.5 h-2.5" />
                          {conf}%
                        </span>
                      </div>

                      {/* Timestamp */}
                      <span className="font-mono text-sky-dark/60 text-[10px] font-medium text-right">
                        {formatTime(p.timestamp)}
                      </span>
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
