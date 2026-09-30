import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Compass, Search, Download, ShieldCheck, AlertTriangle } from "lucide-react";

function getVehicleIcon(cls) {
  if (!cls) return "🚗";
  const c = cls.toLowerCase();
  if (c.includes("bike") || c.includes("motorcycle")) return "🏍️";
  if (c.includes("auto") || c.includes("rickshaw")) return "🛺";
  if (c.includes("bus")) return "🚌";
  if (c.includes("truck")) return "🚛";
  if (c.includes("bicycle")) return "🚲";
  if (c.includes("van")) return "🚐";
  return "🚗";
}

export default function CrossingVehiclesTable({ vehicles = [] }) {
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState("all"); // "all", "twoWheelers", "withPlates"

  const filtered = useMemo(() => {
    let list = [...vehicles].sort((a, b) => (Number(a.track_id) || 0) - (Number(b.track_id) || 0));

    if (filterType === "twoWheelers") {
      list = list.filter(v => {
        const cls = String(v.vehicle_class || "").toLowerCase();
        return cls.includes("bike") || cls.includes("motorcycle") || cls.includes("scooter");
      });
    } else if (filterType === "withPlates") {
      list = list.filter(v => v.plate && v.plate !== "not detected" && v.plate !== "none");
    }

    if (searchTerm.trim()) {
      const term = searchTerm.toLowerCase().trim();
      list = list.filter(v => {
        const tid = String(v.track_id || "").toLowerCase();
        const plt = String(v.plate || "").toLowerCase();
        const cls = String(v.vehicle_class || "").toLowerCase();
        return tid.includes(term) || plt.includes(term) || cls.includes(term);
      });
    }

    return list;
  }, [vehicles, filterType, searchTerm]);

  const handleExportCSV = () => {
    if (!vehicles.length) return;
    const headers = ["Track ID", "Vehicle Type", "Number Plate", "Helmet Status"];
    const rows = vehicles.map(v => [
      `#${v.track_id}`,
      v.vehicle_class || "Unknown",
      v.plate || "not detected",
      v.helmet_status || "N/A"
    ]);
    const csvContent = [headers.join(","), ...rows.map(r => r.map(x => `"${x}"`).join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `crossing_vehicles_log_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="select-none">
      {/* Title & Actions Block */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5 mb-3 border-b border-sky-border/30 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-sky-default/10 flex items-center justify-center text-sky-default">
            <Compass className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider">
              Itemized Vehicle Log (Crossing Line)
            </h3>
            <p className="text-[9px] text-sky-dark/45 font-mono uppercase">
              Mathematical Vector Gate Crossings
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          {/* Quick Search */}
          <div className="relative flex-1 sm:w-40">
            <Search className="w-3 h-3 text-sky-dark/40 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search vehicle..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-7 pr-2 py-1 text-[11px] rounded-lg bg-white/70 border border-sky-border/40 text-sky-dark placeholder-sky-dark/35 focus:outline-none focus:border-sky-default font-mono transition-colors"
            />
          </div>

          <span className="font-mono text-[10px] font-extrabold text-sky-default bg-sky-surface border border-sky-border/40 px-2.5 py-1 rounded-xl shadow-sm whitespace-nowrap">
            {vehicles.length} COUNTED
          </span>

          {vehicles.length > 0 && (
            <button
              onClick={handleExportCSV}
              title="Download Crossing Log CSV"
              className="p-1 rounded-lg bg-white/60 hover:bg-sky-surface text-sky-dark/70 hover:text-sky-default border border-sky-border/40 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center gap-1.5 mb-3">
        <button
          onClick={() => setFilterType("all")}
          className={`px-2.5 py-1 rounded-lg text-[10px] font-heading font-bold transition-all ${
            filterType === "all"
              ? "bg-sky-default text-white shadow-sm"
              : "bg-white/60 text-sky-dark/70 hover:bg-sky-surface border border-sky-border/30"
          }`}
        >
          All ({vehicles.length})
        </button>
        <button
          onClick={() => setFilterType("twoWheelers")}
          className={`px-2.5 py-1 rounded-lg text-[10px] font-heading font-bold transition-all ${
            filterType === "twoWheelers"
              ? "bg-sky-default text-white shadow-sm"
              : "bg-white/60 text-sky-dark/70 hover:bg-sky-surface border border-sky-border/30"
          }`}
        >
          🏍️ Two-Wheelers
        </button>
        <button
          onClick={() => setFilterType("withPlates")}
          className={`px-2.5 py-1 rounded-lg text-[10px] font-heading font-bold transition-all ${
            filterType === "withPlates"
              ? "bg-sky-default text-white shadow-sm"
              : "bg-white/60 text-sky-dark/70 hover:bg-sky-surface border border-sky-border/30"
          }`}
        >
          🎯 Plates Detected
        </button>
      </div>

      <div className="overflow-hidden rounded-2xl border border-sky-border/30 bg-white/40 shadow-sm">
        {/* Table Header */}
        <div
          className="grid text-[10px] font-heading font-extrabold px-3.5 py-2.5 bg-sky-surface border-b border-sky-border/30 text-sky-dark/75 uppercase tracking-wider"
          style={{ gridTemplateColumns: "1fr 1.6fr 2fr 1.8fr" }}
        >
          <span>Track ID</span>
          <span>Vehicle Type</span>
          <span>Number Plate</span>
          <span className="text-center">Helmet Status</span>
        </div>

        {/* Scrollable rows */}
        <div className="overflow-y-auto max-h-[220px] pr-1 custom-scrollbar">
          {filtered.length === 0 ? (
            <div className="px-4 py-8 text-center text-xs font-heading font-bold text-sky-dark/40 uppercase tracking-wider">
              {searchTerm || filterType !== "all" ? "No matching vehicles" : "Waiting for vehicle crossings..."}
            </div>
          ) : (
            <div className="flex flex-col">
              <AnimatePresence initial={false}>
                {filtered.map((v, i) => {
                  const hasPlate = v.plate && v.plate !== "not detected" && v.plate !== "none";
                  const hStatus = String(v.helmet_status || "N/A");
                  const isViolation = hStatus.toLowerCase().includes("violation") || hStatus.toLowerCase().includes("no helmet");
                  const isCompliant = hStatus.toLowerCase().includes("compliant") || hStatus.toLowerCase().includes("helmet");

                  return (
                    <motion.div
                      key={v.track_id + i}
                      className="grid text-xs px-3.5 py-2.5 items-center border-b border-sky-border/20 transition-all duration-300"
                      style={{
                        gridTemplateColumns: "1fr 1.6fr 2fr 1.8fr",
                        backgroundColor: i % 2 === 0 ? "rgba(255,255,255,0.3)" : "rgba(199, 232, 253, 0.12)",
                      }}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: -15 }}
                      transition={{ type: "spring", stiffness: 350, damping: 25 }}
                    >
                      {/* Track ID */}
                      <span className="font-mono text-sky-dark font-extrabold text-[11px]">
                        #{v.track_id}
                      </span>

                      {/* Vehicle Type */}
                      <div className="flex items-center gap-1.5 overflow-hidden">
                        <span className="text-sm select-none">{getVehicleIcon(v.vehicle_class)}</span>
                        <span className="font-heading font-bold text-[11px] text-sky-dark truncate">
                          {v.vehicle_class || "Unknown"}
                        </span>
                      </div>

                      {/* Plate */}
                      <div className="flex items-center">
                        {hasPlate ? (
                          <div className="inline-flex items-center overflow-hidden rounded border border-gray-300 bg-white text-[10px] font-mono font-bold shadow-xs">
                            <span className="bg-blue-600 text-white text-[6px] px-1 py-0.5 leading-none select-none">
                              IND
                            </span>
                            <span className="px-1.5 py-0.5 font-extrabold text-gray-800">
                              {v.plate}
                            </span>
                          </div>
                        ) : (
                          <span className="text-[10px] font-mono text-sky-dark/40 italic">
                            not detected
                          </span>
                        )}
                      </div>

                      {/* Helmet Status */}
                      <div className="flex items-center justify-center">
                        {isViolation ? (
                          <span className="inline-flex items-center gap-1 text-[9px] font-heading font-extrabold text-red-600 bg-red-50 border border-red-200 px-2 py-0.5 rounded-lg animate-pulse whitespace-nowrap">
                            <AlertTriangle className="w-2.5 h-2.5" /> {hStatus}
                          </span>
                        ) : isCompliant ? (
                          <span className="inline-flex items-center gap-1 text-[9px] font-heading font-extrabold text-emerald-600 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-lg whitespace-nowrap">
                            <ShieldCheck className="w-2.5 h-2.5" /> Compliant
                          </span>
                        ) : (
                          <span className="text-[10px] font-mono text-sky-dark/40 font-medium">
                            N/A
                          </span>
                        )}
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
