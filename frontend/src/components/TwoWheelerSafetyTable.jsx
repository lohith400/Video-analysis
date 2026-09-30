import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  ShieldCheck, 
  ShieldAlert, 
  CheckCircle, 
  AlertTriangle, 
  MinusCircle, 
  Search, 
  Download, 
  Filter 
} from "lucide-react";

export default function TwoWheelerSafetyTable({ statuses = [] }) {
  const [filterMode, setFilterMode] = useState("all"); // "all", "violations", "compliant"
  const [searchTerm, setSearchTerm] = useState("");

  // Determine if a status entry is a violation
  const isViolation = (s) => {
    const r = String(s.rider_helmet || "").toLowerCase();
    const p = String(s.pillion_helmet || "").toLowerCase();
    const v = String(s.verdict || "").toLowerCase();
    return s.has_violation || r === "no_helmet" || p === "no_helmet" || v.includes("violation");
  };

  // Metrics
  const totalCount = statuses.length;
  const violationCount = statuses.filter(isViolation).length;
  const compliantCount = totalCount - violationCount;
  const complianceRate = totalCount > 0 ? Math.round((compliantCount / totalCount) * 100) : 100;

  // Filtered & Sorted list (newest track_id first)
  const filteredStatuses = useMemo(() => {
    let result = [...statuses].sort((a, b) => (Number(b.track_id) || 0) - (Number(a.track_id) || 0));

    if (filterMode === "violations") {
      result = result.filter(isViolation);
    } else if (filterMode === "compliant") {
      result = result.filter((s) => !isViolation(s));
    }

    if (searchTerm.trim()) {
      const term = searchTerm.toLowerCase().trim();
      result = result.filter((s) => {
        const tid = String(s.track_id || "").toLowerCase();
        const plt = String(s.plate || "").toLowerCase();
        const vcls = String(s.vehicle_class || "").toLowerCase();
        return tid.includes(term) || plt.includes(term) || vcls.includes(term);
      });
    }

    return result;
  }, [statuses, filterMode, searchTerm]);

  const handleExportCSV = () => {
    if (!statuses.length) return;
    const headers = ["Track ID", "Vehicle Type", "Number Plate", "Rider Helmet", "Pillion Helmet", "Verdict"];
    const rows = statuses.map((s) => {
      const viol = isViolation(s);
      return [
        `#${s.track_id}`,
        s.vehicle_class || "Bike/Motorcycle",
        s.plate || "not detected",
        s.rider_helmet === "no_helmet" ? "NO HELMET" : "HELMET",
        s.pillion_helmet === "no_helmet" ? "NO HELMET" : s.pillion_helmet === "helmet" ? "HELMET" : "None",
        viol ? "VIOLATION" : "COMPLIANT"
      ];
    });
    const csvContent = [headers.join(","), ...rows.map((r) => r.map((x) => `"${x}"`).join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `two_wheeler_safety_report_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="select-none">
      {/* Table Title Block */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5 mb-3 border-b border-sky-border/30 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-sky-default/10 flex items-center justify-center text-sky-default">
            <span className="text-sm">🏍️</span>
          </div>
          <div>
            <h3 className="font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider">
              Two-Wheeler Helmet Compliance
            </h3>
            <p className="text-[9px] text-sky-dark/45 font-mono uppercase">
              Rider & Pillion Safety Enforcement Feed
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          {/* Search Input */}
          <div className="relative flex-1 sm:w-40">
            <Search className="w-3 h-3 text-sky-dark/40 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search ID / Plate..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-7 pr-2 py-1 text-[11px] rounded-lg bg-white/70 border border-sky-border/40 text-sky-dark placeholder-sky-dark/35 focus:outline-none focus:border-sky-default font-mono transition-colors"
            />
          </div>

          {statuses.length > 0 && (
            <button
              onClick={handleExportCSV}
              title="Download Safety Report CSV"
              className="p-1 rounded-lg bg-white/60 hover:bg-sky-surface text-sky-dark/70 hover:text-sky-default border border-sky-border/40 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Compliance Meter & Filter Tabs */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-2 mb-3 bg-white/40 p-2 rounded-xl border border-sky-border/30">
        {/* Compliance Rate Progress Bar */}
        <div className="flex items-center gap-2.5 flex-1 min-w-[200px]">
          <span className="text-[10px] font-heading font-extrabold text-sky-dark/70 uppercase">
            Compliance
          </span>
          <div className="flex-1 h-2 rounded-full bg-red-100 overflow-hidden relative">
            <div
              className="h-full bg-emerald-500 rounded-full transition-all duration-500"
              style={{ width: `${complianceRate}%` }}
            />
          </div>
          <span className="font-mono text-[10px] font-extrabold text-sky-default">
            {complianceRate}%
          </span>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-1 self-end sm:self-auto">
          <button
            onClick={() => setFilterMode("all")}
            className={`px-2.5 py-1 rounded-lg text-[10px] font-heading font-bold transition-all ${
              filterMode === "all"
                ? "bg-sky-default text-white shadow-sm"
                : "bg-white/60 text-sky-dark/70 hover:bg-sky-surface border border-sky-border/30"
            }`}
          >
            All ({totalCount})
          </button>
          <button
            onClick={() => setFilterMode("violations")}
            className={`px-2.5 py-1 rounded-lg text-[10px] font-heading font-bold transition-all flex items-center gap-1 ${
              filterMode === "violations"
                ? "bg-red-500 text-white shadow-sm"
                : "bg-red-50/50 text-red-600 hover:bg-red-100/60 border border-red-200/50"
            }`}
          >
            Violations ({violationCount})
          </button>
          <button
            onClick={() => setFilterMode("compliant")}
            className={`px-2.5 py-1 rounded-lg text-[10px] font-heading font-bold transition-all flex items-center gap-1 ${
              filterMode === "compliant"
                ? "bg-emerald-600 text-white shadow-sm"
                : "bg-emerald-50/50 text-emerald-600 hover:bg-emerald-100/60 border border-emerald-200/50"
            }`}
          >
            Compliant ({compliantCount})
          </button>
        </div>
      </div>

      <div className="overflow-hidden rounded-2xl border border-sky-border/30 bg-white/40 shadow-sm">
        {/* Table Header */}
        <div
          className="grid text-[10px] font-heading font-extrabold px-3.5 py-2.5 bg-sky-surface border-b border-sky-border/30 text-sky-dark/75 uppercase tracking-wider"
          style={{ gridTemplateColumns: "0.8fr 1.6fr 1.2fr 1.2fr 1.8fr" }}
        >
          <span>Track ID</span>
          <span>Number Plate</span>
          <span className="text-center flex items-center justify-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5 text-sky-default/60" /> Rider
          </span>
          <span className="text-center flex items-center justify-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5 text-sky-default/60" /> Pillion
          </span>
          <span className="text-center">Verdict</span>
        </div>

        {/* Scrollable rows */}
        <div className="overflow-y-auto max-h-[230px] pr-1 custom-scrollbar">
          {filteredStatuses.length === 0 ? (
            <div className="px-4 py-8 text-center text-xs font-heading font-bold text-sky-dark/40 uppercase tracking-wider">
              {searchTerm || filterMode !== "all"
                ? "No matching two-wheelers found"
                : "Waiting for two-wheelers..."}
            </div>
          ) : (
            <div className="flex flex-col">
              <AnimatePresence initial={false}>
                {filteredStatuses.map((s, i) => {
                  const isNewest = i === 0 && filterMode === "all" && !searchTerm;
                  const viol = isViolation(s);

                  // Rider status badge
                  const rawRider = String(s.rider_helmet || "").toLowerCase();
                  let riderBadge = (
                    <span className="inline-flex items-center gap-1 font-mono text-[9px] font-extrabold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-200">
                      <CheckCircle className="w-2.5 h-2.5" /> HELMET
                    </span>
                  );
                  if (rawRider === "no_helmet") {
                    riderBadge = (
                      <span className="inline-flex items-center gap-1 font-mono text-[9px] font-extrabold px-2 py-0.5 rounded-full bg-red-50 text-red-600 border border-red-200 animate-pulse">
                        <AlertTriangle className="w-2.5 h-2.5" /> NO HELMET
                      </span>
                    );
                  } else if (rawRider === "unknown") {
                    riderBadge = (
                      <span className="inline-flex items-center gap-1 font-mono text-[9px] font-bold px-2 py-0.5 rounded-full bg-gray-50 text-gray-500 border border-gray-200">
                        <MinusCircle className="w-2.5 h-2.5" /> Checking
                      </span>
                    );
                  }

                  // Pillion status badge
                  const rawPillion = String(s.pillion_helmet || "").toLowerCase();
                  let pillionBadge = (
                    <span className="inline-flex items-center gap-1 font-mono text-[9px] font-medium px-2 py-0.5 rounded-full bg-sky-50/40 text-sky-dark/40 border border-sky-100">
                      None
                    </span>
                  );
                  if (rawPillion === "no_helmet") {
                    pillionBadge = (
                      <span className="inline-flex items-center gap-1 font-mono text-[9px] font-extrabold px-2 py-0.5 rounded-full bg-red-50 text-red-600 border border-red-200 animate-pulse">
                        <AlertTriangle className="w-2.5 h-2.5" /> NO HELMET
                      </span>
                    );
                  } else if (rawPillion === "helmet") {
                    pillionBadge = (
                      <span className="inline-flex items-center gap-1 font-mono text-[9px] font-extrabold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-200">
                        <CheckCircle className="w-2.5 h-2.5" /> HELMET
                      </span>
                    );
                  }

                  // Verdict badge
                  let verdictBadge = (
                    <span className="inline-flex items-center gap-1 text-[10px] font-heading font-extrabold text-emerald-600 bg-emerald-50/70 border border-emerald-200 px-2 py-0.5 rounded-lg">
                      ✅ COMPLIANT
                    </span>
                  );
                  if (viol) {
                    let text = "❌ VIOLATION";
                    if (rawRider === "no_helmet" && rawPillion === "no_helmet") {
                      text = "❌ RIDER + PILLION";
                    } else if (rawRider === "no_helmet") {
                      text = "❌ RIDER NO HELMET";
                    } else if (rawPillion === "no_helmet") {
                      text = "❌ PILLION NO HELMET";
                    }
                    verdictBadge = (
                      <span className="inline-flex items-center gap-1 text-[9px] font-heading font-extrabold text-red-600 bg-red-50 border border-red-200 px-2 py-0.5 rounded-lg animate-pulse whitespace-nowrap">
                        {text}
                      </span>
                    );
                  }

                  const hasPlate = s.plate && s.plate !== "not detected" && s.plate !== "UNKNOWN";

                  return (
                    <motion.div
                      key={s.track_id}
                      className="grid text-xs px-3.5 py-2.5 items-center border-b border-sky-border/20 transition-all duration-300"
                      style={{
                        gridTemplateColumns: "0.8fr 1.6fr 1.2fr 1.2fr 1.8fr",
                        backgroundColor: isNewest 
                          ? "rgba(2, 132, 199, 0.06)" 
                          : viol 
                          ? "rgba(239, 68, 68, 0.03)" 
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
                      <span className="font-mono text-sky-dark font-extrabold text-[11px]">
                        #{s.track_id}
                      </span>

                      {/* Plate Badge or 'not detected' */}
                      <div className="flex items-center">
                        {hasPlate ? (
                          <div className="inline-flex items-center overflow-hidden rounded border border-gray-300 bg-white text-[10px] font-mono font-bold shadow-xs">
                            <span className="bg-blue-600 text-white text-[6px] px-1 py-0.5 leading-none select-none">
                              IND
                            </span>
                            <span className="px-1.5 py-0.5 font-extrabold text-gray-800">
                              {s.plate}
                            </span>
                          </div>
                        ) : (
                          <span className="text-[10px] font-mono text-sky-dark/40 italic">
                            not detected
                          </span>
                        )}
                      </div>

                      {/* Rider Status */}
                      <div className="flex items-center justify-center">
                        {riderBadge}
                      </div>

                      {/* Pillion Status */}
                      <div className="flex items-center justify-center">
                        {pillionBadge}
                      </div>

                      {/* Verdict */}
                      <div className="flex items-center justify-center">
                        {verdictBadge}
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
