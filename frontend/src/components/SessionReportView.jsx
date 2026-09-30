import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Check, 
  Car, 
  CreditCard, 
  ShieldCheck, 
  Users, 
  Compass, 
  Download, 
  Share2, 
  Copy, 
  Sparkles,
  Award,
  AlertTriangle
} from "lucide-react";
import PlateTable from "./PlateTable";
import TwoWheelerSafetyTable from "./TwoWheelerSafetyTable";
import PedestrianDemographics from "./PedestrianDemographics";
import CrossingVehiclesTable from "./CrossingVehiclesTable";

export default function SessionReportView({ report = {}, onBack = null }) {
  const [activeTab, setActiveTab] = useState("overview"); // "overview", "plates", "twoWheelers", "crossing", "demographics"
  const [copied, setCopied] = useState(false);

  const {
    generatedAt = new Date().toISOString(),
    vehicleCounts = {},
    humans = { total: 0, males: 0, females: 0, children: 0, unknown: 0 },
    plates = [],
    twoWheelers = [],
    perVehicle = []
  } = report;

  const totalVehicles = vehicleCounts.total || vehicleCounts.TOTAL || 
    Object.entries(vehicleCounts).reduce((acc, [k, v]) => k.toLowerCase() !== "total" ? acc + Number(v) : acc, 0);

  const vehicleRows = Object.entries(vehicleCounts).filter(
    ([k]) => k.toLowerCase() !== "total" && Number(vehicleCounts[k]) >= 0
  );

  const violationCount = twoWheelers.filter(t => {
    const v = String(t.verdict || "").toLowerCase();
    const r = String(t.rider_helmet || "").toLowerCase();
    const p = String(t.pillion_helmet || "").toLowerCase();
    return v.includes("violation") || r === "no_helmet" || p === "no_helmet";
  }).length;

  const handleCopyReport = () => {
    const summaryText = `===== IRIS SESSION REPORT =====
Generated: ${generatedAt}
Total Counted Vehicles: ${totalVehicles}
Plates Recognized: ${plates.length}
Two-Wheelers: ${twoWheelers.length} (Violations: ${violationCount})
Pedestrians Logged: ${humans.total} (Males: ${humans.males}, Females: ${humans.females}, Children: ${humans.children})
===============================`;
    navigator.clipboard.writeText(summaryText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadFullCSV = () => {
    let csv = `===== FINAL SESSION REPORT =====\ngenerated_at,${generatedAt}\n\n`;

    csv += `-- Vehicle counts by class --\nvehicle_class,count\n`;
    vehicleRows.forEach(([cls, cnt]) => {
      csv += `${cls},${cnt}\n`;
    });
    csv += `TOTAL,${totalVehicles}\n\n`;

    csv += `-- Humans detected --\ncategory,count\n`;
    Object.entries(humans).forEach(([k, v]) => {
      csv += `${k},${v}\n`;
    });
    csv += `\n`;

    if (plates.length > 0) {
      csv += `-- Recognized license plates detail --\ntrack_id,vehicle_class,plate_number,confidence\n`;
      plates.forEach(p => {
        csv += `${p.track_id || ""},${p.vehicle_class || "Car"},${p.plate || ""},${p.confidence || ""}\n`;
      });
      csv += `\n`;
    }

    if (twoWheelers.length > 0) {
      csv += `-- Two-wheeler helmet compliance detail --\ntrack_id,vehicle_class,plate_number,rider_helmet,pillion_helmet,verdict\n`;
      twoWheelers.forEach(t => {
        csv += `${t.track_id || ""},${t.vehicle_class || "Bike/Motorcycle"},${t.plate || "not detected"},${t.rider_helmet || ""},${t.pillion_helmet || ""},${t.verdict || ""}\n`;
      });
      csv += `\n`;
    }

    if (perVehicle.length > 0) {
      csv += `-- Per-vehicle detail (every vehicle counted this run) --\ntrack_id,vehicle_class,plate_number,helmet_status\n`;
      perVehicle.forEach(v => {
        csv += `${v.track_id || ""},${v.vehicle_class || ""},${v.plate || "not detected"},${v.helmet_status || "N/A"}\n`;
      });
      csv += `\n`;
    }

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `iris_final_session_report_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const tabs = [
    { id: "overview", label: "Overview & Vehicles", icon: Car, count: totalVehicles },
    { id: "plates", label: "Recognized Plates", icon: CreditCard, count: plates.length },
    { id: "twoWheelers", label: "Two-Wheeler Helmet Report", icon: ShieldCheck, count: twoWheelers.length },
    { id: "crossing", label: "Crossing Log", icon: Compass, count: perVehicle.length },
    { id: "demographics", label: "Demographics", icon: Users, count: humans.total }
  ];

  return (
    <div className="flex flex-col gap-4 select-none">
      {/* Top Banner Card */}
      <div className="glass-card rounded-2xl p-5 border-glow-pulse shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-600 shadow-inner">
            <Check className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-heading font-extrabold text-base text-sky-dark uppercase tracking-wider">
                Final Session Intelligence Report
              </h2>
              <span className="px-2 py-0.5 rounded-full text-[9px] font-heading font-extrabold bg-emerald-100 text-emerald-700 uppercase border border-emerald-200">
                Verified Run
              </span>
            </div>
            <p className="text-[10px] text-sky-dark/55 font-mono mt-0.5">
              Generated: {generatedAt} • Comprehensive Multimodal Traffic Analysis
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2 self-stretch md:self-auto">
          {onBack && (
            <button
              onClick={onBack}
              className="px-3 py-2 rounded-xl text-xs font-heading font-bold bg-white/70 hover:bg-sky-surface text-sky-dark border border-sky-border/40 transition-colors"
            >
              Back to Stream
            </button>
          )}
          <button
            onClick={handleCopyReport}
            className="px-3.5 py-2 rounded-xl text-xs font-heading font-bold bg-white/70 hover:bg-sky-surface text-sky-dark border border-sky-border/40 transition-colors flex items-center gap-1.5 shadow-sm"
          >
            <Copy className="w-3.5 h-3.5" />
            {copied ? "Copied!" : "Copy Summary"}
          </button>
          <button
            onClick={handleDownloadFullCSV}
            className="px-4 py-2 rounded-xl text-xs font-heading font-extrabold bg-sky-default hover:bg-sky-dark text-white transition-all shadow-md shadow-sky-default/15 flex items-center gap-1.5"
          >
            <Download className="w-3.5 h-3.5" />
            Download Complete CSV
          </button>
        </div>
      </div>

      {/* KPI Metric Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="glass-card rounded-xl p-3.5 border border-sky-border/40 shadow-sm flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-sky-surface/50 flex items-center justify-center text-xl">
            🚗
          </div>
          <div>
            <span className="font-mono text-xl font-extrabold text-sky-dark leading-none block">
              {totalVehicles}
            </span>
            <span className="text-[10px] font-heading font-extrabold text-sky-dark/50 uppercase mt-0.5 block">
              Total Counted
            </span>
          </div>
        </div>

        <div className="glass-card rounded-xl p-3.5 border border-sky-border/40 shadow-sm flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-sky-surface/50 flex items-center justify-center text-xl">
            🎯
          </div>
          <div>
            <span className="font-mono text-xl font-extrabold text-sky-default leading-none block">
              {plates.length}
            </span>
            <span className="text-[10px] font-heading font-extrabold text-sky-dark/50 uppercase mt-0.5 block">
              Plates Detected
            </span>
          </div>
        </div>

        <div className="glass-card rounded-xl p-3.5 border border-sky-border/40 shadow-sm flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-red-50 flex items-center justify-center text-xl">
            🏍️
          </div>
          <div>
            <span className="font-mono text-xl font-extrabold text-red-600 leading-none block">
              {violationCount}
            </span>
            <span className="text-[10px] font-heading font-extrabold text-sky-dark/50 uppercase mt-0.5 block">
              Helmet Violations
            </span>
          </div>
        </div>

        <div className="glass-card rounded-xl p-3.5 border border-sky-border/40 shadow-sm flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-purple-50 flex items-center justify-center text-xl">
            🚶
          </div>
          <div>
            <span className="font-mono text-xl font-extrabold text-purple-600 leading-none block">
              {humans.total}
            </span>
            <span className="text-[10px] font-heading font-extrabold text-sky-dark/50 uppercase mt-0.5 block">
              Pedestrians Logged
            </span>
          </div>
        </div>
      </div>

      {/* Tab Navigation Pill Bar */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 custom-scrollbar">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3.5 py-2 rounded-xl text-xs font-heading font-extrabold uppercase tracking-wider flex items-center gap-2 transition-all whitespace-nowrap border ${
                isActive
                  ? "bg-sky-default text-white border-sky-default shadow-md shadow-sky-default/10"
                  : "bg-white/60 text-sky-dark/70 hover:bg-white border-sky-border/30"
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
              <span className={`px-1.5 py-0.2 rounded-md font-mono text-[9px] ${
                isActive ? "bg-white/20 text-white" : "bg-sky-surface text-sky-dark/60"
              }`}>
                {tab.count}
              </span>
            </button>
          );
        })}
      </div>

      {/* Tab Content Panes */}
      <div className="glass-card rounded-2xl p-5 shadow-sm border border-sky-border/40">
        <AnimatePresence mode="wait">
          {/* TAB 1: OVERVIEW & VEHICLE BREAKDOWN */}
          {activeTab === "overview" && (
            <motion.div
              key="overview"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              className="flex flex-col gap-4"
            >
              <div className="flex items-center justify-between border-b border-sky-border/30 pb-3">
                <h3 className="font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider">
                  Vehicle Classification Counts
                </h3>
                <span className="font-mono text-xs font-bold text-sky-default">
                  {totalVehicles} CUMULATIVE VEHICLES
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {vehicleRows.map(([cls, cnt]) => {
                  const count = Number(cnt);
                  const pct = totalVehicles > 0 ? Math.round((count / totalVehicles) * 100) : 0;
                  const icon = cls.toLowerCase().includes("bike") ? "🏍️" :
                               cls.toLowerCase().includes("car") ? "🚗" :
                               cls.toLowerCase().includes("auto") ? "🛺" :
                               cls.toLowerCase().includes("truck") ? "🚛" :
                               cls.toLowerCase().includes("bus") ? "🚌" :
                               cls.toLowerCase().includes("bicycle") ? "🚲" : "🚐";

                  return (
                    <div
                      key={cls}
                      className="p-3.5 rounded-xl bg-white/50 border border-sky-border/30 shadow-xs flex flex-col justify-between"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-2xl">{icon}</span>
                        <span className="px-1.5 py-0.5 rounded font-mono text-[9px] font-extrabold bg-sky-surface/60 text-sky-default">
                          {pct}%
                        </span>
                      </div>
                      <div>
                        <span className="font-mono text-xl font-extrabold text-sky-dark leading-none block">
                          {count}
                        </span>
                        <span className="text-[10px] font-heading font-extrabold text-sky-dark/60 uppercase tracking-wide mt-1 block">
                          {cls}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </motion.div>
          )}

          {/* TAB 2: ALL RECOGNIZED PLATES */}
          {activeTab === "plates" && (
            <motion.div
              key="plates"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
            >
              <PlateTable plates={plates} />
            </motion.div>
          )}

          {/* TAB 3: TWO-WHEELER HELMET REPORT */}
          {activeTab === "twoWheelers" && (
            <motion.div
              key="twoWheelers"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
            >
              <TwoWheelerSafetyTable statuses={twoWheelers} />
            </motion.div>
          )}

          {/* TAB 4: CROSSING LOG */}
          {activeTab === "crossing" && (
            <motion.div
              key="crossing"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
            >
              <CrossingVehiclesTable vehicles={perVehicle} />
            </motion.div>
          )}

          {/* TAB 5: DEMOGRAPHICS */}
          {activeTab === "demographics" && (
            <motion.div
              key="demographics"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
            >
              <PedestrianDemographics data={humans} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
