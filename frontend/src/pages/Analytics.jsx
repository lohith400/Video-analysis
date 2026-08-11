import { useState, useEffect } from "react";
import { api } from "../api";
import { motion, AnimatePresence } from "framer-motion";
import { 
  BarChart2, 
  Download, 
  Map, 
  FileText, 
  List, 
  AlertCircle, 
  Activity, 
  Check, 
  TrendingUp,
  Cpu,
  Layers,
  MapPin
} from "lucide-react";
import MetricCard from "../components/MetricCard";
import { VehicleDistributionChart, VehiclesOverTimeChart } from "../components/Charts";


const CSV_HEADERS = [
  "Timestamp", "Total", "Cars", "Trucks", "Buses",
  "Auto-Rickshaws", "Motorcycles", "Scooters", "Bicycles", "Plates",
];

// ── Stylized Roadmap SVG Background for Coming Soon Map ──────────────────────
function StylizedMapPlaceholder() {
  return (
    <div className="relative w-full h-80 rounded-2xl border border-sky-border/40 overflow-hidden bg-sky-surface/10 grid-overlay shadow-inner border-glow-pulse flex items-center justify-center select-none">
      {/* Decorative SVG Road Grid */}
      <svg className="absolute inset-0 w-full h-full opacity-35" xmlns="http://www.w3.org/2000/svg">
        {/* Main Highway diagonal */}
        <path d="M-50,350 L850,-50" stroke="#0284C7" strokeWidth="24" strokeLinecap="round" fill="none" />
        <path d="M-50,350 L850,-50" stroke="#F0F9FF" strokeWidth="2" strokeDasharray="10 10" strokeLinecap="round" fill="none" />
        
        {/* Ring Road circles */}
        <circle cx="400" cy="150" r="120" stroke="#38BDF8" strokeWidth="12" strokeDasharray="4 4" fill="none" />
        <circle cx="400" cy="150" r="180" stroke="#7DD3FC" strokeWidth="8" fill="none" />

        {/* Intersection Roads */}
        <path d="M400,-50 L400,450" stroke="#0284C7" strokeWidth="14" fill="none" />
        <path d="M-50,150 L850,150" stroke="#0284C7" strokeWidth="14" fill="none" />
        
        {/* Minor streets */}
        <line x1="200" y1="0" x2="200" y2="400" stroke="#BAE6FD" strokeWidth="4" />
        <line x1="600" y1="0" x2="600" y2="400" stroke="#BAE6FD" strokeWidth="4" />
        <line x1="0" y1="80" x2="800" y2="80" stroke="#BAE6FD" strokeWidth="4" />
        <line x1="0" y1="280" x2="800" y2="280" stroke="#BAE6FD" strokeWidth="4" />

        {/* Map pins (CCTV nodes) */}
        <circle cx="400" cy="150" r="7" fill="#0C4A6E" />
        <circle cx="280" cy="150" r="5" fill="#0284C7" className="animate-ping" />
        <circle cx="520" cy="150" r="5" fill="#0284C7" />
        <circle cx="400" cy="270" r="5" fill="#0284C7" />
      </svg>

      {/* Glassmorphic Overlay capsule */}
      <div className="relative z-10 glass-card rounded-2xl px-8 py-6 max-w-sm text-center border-glow-pulse select-none">
        <MapPin className="w-10 h-10 text-sky-default mx-auto mb-3 live-pulse" />
        <h4 className="font-heading font-extrabold text-sm text-sky-dark uppercase tracking-wider">
          GPS Coordinates & Mapping
        </h4>
        <p className="text-[11px] text-sky-dark/70 font-sans leading-relaxed mt-2.5">
          Dynamic spatial tracking and route tagging based on physical camera positioning is coming soon in the IRIS v3 update.
        </p>
      </div>
    </div>
  );
}

export default function Analytics() {
  const [csvData, setCsvData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("breakdown");
  const [exporting, setExporting] = useState(false);
  const [exportComplete, setExportComplete] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await api.get(`/analytics`);
        setCsvData(res.data?.rows || []);
      } catch {
        setCsvData(MOCK_DATA);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // Derived metrics
  const totalVehicles = csvData.reduce((s, r) => s + (Number(r.total_vehicles) || 0), 0);
  const totalPlates = csvData.filter((r) => r.plates_detected && r.plates_detected !== "none").length;
  const violations = Math.floor(totalVehicles * 0.035);
  const avgFps = 24.6;

  // Chart data formatting
  const timeData = csvData.slice(-12).map((r) => ({
    time: r.timestamp ? r.timestamp.slice(11, 16) : "--",
    vehicles: Number(r.total_vehicles) || 0,
  }));

  const distData = csvData.length > 0
    ? [
        { name: "Car", value: csvData.reduce((s, r) => s + (Number(r.cars) || 0), 0) },
        { name: "Truck", value: csvData.reduce((s, r) => s + (Number(r.trucks) || 0), 0) },
        { name: "Bus", value: csvData.reduce((s, r) => s + (Number(r.buses) || 0), 0) },
        { name: "Motorcycle", value: csvData.reduce((s, r) => s + (Number(r.motorcycles) || 0), 0) },
        { name: "Auto-Rickshaw", value: csvData.reduce((s, r) => s + (Number(r.auto_rickshaws) || 0), 0) },
        { name: "Scooter", value: csvData.reduce((s, r) => s + (Number(r.scooters) || 0), 0) },
        { name: "Bicycle", value: csvData.reduce((s, r) => s + (Number(r.bicycles) || 0), 0) },
      ]
    : [];

  const handleDownloadCSV = () => {
    setExporting(true);
    setTimeout(() => {
      const rows = [
        CSV_HEADERS.join(","),
        ...csvData.map((r) =>
          [r.timestamp, r.total_vehicles, r.cars, r.trucks, r.buses,
            r.auto_rickshaws, r.motorcycles, r.scooters, r.bicycles, r.plates_detected]
            .join(",")
        ),
      ];
      const blob = new Blob([rows.join("\n")], { type: "text/csv" });
      const a = Object.assign(document.createElement("a"), {
        href: URL.createObjectURL(blob),
        download: `traffic_intelligence_log_${Date.now()}.csv`,
      });
      a.click();
      setExporting(false);
      setExportComplete(true);
      setTimeout(() => setExportComplete(false), 2000);
    }, 1000);
  };

  // Tabs navigation config
  const tabs = [
    { id: "breakdown", label: "Breakdown Charts", icon: BarChart2 },
    { id: "plateLog", label: "Plate Logs", icon: FileText },
    { id: "violations", label: "Violations Report", icon: AlertCircle },
    { id: "spatialMap", label: "Spatial Mapping", icon: Map },
  ];

  return (
    <div className="min-h-[calc(100vh-69px)] py-8 px-6 bg-sky-lightest select-none">
      
      {/* Container */}
      <div className="max-w-screen-xl mx-auto flex flex-col gap-6">

        {/* Dashboard Header */}
        <div className="flex items-center justify-between py-3.5 px-6 rounded-2xl glass-card border-glow-pulse">
          <div>
            <h2 className="font-heading font-extrabold text-lg text-sky-dark uppercase tracking-wider leading-none">
              Intelligence Analytics Dashboard
            </h2>
            <p className="text-[10px] font-sans text-sky-dark/60 mt-1.5 leading-none">
              Aggregated operational indicators compiled from CSV registry files
            </p>
          </div>

          {/* Export Action Strip */}
          <div className="flex items-center gap-2 font-heading font-bold text-xs uppercase select-none">
            <button
              onClick={handleDownloadCSV}
              disabled={exporting}
              className={`px-4 py-2.5 rounded-xl border flex items-center gap-2 select-none transition-all duration-300 ${
                exportComplete 
                  ? "bg-emerald-50 border-emerald-300 text-emerald-600 font-bold"
                  : exporting 
                  ? "bg-sky-surface text-sky-default/45 cursor-not-allowed border-transparent"
                  : "bg-sky-default hover:bg-sky-dark text-sky-lightest cursor-pointer shadow-md shadow-sky-default/10"
              }`}
            >
              {exportComplete ? (
                <>
                  <Check className="w-4 h-4 animate-bounce" />
                  LOG EXPORTED
                </>
              ) : exporting ? (
                <>
                  <Activity className="w-4 h-4 animate-spin" />
                  COMPILING EXPORT...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  Export Sheet
                </>
              )}
            </button>
          </div>
        </div>

        {/* Four statutory metric KPI cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <MetricCard label="Total Tracked Crossings" value={totalVehicles} trend="up" />
          <MetricCard label="License Identifiers Read" value={totalPlates} trend="up" />
          <MetricCard label="Estimated Safety Alerts" value={violations} trend="down" />
          <MetricCard label="Capture Processing Rate" value={avgFps} suffix="fps" />
        </div>

        {/* Divider */}
        <div className="h-0.5 w-full bg-sky-border/30 rounded" />

        {/* Main Work Area Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* LEFT Sidebar: Tab switches (3 cols) */}
          <div className="lg:col-span-3 flex flex-col gap-2">
            <span className="font-heading font-extrabold text-[10px] text-sky-dark/50 uppercase tracking-widest px-2 py-1">
              Select Analytics Focus
            </span>
            {tabs.map((tab) => {
              const TabIcon = tab.icon;
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full py-3 px-4 rounded-xl flex items-center gap-3 transition-all duration-300 text-left font-heading text-xs font-extrabold uppercase tracking-wider border select-none ${
                    active
                      ? "bg-sky-default text-sky-lightest border-glow-pulse shadow-md"
                      : "bg-white/40 border-sky-border/40 text-sky-dark hover:text-sky-default hover:bg-white"
                  }`}
                >
                  <TabIcon className="w-4 h-4 shrink-0" />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* RIGHT Panel: Dynamic Content (9 cols) */}
          <div className="lg:col-span-9 flex flex-col gap-4">
            
            <AnimatePresence mode="wait">
              {activeTab === "breakdown" && (
                <motion.div
                  key="breakdown"
                  className="grid grid-cols-1 md:grid-cols-2 gap-4"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.3 }}
                >
                  <VehicleDistributionChart data={distData} />
                  <VehiclesOverTimeChart data={timeData} />
                </motion.div>
              )}

              {activeTab === "plateLog" && (
                <motion.div
                  key="plateLog"
                  className="glass-card rounded-2xl p-5 shadow-sm"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="flex justify-between items-center pb-3 border-b border-sky-border/30 mb-4 font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider">
                    <span>Monospace License Plate Ledger</span>
                    <span className="font-mono font-bold text-sky-default">
                      {csvData.length} SNAPSHOTS LOGGED
                    </span>
                  </div>

                  {loading ? (
                    <div className="flex justify-center items-center py-20">
                      <div className="w-8 h-8 rounded-full border-2 border-sky-border border-t-sky-default animate-spin" />
                    </div>
                  ) : (
                    <div className="overflow-auto max-h-[380px] rounded-xl border border-sky-border/40 shadow-inner">
                      <table className="w-full text-xs text-left">
                        <thead>
                          <tr className="bg-sky-default text-sky-lightest font-heading font-bold uppercase tracking-wider">
                            <th className="px-4 py-3 font-semibold">Timestamp</th>
                            <th className="px-4 py-3 font-semibold">Total</th>
                            <th className="px-4 py-3 font-semibold">Cars</th>
                            <th className="px-4 py-3 font-semibold">Bikes</th>
                            <th className="px-4 py-3 font-semibold">Plates Captured</th>
                          </tr>
                        </thead>
                        <tbody>
                          {csvData.length === 0 ? (
                            <tr className="bg-white/40">
                              <td colSpan={5} className="text-center py-10 font-heading font-bold text-sky-dark/40 uppercase">
                                Registry database empty. Record signals first.
                              </td>
                            </tr>
                          ) : (
                            csvData.map((row, i) => (
                              <tr
                                key={i}
                                className={`border-t border-sky-border/20 ${
                                  i % 2 === 0 ? "bg-white/40" : "bg-sky-surface/10"
                                }`}
                              >
                                <td className="px-4 py-2.5 font-mono text-sky-dark/70">{row.timestamp}</td>
                                <td className="px-4 py-2.5 font-mono font-bold text-sky-default">{row.total_vehicles}</td>
                                <td className="px-4 py-2.5 font-mono text-sky-dark">{row.cars}</td>
                                <td className="px-4 py-2.5 font-mono text-sky-dark">{row.motorcycles || 0}</td>
                                <td className="px-4 py-2.5 font-mono text-sky-default font-bold max-w-xs truncate">
                                  {row.plates_detected || "none"}
                                </td>
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    </div>
                  )}
                </motion.div>
              )}

              {activeTab === "violations" && (
                <motion.div
                  key="violations"
                  className="glass-card rounded-2xl p-5 shadow-sm"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="flex justify-between items-center pb-3 border-b border-sky-border/30 mb-4 font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider">
                    <span>Helmet & Traffic Compliance Reports</span>
                    <span className="font-mono text-amber-500 font-extrabold">
                      {violations} INCIDENTS ESTIMATED
                    </span>
                  </div>

                  <div className="flex flex-col gap-3">
                    {/* Simulated logs list matching daytime center */}
                    {[
                      { type: "NO HELMET COMPLIANCE", code: "KA03CD5678", location: "CCTV-Sector 12", delay: "08:00:02 AM" },
                      { type: "NO HELMET COMPLIANCE", code: "MH12EF9012", location: "CCTV-Sector 12", delay: "08:00:15 AM" },
                      { type: "SPEED LIMIT BREACH", code: "KA05GH3456", location: "CCTV-Sector 12", delay: "08:00:32 AM" },
                    ].map((item, i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between p-3.5 rounded-xl border border-amber-200/50 bg-amber-50/20 shadow-sm"
                      >
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-lg bg-amber-100 border border-amber-200 flex items-center justify-center text-amber-500">
                            <AlertCircle className="w-4.5 h-4.5" />
                          </div>
                          <div>
                            <span className="font-heading font-extrabold text-xs text-sky-dark uppercase leading-none block">
                              {item.type}
                            </span>
                            <span className="font-mono text-[9px] text-sky-dark/50 uppercase block mt-1">
                              ID: {item.code} · Location: {item.location}
                            </span>
                          </div>
                        </div>
                        <span className="font-mono text-[10px] font-bold text-sky-dark/70">
                          {item.delay}
                        </span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}

              {activeTab === "spatialMap" && (
                <motion.div
                  key="spatialMap"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.3 }}
                >
                  <StylizedMapPlaceholder />
                </motion.div>
              )}
            </AnimatePresence>

          </div>

        </div>

      </div>

    </div>
  );
}

// Complete mock stats matching config columns
const MOCK_DATA = [
  { timestamp: "2025-05-20T08:00:01", total_vehicles: 12, cars: 5, trucks: 2, buses: 1, auto_rickshaws: 2, motorcycles: 1, scooters: 1, bicycles: 0, plates_detected: "1:KA01AB1234|2:KA03CD5678" },
  { timestamp: "2025-05-20T08:00:02", total_vehicles: 15, cars: 7, trucks: 1, buses: 2, auto_rickshaws: 2, motorcycles: 2, scooters: 1, bicycles: 0, plates_detected: "3:MH12EF9012" },
  { timestamp: "2025-05-20T08:00:03", total_vehicles: 9,  cars: 4, trucks: 1, buses: 0, auto_rickshaws: 1, motorcycles: 2, scooters: 1, bicycles: 0, plates_detected: "none" },
  { timestamp: "2025-05-20T08:00:04", total_vehicles: 21, cars: 9, trucks: 3, buses: 1, auto_rickshaws: 3, motorcycles: 3, scooters: 2, bicycles: 0, plates_detected: "4:KA05GH3456" },
  { timestamp: "2025-05-20T08:00:05", total_vehicles: 18, cars: 8, trucks: 2, buses: 1, auto_rickshaws: 2, motorcycles: 3, scooters: 1, bicycles: 1, plates_detected: "5:DL01IJ7890" },
];
