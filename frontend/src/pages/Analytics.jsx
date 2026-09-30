import { useState, useEffect, useMemo, useRef } from "react";
import { api } from "../api";
import { motion, AnimatePresence } from "framer-motion";
import { 
  BarChart2, 
  Download, 
  Map, 
  FileText, 
  AlertCircle, 
  Activity, 
  Check, 
  Upload, 
  Compass, 
  CreditCard, 
  ShieldCheck, 
  Users, 
  MapPin,
  RefreshCw,
  Search
} from "lucide-react";
import MetricCard from "../components/MetricCard";
import { VehicleDistributionChart, VehiclesOverTimeChart } from "../components/Charts";
import SessionReportView from "../components/SessionReportView";
import PlateTable from "../components/PlateTable";
import TwoWheelerSafetyTable from "../components/TwoWheelerSafetyTable";
import PedestrianDemographics from "../components/PedestrianDemographics";
import CrossingVehiclesTable from "../components/CrossingVehiclesTable";
import { parseTrafficLogCSV, parseAnalyticsRows } from "../utils/csvReportParser";

// ── Stylized Roadmap SVG Background for Coming Soon Map ──────────────────────
function StylizedMapPlaceholder() {
  return (
    <div className="relative w-full h-80 rounded-2xl border border-sky-border/40 overflow-hidden bg-sky-surface/10 grid-overlay shadow-inner border-glow-pulse flex items-center justify-center select-none">
      <svg className="absolute inset-0 w-full h-full opacity-35" xmlns="http://www.w3.org/2000/svg">
        <path d="M-50,350 L850,-50" stroke="#0284C7" strokeWidth="24" strokeLinecap="round" fill="none" />
        <path d="M-50,350 L850,-50" stroke="#F0F9FF" strokeWidth="2" strokeDasharray="10 10" strokeLinecap="round" fill="none" />
        <circle cx="400" cy="150" r="120" stroke="#38BDF8" strokeWidth="12" strokeDasharray="4 4" fill="none" />
        <circle cx="400" cy="150" r="180" stroke="#7DD3FC" strokeWidth="8" fill="none" />
        <path d="M400,-50 L400,450" stroke="#0284C7" strokeWidth="14" fill="none" />
        <path d="M-50,150 L850,150" stroke="#0284C7" strokeWidth="14" fill="none" />
        <line x1="200" y1="0" x2="200" y2="400" stroke="#BAE6FD" strokeWidth="4" />
        <line x1="600" y1="0" x2="600" y2="400" stroke="#BAE6FD" strokeWidth="4" />
        <circle cx="400" cy="150" r="7" fill="#0C4A6E" />
        <circle cx="280" cy="150" r="5" fill="#0284C7" className="animate-ping" />
        <circle cx="520" cy="150" r="5" fill="#0284C7" />
        <circle cx="400" cy="270" r="5" fill="#0284C7" />
      </svg>
      <div className="relative z-10 glass-card rounded-2xl px-8 py-6 max-w-sm text-center border-glow-pulse select-none">
        <MapPin className="w-10 h-10 text-sky-default mx-auto mb-3 live-pulse" />
        <h4 className="font-heading font-extrabold text-sm text-sky-dark uppercase tracking-wider">
          GPS Coordinates & Camera Mapping
        </h4>
        <p className="text-[11px] text-sky-dark/70 font-sans leading-relaxed mt-2.5">
          Dynamic spatial tracking and route tagging based on physical camera positioning is coming in IRIS v3.
        </p>
      </div>
    </div>
  );
}

export default function Analytics() {
  const [csvData, setCsvData] = useState([]);
  const [sessionReport, setSessionReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("sessionReport");
  const [exporting, setExporting] = useState(false);
  const [exportComplete, setExportComplete] = useState(false);
  const [customFileLoaded, setCustomFileLoaded] = useState(false);
  const fileInputRef = useRef(null);

  // Fetch /analytics data from backend
  const fetchBackendData = async () => {
    setLoading(true);
    try {
      const res = await api.get(`/analytics`);
      const rows = res.data?.rows || [];
      setCsvData(rows);

      // Check if rows contains report sections or parse rows
      const parsedReport = parseAnalyticsRows(rows);
      if (parsedReport) {
        setSessionReport(parsedReport);
      }
      setCustomFileLoaded(false);
    } catch {
      // Fallback mock
      const mockRows = MOCK_DATA;
      setCsvData(mockRows);
      setSessionReport(parseAnalyticsRows(mockRows));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBackendData();
  }, []);

  // Handle direct file upload of traffic_log.csv
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result;
      if (typeof text === "string") {
        const { timeSeries, report } = parseTrafficLogCSV(text);
        if (timeSeries.length > 0) {
          setCsvData(timeSeries);
        }
        if (report) {
          setSessionReport(report);
          setActiveTab("sessionReport");
        } else {
          setSessionReport(parseAnalyticsRows(timeSeries));
        }
        setCustomFileLoaded(true);
      }
    };
    reader.readAsText(file);
  };

  // Derived metrics
  const totalVehicles = sessionReport?.vehicleCounts?.total ||
    (csvData.length > 0 ? Number(csvData[csvData.length - 1]?.total_vehicles) || 0 : 0);

  const totalPlates = sessionReport?.plates?.length ||
    (csvData.filter(r => r.plates_detected && r.plates_detected !== "none").length);

  const totalViolations = sessionReport?.twoWheelers?.filter(t => {
    const v = String(t.verdict || "").toLowerCase();
    return v.includes("violation");
  }).length || 0;

  const totalPedestrians = sessionReport?.humans?.total || 
    (csvData.length > 0 ? Number(csvData[csvData.length - 1]?.pedestrians_detected) || 0 : 0);

  // Chart data formatting
  const timeData = useMemo(() => {
    return csvData.slice(-15).map(r => ({
      time: r.timestamp ? r.timestamp.slice(11, 19) : "--",
      vehicles: Number(r.total_vehicles) || 0,
      cars: Number(r.cars) || 0,
      bikes: (Number(r.motorcycles) || 0) + (Number(r.scooters) || 0)
    }));
  }, [csvData]);

  const distData = useMemo(() => {
    if (sessionReport?.vehicleCounts && Object.keys(sessionReport.vehicleCounts).length > 0) {
      return Object.entries(sessionReport.vehicleCounts)
        .filter(([k]) => k.toLowerCase() !== "total")
        .map(([name, val]) => ({ name, value: Number(val) || 0 }));
    }
    if (csvData.length > 0) {
      const last = csvData[csvData.length - 1];
      return [
        { name: "Car", value: Number(last.cars) || 0 },
        { name: "Bike/Motorcycle", value: (Number(last.motorcycles) || 0) + (Number(last.scooters) || 0) },
        { name: "Auto Rickshaw", value: Number(last.auto_rickshaws) || 0 },
        { name: "Truck", value: Number(last.trucks) || 0 },
        { name: "Bus", value: Number(last.buses) || 0 },
        { name: "Bicycle", value: Number(last.bicycles) || 0 }
      ];
    }
    return [];
  }, [sessionReport, csvData]);

  const handleDownloadCSV = () => {
    setExporting(true);
    setTimeout(() => {
      const headers = [
        "timestamp", "total_vehicles", "cars", "trucks", "buses",
        "auto_rickshaws", "motorcycles", "scooters", "bicycles", "plates_detected"
      ];
      const rows = [
        headers.join(","),
        ...csvData.map(r =>
          [r.timestamp, r.total_vehicles, r.cars, r.trucks, r.buses,
            r.auto_rickshaws, r.motorcycles, r.scooters, r.bicycles, r.plates_detected]
            .join(",")
        )
      ];
      const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8;" });
      const a = Object.assign(document.createElement("a"), {
        href: URL.createObjectURL(blob),
        download: `traffic_log_export_${Date.now()}.csv`
      });
      a.click();
      setExporting(false);
      setExportComplete(true);
      setTimeout(() => setExportComplete(false), 2000);
    }, 600);
  };

  // Tabs navigation config
  const tabs = [
    { id: "sessionReport", label: "Executive Session Report", icon: Check, count: sessionReport ? "Ready" : null },
    { id: "breakdown", label: "Traffic Flow Charts", icon: BarChart2 },
    { id: "plates", label: "Recognized Plates Ledger", icon: CreditCard, count: sessionReport?.plates?.length },
    { id: "twoWheelers", label: "Two-Wheeler Safety Report", icon: ShieldCheck, count: sessionReport?.twoWheelers?.length },
    { id: "demographics", label: "Pedestrian Demographics", icon: Users, count: totalPedestrians },
    { id: "crossing", label: "Itemized Crossing Log", icon: Compass, count: sessionReport?.perVehicle?.length },
    { id: "rawCsv", label: "Raw Snapshot Ledger", icon: FileText, count: csvData.length },
    { id: "spatialMap", label: "Spatial Mapping", icon: Map }
  ];

  return (
    <div className="min-h-[calc(100vh-69px)] py-6 px-6 bg-sky-lightest select-none">
      <div className="max-w-screen-xl mx-auto flex flex-col gap-6">

        {/* Dashboard Header Bar */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 py-3.5 px-6 rounded-2xl glass-card border-glow-pulse">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-heading font-extrabold text-lg text-sky-dark uppercase tracking-wider leading-none">
                Intelligence Analytics & Session Report
              </h2>
              {customFileLoaded && (
                <span className="px-2 py-0.5 rounded-full text-[9px] font-heading font-extrabold bg-blue-100 text-blue-700 uppercase border border-blue-200">
                  Custom CSV Loaded
                </span>
              )}
            </div>
            <p className="text-[10px] font-sans text-sky-dark/60 mt-1.5 leading-none">
              Operational KPIs, Number Plates, Helmet Violations, and Demographic Intelligence
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2 font-heading font-bold text-xs uppercase select-none w-full md:w-auto justify-end">
            <input
              type="file"
              accept=".csv"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-3.5 py-2.5 rounded-xl border border-sky-border/40 bg-white/70 hover:bg-sky-surface text-sky-dark flex items-center gap-1.5 transition-colors shadow-sm"
              title="Upload any traffic_log.csv file from your computer"
            >
              <Upload className="w-4 h-4 text-sky-default" />
              Upload Log CSV
            </button>

            <button
              onClick={fetchBackendData}
              className="p-2.5 rounded-xl border border-sky-border/40 bg-white/70 hover:bg-sky-surface text-sky-dark transition-colors shadow-sm"
              title="Reload from backend traffic_log.csv"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            </button>

            <button
              onClick={handleDownloadCSV}
              disabled={exporting || csvData.length === 0}
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

        {/* Four Statutory Metric KPI Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard label="Total Tracked Vehicles" value={totalVehicles} trend="up" />
          <MetricCard label="Unique Plates Identified" value={totalPlates} trend="up" />
          <MetricCard label="Helmet Safety Violations" value={totalViolations} trend="down" />
          <MetricCard label="Pedestrians Logged" value={totalPedestrians} trend="up" />
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
                  className={`w-full py-3 px-3.5 rounded-xl flex items-center justify-between transition-all duration-300 text-left font-heading text-xs font-extrabold uppercase tracking-wider border select-none ${
                    active
                      ? "bg-sky-default text-sky-lightest border-glow-pulse shadow-md"
                      : "bg-white/40 border-sky-border/40 text-sky-dark hover:text-sky-default hover:bg-white"
                  }`}
                >
                  <div className="flex items-center gap-2.5 truncate">
                    <TabIcon className="w-4 h-4 shrink-0" />
                    <span className="truncate">{tab.label}</span>
                  </div>
                  {tab.count !== undefined && tab.count !== null && (
                    <span className={`px-2 py-0.5 rounded-md font-mono text-[9px] ${
                      active ? "bg-white/20 text-white" : "bg-sky-surface text-sky-dark/70"
                    }`}>
                      {tab.count}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* RIGHT Panel: Dynamic Content (9 cols) */}
          <div className="lg:col-span-9 flex flex-col gap-4">
            
            <AnimatePresence mode="wait">
              {/* TAB 1: EXECUTIVE SESSION REPORT */}
              {activeTab === "sessionReport" && (
                <motion.div
                  key="sessionReport"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.25 }}
                >
                  {sessionReport ? (
                    <SessionReportView report={sessionReport} />
                  ) : (
                    <div className="glass-card rounded-2xl p-12 text-center text-sky-dark/50">
                      <p className="font-heading font-bold text-sm uppercase">No session report available</p>
                      <p className="text-xs font-mono mt-1">Upload a traffic_log.csv or run an analysis video to generate report.</p>
                    </div>
                  )}
                </motion.div>
              )}

              {/* TAB 2: BREAKDOWN CHARTS */}
              {activeTab === "breakdown" && (
                <motion.div
                  key="breakdown"
                  className="grid grid-cols-1 md:grid-cols-2 gap-4"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.25 }}
                >
                  <VehicleDistributionChart data={distData} />
                  <VehiclesOverTimeChart data={timeData} />
                </motion.div>
              )}

              {/* TAB 3: RECOGNIZED PLATES LEDGER */}
              {activeTab === "plates" && (
                <motion.div
                  key="plates"
                  className="glass-card rounded-2xl p-5 shadow-sm"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.25 }}
                >
                  <PlateTable plates={sessionReport?.plates || []} />
                </motion.div>
              )}

              {/* TAB 4: TWO-WHEELER SAFETY REPORT */}
              {activeTab === "twoWheelers" && (
                <motion.div
                  key="twoWheelers"
                  className="glass-card rounded-2xl p-5 shadow-sm"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.25 }}
                >
                  <TwoWheelerSafetyTable statuses={sessionReport?.twoWheelers || []} />
                </motion.div>
              )}

              {/* TAB 5: DEMOGRAPHICS */}
              {activeTab === "demographics" && (
                <motion.div
                  key="demographics"
                  className="glass-card rounded-2xl p-5 shadow-sm"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.25 }}
                >
                  <PedestrianDemographics data={sessionReport?.humans || {}} />
                </motion.div>
              )}

              {/* TAB 6: CROSSING LOG */}
              {activeTab === "crossing" && (
                <motion.div
                  key="crossing"
                  className="glass-card rounded-2xl p-5 shadow-sm"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.25 }}
                >
                  <CrossingVehiclesTable vehicles={sessionReport?.perVehicle || []} />
                </motion.div>
              )}

              {/* TAB 7: RAW TIME-SERIES SNAPSHOT LEDGER */}
              {activeTab === "rawCsv" && (
                <motion.div
                  key="rawCsv"
                  className="glass-card rounded-2xl p-5 shadow-sm"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.25 }}
                >
                  <div className="flex justify-between items-center pb-3 border-b border-sky-border/30 mb-4 font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider">
                    <span>Raw Periodic Snapshot Registry</span>
                    <span className="font-mono font-bold text-sky-default">
                      {csvData.length} SNAPSHOTS LOGGED
                    </span>
                  </div>

                  {loading ? (
                    <div className="flex justify-center items-center py-20">
                      <div className="w-8 h-8 rounded-full border-2 border-sky-border border-t-sky-default animate-spin" />
                    </div>
                  ) : (
                    <div className="overflow-auto max-h-[380px] rounded-xl border border-sky-border/40 shadow-inner custom-scrollbar">
                      <table className="w-full text-xs text-left">
                        <thead>
                          <tr className="bg-sky-default text-sky-lightest font-heading font-bold uppercase tracking-wider text-[10px]">
                            <th className="px-3.5 py-2.5 font-semibold">Timestamp</th>
                            <th className="px-3.5 py-2.5 font-semibold">Total</th>
                            <th className="px-3.5 py-2.5 font-semibold">Cars</th>
                            <th className="px-3.5 py-2.5 font-semibold">Bikes</th>
                            <th className="px-3.5 py-2.5 font-semibold">Rickshaws</th>
                            <th className="px-3.5 py-2.5 font-semibold">Plates Captured</th>
                            <th className="px-3.5 py-2.5 font-semibold">Violations</th>
                          </tr>
                        </thead>
                        <tbody>
                          {csvData.length === 0 ? (
                            <tr className="bg-white/40">
                              <td colSpan={7} className="text-center py-10 font-heading font-bold text-sky-dark/40 uppercase">
                                Registry database empty. Record signals or upload CSV.
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
                                <td className="px-3.5 py-2 font-mono text-[10px] text-sky-dark/70">{row.timestamp}</td>
                                <td className="px-3.5 py-2 font-mono font-bold text-sky-default">{row.total_vehicles}</td>
                                <td className="px-3.5 py-2 font-mono text-sky-dark">{row.cars}</td>
                                <td className="px-3.5 py-2 font-mono text-sky-dark">{(Number(row.motorcycles) || 0) + (Number(row.scooters) || 0)}</td>
                                <td className="px-3.5 py-2 font-mono text-sky-dark">{row.auto_rickshaws || 0}</td>
                                <td className="px-3.5 py-2 font-mono text-sky-default font-bold max-w-xs truncate">
                                  {row.plates_detected || "none"}
                                </td>
                                <td className="px-3.5 py-2 font-mono text-red-600 font-bold max-w-xs truncate">
                                  {row.violation_details || "none"}
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

              {/* TAB 8: SPATIAL MAPPING */}
              {activeTab === "spatialMap" && (
                <motion.div
                  key="spatialMap"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.25 }}
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
  { timestamp: "2026-09-30T10:26:46", total_vehicles: 1, cars: 0, trucks: 0, buses: 0, auto_rickshaws: 0, motorcycles: 1, scooters: 0, bicycles: 0, plates_detected: "none", helmet_violations: 0, violation_details: "none", pedestrians_detected: 6, males: 2, females: 0, children: 1 },
  { timestamp: "2026-09-30T10:26:48", total_vehicles: 1, cars: 0, trucks: 0, buses: 0, auto_rickshaws: 0, motorcycles: 1, scooters: 0, bicycles: 0, plates_detected: "none", helmet_violations: 1, violation_details: "11:no_helmet_rider", pedestrians_detected: 10, males: 5, females: 0, children: 1 },
  { timestamp: "2026-09-30T10:26:54", total_vehicles: 2, cars: 0, trucks: 0, buses: 0, auto_rickshaws: 0, motorcycles: 2, scooters: 0, bicycles: 0, plates_detected: "30:KA05NF9274", helmet_violations: 0, violation_details: "none", pedestrians_detected: 5, males: 3, females: 0, children: 1 },
  { timestamp: "2026-09-30T10:27:00", total_vehicles: 3, cars: 1, trucks: 0, buses: 0, auto_rickshaws: 0, motorcycles: 2, scooters: 0, bicycles: 0, plates_detected: "30:KA05NF9274|50:KA05MF9274", helmet_violations: 0, violation_details: "none", pedestrians_detected: 4, males: 2, females: 0, children: 0 },
  { timestamp: "2026-09-30T10:27:36", total_vehicles: 8, cars: 2, trucks: 0, buses: 0, auto_rickshaws: 1, motorcycles: 5, scooters: 0, bicycles: 0, plates_detected: "30:KA05NF9274|50:KA05MF9274|113:KA05HN8088", helmet_violations: 0, violation_details: "none", pedestrians_detected: 1, males: 1, females: 0, children: 0 },
  { timestamp: "2026-09-30T10:28:28", total_vehicles: 12, cars: 5, trucks: 0, buses: 0, auto_rickshaws: 2, motorcycles: 5, scooters: 0, bicycles: 0, plates_detected: "30:KA05NF9274|50:KA05MF9274|113:KA05HN8088|138:KA09HF8899", helmet_violations: 2, violation_details: "205:no_helmet_rider|205:no_helmet_pillion", pedestrians_detected: 1, males: 0, females: 0, children: 0 },
  { timestamp: "2026-09-30T10:28:37", total_vehicles: 15, cars: 6, trucks: 0, buses: 0, auto_rickshaws: 3, motorcycles: 6, scooters: 0, bicycles: 0, plates_detected: "30:KA05NF9274|50:KA05MF9274|113:KA05HN8088|138:KA09HF8899|241:KA51H9292", helmet_violations: 0, violation_details: "none", pedestrians_detected: 4, males: 1, females: 1, children: 0 },
  { timestamp: "2026-09-30T10:29:23", total_vehicles: 18, cars: 9, trucks: 0, buses: 0, auto_rickshaws: 3, motorcycles: 6, scooters: 0, bicycles: 0, plates_detected: "30:KA05NF9274|50:KA05MF9274|113:KA05HN8088|138:KA09HF8899|241:KA51H9292|331:KA32AB9987", helmet_violations: 0, violation_details: "none", pedestrians_detected: 1, males: 1, females: 0, children: 0 },
  { timestamp: "2026-09-30T10:29:55", total_vehicles: 20, cars: 10, trucks: 0, buses: 0, auto_rickshaws: 3, motorcycles: 7, scooters: 0, bicycles: 0, plates_detected: "30:KA05NF9274|50:KA05MF9274|113:KA05HN8088|138:KA09HF8899|241:KA51H9292|331:KA32AB9987|354:KA05OH5938", helmet_violations: 0, violation_details: "none", pedestrians_detected: 4, males: 2, females: 0, children: 2 },
  { timestamp: "2026-09-30T10:30:19", total_vehicles: 21, cars: 10, trucks: 0, buses: 0, auto_rickshaws: 3, motorcycles: 8, scooters: 0, bicycles: 0, plates_detected: "30:KA05NF9274|50:KA05MF9274|113:KA05HN8088|138:KA09HF8899|241:KA51H9292|331:KA32AB9987|354:KA05OH5938", helmet_violations: 0, violation_details: "none", pedestrians_detected: 5, males: 2, females: 0, children: 0 }
];
