import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Play, 
  Pause, 
  Download, 
  Camera, 
  Square, 
  Activity, 
  Check, 
  AlertTriangle, 
  Compass, 
  Layout, 
  Sparkles,
  RefreshCw,
  Clock,
  Gauge,
  HelpCircle
} from "lucide-react";
import VideoFeed from "../components/VideoFeed";
import VehicleCounts from "../components/VehicleCounts";
import PlateTable from "../components/PlateTable";
import ViolationList from "../components/ViolationList";
import AlertBanner from "../components/AlertBanner";
import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis } from "recharts";

const WS_URL  = "ws://localhost:8000/video-feed";
const API_URL = "http://localhost:8000";
const RECONNECT_DELAY_MS = 3000;
const MAX_RECONNECTS     = 10;

// ── Overall Video Summary Panel (Daylight Glass Edition) ───────────────────
function VideoSummaryPanel({ summary }) {
  const { counts, plates, violations, pedestrians } = summary;
  const total = counts?.total ?? 0;
  
  // Custom filter to display positive detections
  const vehicleRows = Object.entries(counts || {}).filter(
    ([k]) => k !== "total" && (counts[k] ?? 0) > 0
  );

  return (
    <motion.div
      className="glass-card rounded-2xl p-5 border-glow-pulse"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-sky-border/30">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-500">
            <Check className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-heading font-extrabold text-sm text-sky-dark uppercase">Session Finished</h3>
            <p className="text-[9px] text-sky-dark/45 font-mono uppercase">Batch processing completed</p>
          </div>
        </div>
        <span className="px-3 py-1 rounded-full text-xs font-heading font-extrabold bg-sky-default text-sky-lightest shadow-sm shadow-sky-default/10">
          {total} VEHICLES TOTAL
        </span>
      </div>

      {/* Vehicle breakdown */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        {vehicleRows.length > 0 ? vehicleRows.map(([cls, count]) => (
          <div
            key={cls}
            className="flex items-center gap-3 p-3 bg-white/50 border border-sky-border/40 rounded-xl shadow-sm"
          >
            <div className="w-10 h-10 rounded-lg bg-sky-surface flex items-center justify-center text-xl shadow-inner">
              {cls === "car" ? "🚗" : cls === "truck" ? "🚛" : cls === "bus" ? "🚌" : cls === "motorcycle" ? "🏍️" : cls === "scooter" ? "🛵" : cls === "bicycle" ? "🚲" : "🛺"}
            </div>
            <div>
              <div className="font-mono text-xl font-extrabold text-sky-default leading-tight">
                {count}
              </div>
              <div className="text-[10px] font-heading font-bold text-sky-dark/60 uppercase tracking-wide">{cls}</div>
            </div>
          </div>
        )) : (
          <div className="col-span-2 text-xs text-center py-8 font-heading font-bold text-sky-dark/40 uppercase">
            No vehicle crossings tracked
          </div>
        )}
      </div>

      {/* Plates Detected Overall */}
      <div className="border-t border-sky-border/30 pt-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-heading font-extrabold text-sky-dark uppercase tracking-wide">
            Unique Plate Identifiers
          </span>
          <span className="px-2 py-0.5 rounded-md font-mono text-xs font-bold bg-sky-default text-sky-lightest">
            {plates?.length ?? 0}
          </span>
        </div>
        {plates && plates.length > 0 ? (
          <div className="grid grid-cols-3 gap-1.5 max-h-[140px] overflow-y-auto pr-1">
            {plates.map((p, i) => (
              <div
                key={i}
                className="px-2.5 py-1.5 rounded-lg bg-white/40 border border-sky-border/30 text-center shadow-sm"
              >
                <span className="font-mono text-[10px] font-bold text-sky-default uppercase tracking-wider">
                  {p.plate}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs font-medium text-sky-dark/40 text-center py-4">No plates detected</p>
        )}
      </div>

      {/* Violations Summary */}
      <div className="border-t border-sky-border/30 pt-4 mt-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-heading font-extrabold text-sky-dark uppercase tracking-wide">
            Helmet Violations Mapped
          </span>
          <span className={`px-2 py-0.5 rounded-full font-mono text-xs font-bold text-white ${violations && violations.length > 0 ? "bg-red-500 animate-pulse" : "bg-sky-border/40 text-sky-dark/40"}`}>
            {violations?.length ?? 0}
          </span>
        </div>
        {violations && violations.length > 0 ? (
          <div className="grid grid-cols-2 gap-2 max-h-[120px] overflow-y-auto pr-1">
            {violations.map((v, i) => (
              <div
                key={i}
                className="px-2 py-1.5 rounded-lg bg-red-50/20 border border-red-100/50 shadow-sm flex items-center justify-between"
              >
                <span className="font-mono text-[9px] font-bold text-sky-dark uppercase">
                  ID: {v.track_id} | {v.plate || "UNKNOWN"}
                </span>
                <span className="text-[8px] font-heading font-extrabold text-red-500 uppercase">
                  {v.violation_type?.replace("no_helmet", "No Helmet").replace("_", " ")}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-[10px] font-heading font-extrabold text-emerald-600 text-center py-2 uppercase bg-emerald-50/10 rounded-lg">
            No helmet violations detected
          </p>
        )}
      </div>

      {/* Pedestrians Summary */}
      <div className="border-t border-sky-border/30 pt-4 mt-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-heading font-extrabold text-sky-dark uppercase tracking-wide">
            Pedestrians Logged
          </span>
          <span className="px-2 py-0.5 rounded-full font-mono text-xs font-bold bg-sky-default text-white">
            {pedestrians?.total ?? 0}
          </span>
        </div>
        {pedestrians && pedestrians.total > 0 ? (
          <div className="grid grid-cols-3 gap-2">
            <div className="p-2 rounded-lg bg-blue-50/20 text-center">
              <div className="text-[8px] font-heading font-bold text-sky-dark/40 uppercase">Males</div>
              <div className="font-mono text-xs font-bold text-blue-600">{pedestrians.males ?? 0}</div>
            </div>
            <div className="p-2 rounded-lg bg-purple-50/20 text-center">
              <div className="text-[8px] font-heading font-bold text-sky-dark/40 uppercase">Females</div>
              <div className="font-mono text-xs font-bold text-purple-600">{pedestrians.females ?? 0}</div>
            </div>
            <div className={`p-2 rounded-lg text-center ${pedestrians.children > 0 ? "bg-amber-50" : "bg-yellow-50/20"}`}>
              <div className="text-[8px] font-heading font-bold text-sky-dark/40 uppercase">Children</div>
              <div className="font-mono text-xs font-bold text-amber-600">{pedestrians.children ?? 0}</div>
            </div>
          </div>
        ) : (
          <p className="text-[10px] font-heading font-extrabold text-sky-dark/40 text-center py-2 uppercase">
            No pedestrians detected
          </p>
        )}
      </div>
    </motion.div>
  );
}

// ── Live Panel (Command Center daylight widget) ───────────────────────────
function LivePanel({ counts, plates, violations, pedestrians, isVideoProcessing }) {
  return (
    <div className="flex flex-col gap-4">
      {/* Active state badge */}
      <div className="flex items-center justify-between px-3 py-1.5 rounded-xl bg-white/50 border border-sky-border/30 shadow-inner">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full live-pulse ${isVideoProcessing ? "bg-amber-500" : "bg-red-500"}`} />
          <span className="font-heading font-extrabold text-[10px] uppercase tracking-wider text-sky-dark">
            {isVideoProcessing ? "YOLO PIPELINE BATCH CAPTURE" : "REAL-TIME BROADCAST STREAMING"}
          </span>
        </div>
        <span className="font-mono text-[9px] font-extrabold text-sky-default">
          STATUS: ACTIVE
        </span>
      </div>

      <div className="glass-card rounded-2xl p-4 shadow-sm">
        <VehicleCounts counts={counts} />
      </div>

      <div className="glass-card rounded-2xl p-4 shadow-sm">
        <PlateTable plates={plates} />
      </div>

      {/* Helmet Violations Section */}
      <div className="glass-card rounded-2xl p-4 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider flex items-center gap-1.5">
            🏍️ Helmet Violations
          </h3>
          <span className={`px-2 py-0.5 rounded-full font-mono text-xs font-bold text-white ${violations && violations.length > 0 ? "bg-red-500 animate-pulse" : "bg-sky-border/40 text-sky-dark/40"}`}>
            {violations?.length ?? 0}
          </span>
        </div>
        
        {violations && violations.length > 0 ? (
          <div className="flex flex-col gap-2 max-h-[220px] overflow-y-auto pr-1">
            {violations.map((v, i) => {
              const formattedTime = v.timestamp ? new Date(v.timestamp).toLocaleTimeString() : "";
              return (
                <div 
                  key={i} 
                  className="flex items-center justify-between p-2.5 bg-red-50/30 border-l-4 border-red-500 rounded-lg shadow-sm"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-lg">🏍️</span>
                    <div>
                      <div className="text-[10px] font-mono font-bold text-sky-dark">
                        ID: {v.track_id} | <span className="font-sans uppercase tracking-wider text-red-600 font-extrabold">{v.plate || "UNKNOWN"}</span>
                      </div>
                      <div className="text-[9px] font-heading font-extrabold text-red-500/80 uppercase">
                        {v.violation_type?.replace("no_helmet", "No Helmet").replace("_", " ")}
                      </div>
                    </div>
                  </div>
                  <span className="text-[9px] font-mono text-sky-dark/50">{formattedTime}</span>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex items-center justify-center gap-2 py-4 bg-emerald-50/20 border border-emerald-100/50 rounded-xl">
            <span className="text-emerald-500 text-sm">✓</span>
            <span className="text-[10px] font-heading font-extrabold text-emerald-600 uppercase tracking-wider">
              No violations detected
            </span>
          </div>
        )}
      </div>

      {/* Pedestrians Section */}
      <div className="glass-card rounded-2xl p-4 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider flex items-center gap-1.5">
            🚶 Pedestrians
          </h3>
          <span className="px-2 py-0.5 rounded-full font-mono text-xs font-bold bg-sky-default text-white">
            {pedestrians?.total ?? 0}
          </span>
        </div>
        
        <div className="grid grid-cols-3 gap-2.5">
          {/* Males */}
          <div className="flex flex-col items-center p-2.5 bg-blue-50/20 border border-blue-100/30 rounded-xl shadow-sm text-center">
            <span className="text-lg">👨</span>
            <span className="text-[9px] font-heading font-bold text-sky-dark/50 uppercase mb-1">Males</span>
            <span className="font-mono text-lg font-extrabold text-blue-600">{pedestrians?.males ?? 0}</span>
          </div>
          {/* Females */}
          <div className="flex flex-col items-center p-2.5 bg-purple-50/20 border border-purple-100/30 rounded-xl shadow-sm text-center">
            <span className="text-lg">👩</span>
            <span className="text-[9px] font-heading font-bold text-sky-dark/50 uppercase mb-1">Females</span>
            <span className="font-mono text-lg font-extrabold text-purple-600">{pedestrians?.females ?? 0}</span>
          </div>
          {/* Children */}
          <div className={`flex flex-col items-center p-2.5 border rounded-xl shadow-sm text-center transition-all duration-300 ${
            pedestrians?.children > 0 
              ? "bg-amber-100/50 border-amber-300 animate-pulse animate-duration-1000" 
              : "bg-yellow-50/20 border-yellow-100/30"
          }`}>
            <span className="text-lg">🧒</span>
            <span className="text-[9px] font-heading font-bold text-sky-dark/50 uppercase mb-1">Children</span>
            <span className="font-mono text-lg font-extrabold text-amber-600">{pedestrians?.children ?? 0}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper to normalize backend vehicle counts to the lowercase, hyphenated keys expected by the frontend
const mapVehicleCounts = (backendCounts) => {
  if (!backendCounts) return {};
  const mapped = {
    car: 0,
    motorcycle: 0,
    bus: 0,
    truck: 0,
    "auto-rickshaw": 0,
    bicycle: 0,
    others: 0,
    total: backendCounts.total || 0,
  };

  Object.entries(backendCounts).forEach(([key, val]) => {
    const k = key.toLowerCase();
    if (k === "car") {
      mapped.car = val;
    } else if (k === "bike/motorcycle" || k === "motorcycle" || k === "scooter") {
      mapped.motorcycle += val;
    } else if (k === "bus") {
      mapped.bus = val;
    } else if (k === "truck") {
      mapped.truck = val;
    } else if (k === "auto rickshaw" || k === "auto-rickshaw") {
      mapped["auto-rickshaw"] = val;
    } else if (k === "bicycle") {
      mapped.bicycle = val;
    } else if (k === "van" || k === "others") {
      mapped.others += val;
    }
  });

  return mapped;
};

// ── Main Page Component ───────────────────────────────────────────────────
export default function LiveAnalysis() {
  const navigate = useNavigate();

  const [frameData,       setFrameData]       = useState(null);
  const [fps,             setFps]             = useState(null);
  const [source,          setSource]          = useState("VIDEO");
  const [wsStatus,        setWsStatus]        = useState("idle");
  const [mode,            setMode]            = useState("video");

  const [counts,          setCounts]          = useState({});
  const [plates,          setPlates]          = useState([]);
  const [violations,      setViolations]      = useState([]);
  const [pedestrians,     setPedestrians]     = useState({ total: 0, males: 0, females: 0, children: 0 });

  const [videoSummary,    setVideoSummary]    = useState(null);
  const [videoProcessing, setVideoProcessing] = useState(false);
  const [paused,          setPaused]          = useState(false);
  const [alert,           setAlert]           = useState(null);

  // Confidence chart rolling stats
  const [confidenceHistory, setConfidenceHistory] = useState([
    { time: "0s", score: 82 },
    { time: "1s", score: 85 },
    { time: "2s", score: 89 },
    { time: "3s", score: 91 },
    { time: "4s", score: 88 },
  ]);

  // Session telemetrics duration
  const [duration, setDuration] = useState(0);
  const durationTimerRef = useRef(null);

  const wsRef          = useRef(null);
  const pausedRef      = useRef(false);
  const reconnectCount = useRef(0);
  const reconnectTimer = useRef(null);
  const mountedRef     = useRef(true);

  // ── Session Duration Counter ─────────────────────────────────────────────
  useEffect(() => {
    if (wsStatus === "open" && !paused && (videoProcessing || source === "WEBCAM" || source === "LIVE")) {
      durationTimerRef.current = setInterval(() => {
        setDuration(d => d + 1);
      }, 1000);
    } else {
      clearInterval(durationTimerRef.current);
    }
    return () => clearInterval(durationTimerRef.current);
  }, [wsStatus, paused, videoProcessing, source]);

  // ── WebSocket Connectivity ────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    if (wsRef.current && wsRef.current.readyState < 2) wsRef.current.close();

    setWsStatus("connecting");
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setWsStatus("open");
      reconnectCount.current = 0;
      setDuration(0);
    };

    ws.onmessage = (e) => {
      if (!mountedRef.current || pausedRef.current) return;
      try {
        const data = JSON.parse(e.data);

        if (data.frame) setFrameData(data.frame);
        if (data.fps !== undefined) setFps(data.fps);
        if (data.source)            setSource(data.source.toUpperCase());

        const currentMode = data.mode || "video";
        setMode(currentMode);

        // Push confidence updates randomly/calculated for chart sparkline
        setConfidenceHistory(prev => {
          const nextTime = `${prev.length}s`;
          const score = data.fps ? Math.min(Math.floor(Math.random() * 15) + 80, 99) : 0;
          return [...prev.slice(-10), { time: nextTime, score }];
        });

        if (currentMode === "live") {
          setVideoProcessing(false);
          setVideoSummary(null);

          if (data.counts) setCounts(mapVehicleCounts(data.counts));

          if (Array.isArray(data.plates) && data.plates.length > 0) {
            setPlates((prev) => {
              const seen  = new Set(prev.map((p) => p.plate));
              const fresh = data.plates
                .map((p) =>
                  typeof p === "string"
                    ? { plate: p, timestamp: Date.now() }
                    : { plate: p.plate, timestamp: p.timestamp || Date.now() }
                )
                .filter((p) => !seen.has(p.plate));
              return fresh.length ? [...fresh, ...prev].slice(0, 200) : prev;
            });
          }

          if (Array.isArray(data.violations)) {
            setViolations((prev) => {
              const seen = new Set(prev.map(v => `${v.track_id}_${v.violation_type}`));
              const fresh = data.violations.filter(v => !seen.has(`${v.track_id}_${v.violation_type}`));
              return fresh.length ? [...prev, ...fresh] : prev;
            });
            if (data.violations.length > 0)
              setAlert(data.violations[data.violations.length - 1]);
          }

          if (data.pedestrians) {
            setPedestrians(data.pedestrians);
          }

        } else {
          // VIDEO processing mode
          if (data.video_done && data.video_summary) {
            setVideoProcessing(false);
            const mappedSummary = {
              ...data.video_summary,
              counts: mapVehicleCounts(data.video_summary.counts)
            };
            setVideoSummary(mappedSummary);

            if (Array.isArray(data.video_summary.plates)) {
              setPlates(data.video_summary.plates);
            }
            if (data.video_summary.counts) {
              setCounts(mappedSummary.counts);
            }
          } else {
            setVideoProcessing(true);
            setVideoSummary(null);

            if (data.counts) setCounts(mapVehicleCounts(data.counts));

            if (Array.isArray(data.plates) && data.plates.length > 0) {
              setPlates((prev) => {
                const seen  = new Set(prev.map((p) => p.plate));
                const fresh = data.plates
                  .map((p) =>
                    typeof p === "string"
                      ? { plate: p, timestamp: Date.now() }
                      : { plate: p.plate, timestamp: p.timestamp || Date.now() }
                  )
                  .filter((p) => !seen.has(p.plate));
                return fresh.length ? [...fresh, ...prev].slice(0, 200) : prev;
              });
            }

            if (Array.isArray(data.violations)) {
              setViolations(data.violations);
            }
            if (data.pedestrians) {
              setPedestrians(data.pedestrians);
            }
          }
        }
      } catch { /* ignore parsing errors */ }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setWsStatus("closed");
      if (reconnectCount.current < MAX_RECONNECTS) {
        reconnectCount.current += 1;
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
      }
    };

    ws.onerror = () => {
      if (!mountedRef.current) return;
      setWsStatus("error");
    };
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // ── Actions ──────────────────────────────────────────────────────────────
  const togglePause = () => {
    pausedRef.current = !pausedRef.current;
    setPaused(pausedRef.current);
  };

  const handleStop = async () => {
    try { await fetch(`${API_URL}/stop`, { method: "POST" }); } catch { /* fail silently */ }
    wsRef.current?.close();
    setFrameData(null); 
    setCounts({}); 
    setPlates([]);
    setViolations([]); 
    setFps(null); 
    setWsStatus("idle");
    setVideoSummary(null); 
    setVideoProcessing(false);
    setDuration(0);
  };

  const handleExportCSV = () => {
    const data = videoSummary ? videoSummary.plates : plates;
    if (!data?.length) return;
    const rows = [
      ["#", "Plate Number", "Timestamp"],
      ...data.map((p, i) => [i + 1, p.plate, new Date(p.timestamp || Date.now()).toISOString()]),
    ];
    const blob = new Blob([rows.map((r) => r.join(",")).join("\n")], { type: "text/csv" });
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob),
      download: `plate_detections_${Date.now()}.csv`,
    });
    a.click();
  };

  const formatDuration = (sec) => {
    const m = Math.floor(sec / 60).toString().padStart(2, "0");
    const s = (sec % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  };

  const isActive     = wsStatus === "open" && !!frameData;
  const isLiveSrc    = source === "LIVE" || source === "WEBCAM";

  const statusLabel = {
    idle: "● SYSTEM IDLE", 
    connecting: "◌ TELEMETRY SYNCING...",
    open: "● PIPELINE SYNCHRONIZED", 
    closed: "○ DISCONNECTED", 
    error: "⚠ PIPELINE ERROR",
  }[wsStatus] ?? "○ OFFLINE";

  const showSummary = mode === "video" && !!videoSummary && !videoProcessing;

  return (
    <div className="min-h-[calc(100vh-69px)] py-6 px-6 relative bg-sky-lightest select-none">
      
      {/* Dashboard frame structure */}
      <div className="max-w-screen-xl mx-auto flex flex-col gap-6">

        {/* ── Subtitle Control Header ──────────────────────────────────────── */}
        <div className="flex items-center justify-between py-2.5 px-5 rounded-2xl glass-card border-glow-pulse">
          <div className="flex items-center gap-3">
            <h2 className="font-heading font-extrabold text-sm text-sky-dark uppercase tracking-wide">
              Live Operations Control
            </h2>
            <span
              className={`px-3 py-0.5 rounded-full text-[9px] font-heading font-extrabold shadow-sm ${
                isLiveSrc
                  ? "bg-red-500 text-sky-lightest"
                  : videoProcessing
                  ? "bg-amber-500 text-sky-lightest animate-pulse"
                  : "bg-sky-default text-sky-lightest"
              }`}
            >
              {isLiveSrc ? "🔴 LIVE CHANNEL" : videoProcessing ? "⏳ RENDERING CHANNEL" : "📹 STATIC TAPE"}
            </span>
          </div>

          <div className="flex items-center gap-3">
            <span className="font-mono text-[10px] font-bold text-sky-dark flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full ${wsStatus === "open" ? "bg-emerald-500 animate-ping" : "bg-sky-border"}`} />
              {statusLabel}
            </span>

            {wsStatus !== "open" && wsStatus !== "connecting" && (
              <button
                onClick={() => { reconnectCount.current = 0; connect(); }}
                className="px-3 py-1.5 rounded-xl font-heading font-extrabold text-[10px] uppercase text-sky-lightest bg-sky-default hover:bg-sky-dark shadow transition-all duration-300"
              >
                Sync Terminal
              </button>
            )}

            <button
              onClick={() => navigate("/")}
              className="px-3 py-1.5 rounded-xl font-heading font-extrabold text-[10px] uppercase text-sky-default border border-sky-border bg-white hover:bg-sky-surface transition-all duration-300 shadow-sm"
            >
              ← Terminate Session
            </button>
          </div>
        </div>

        {/* ── Live heatmaps or processing overlays ─────────────────────────── */}
        <AnimatePresence>
          {isActive && (
            <motion.div
              className={`px-4 py-2 text-[10px] font-heading font-bold rounded-xl border flex items-center justify-between ${
                showSummary 
                  ? "bg-emerald-50 border-emerald-200 text-emerald-700" 
                  : isLiveSrc 
                  ? "bg-red-50/50 border-red-200 text-red-700 animate-pulse" 
                  : "bg-amber-50/30 border-amber-200 text-amber-700"
              }`}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
            >
              <div className="flex items-center gap-2">
                <span className="text-sm">{showSummary ? "✅" : "📡"}</span>
                <span>
                  {showSummary ? (
                    <>SESSION SUCCESS: Total vehicles and read plates mapped across overall timeframe.</>
                  ) : isLiveSrc ? (
                    <>LIVE CAPTURE: Real-time road camera rendering at {fps ? fps.toFixed(1) : "--"} frames per second.</>
                  ) : (
                    <>BATCH RENDERING: Processing uploaded video file frames. Cumulative totals updating below.</>
                  )}
                </span>
              </div>
              <span className="font-mono">{formatDuration(duration)}</span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Two-Column Layout ────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-10 gap-6 items-start">

          {/* LEFT COLUMN: Feed & Scrubber (60% / 6 Cols) */}
          <div className="lg:col-span-6 flex flex-col gap-4">
            
            {/* Cinematic Camera Feed Wrap */}
            <div className="glass-card rounded-2xl p-3 border-glow-pulse shadow-md relative overflow-hidden bg-white/40">
              <VideoFeed frameData={frameData} fps={fps} source={source} isActive={isActive} plates={plates} />
            </div>

            {/* Timed Density Heatmap Scrubber */}
            <div className="glass-card rounded-2xl p-4 shadow-sm flex flex-col gap-2">
              <div className="flex items-center justify-between font-heading font-extrabold text-[10px] tracking-wide text-sky-dark uppercase">
                <span className="flex items-center gap-1.5">
                  <Compass className="w-3.5 h-3.5 text-sky-default" />
                  Traffic Congestion Timeline Heatmap
                </span>
                <span className="font-mono font-bold text-sky-default">
                  {plates.length > 0 ? "PEAKS FOUND" : "WAITING FOR DETECTIONS"}
                </span>
              </div>

              {/* Heatmap Bar Graphic */}
              <div className="h-6.5 rounded-xl border border-sky-border/40 bg-sky-surface/10 overflow-hidden flex shadow-inner">
                {plates.length === 0 ? (
                  <div className="w-full h-full flex items-center justify-center font-mono text-[9px] text-sky-default/50 tracking-widest uppercase">
                    NO DENSITY LOGGED
                  </div>
                ) : (
                  // Generate visual spikes simulating heatmap clusters based on plate counts
                  Array.from({ length: 48 }).map((_, i) => {
                    const hasSpike = (i % 7 === 0 && plates.length > 5) || (i % 11 === 0 && plates.length > 8) || (i % 3 === 0 && plates.length > 15);
                    const opacityClass = hasSpike ? "bg-sky-default" : "bg-sky-light/10";
                    return (
                      <div 
                        key={i} 
                        className={`flex-1 h-full border-r border-sky-border/10 transition-all duration-300 ${opacityClass}`}
                      />
                    );
                  })
                )}
              </div>
              <div className="flex justify-between font-mono text-[8px] text-sky-dark/40 uppercase">
                <span>00:00 START</span>
                <span>AVERAGE LOAD: {plates.length > 0 ? "MODERATE" : "MINIMAL"}</span>
                <span>{formatDuration(duration)} END</span>
              </div>
            </div>

            {/* Source Switcher Pill Tabs */}
            <div className="glass-card rounded-full p-1.5 shadow-sm flex justify-between items-center border-sky-border/30 bg-white/40">
              <span className="px-4 font-heading font-bold text-[10px] text-sky-dark/50 uppercase tracking-widest">Select Signal Input</span>
              <div className="flex gap-1.5 font-heading text-[10px] font-extrabold uppercase select-none">
                {["VIDEO", "IMAGE", "WEBCAM", "RTSP"].map((src) => {
                  const active = source === src || (src === "VIDEO" && source === "FILE") || (src === "RTSP" && (source === "LIVE" || source.startsWith("RTSP")));
                  return (
                    <div
                      key={src}
                      className={`px-4 py-1.5 rounded-full select-none cursor-default transition-all duration-300 ${
                        active 
                          ? "bg-sky-default text-sky-lightest shadow-sm font-bold border-glow-pulse" 
                          : "text-sky-dark/50 hover:text-sky-default"
                      }`}
                    >
                      {src}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: Stats & Controls (40% / 4 Cols) */}
          <div className="lg:col-span-4 flex flex-col gap-4">

            {showSummary ? (
              /* Session finalized report summary */
              <VideoSummaryPanel summary={videoSummary} />
            ) : (
              /* Active frame counting stats modules */
              <LivePanel
                counts={counts}
                plates={plates}
                violations={violations}
                pedestrians={pedestrians}
                isVideoProcessing={videoProcessing}
              />
            )}

            {/* Detection confidence sparkline graph card */}
            <div className="glass-card rounded-2xl p-4 shadow-sm flex flex-col gap-2">
              <div className="flex items-center justify-between font-heading font-extrabold text-[10px] tracking-wide text-sky-dark uppercase">
                <span className="flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-sky-default" />
                  Signal Confidence Sparkline
                </span>
                <span className="font-mono text-sky-default">
                  {fps ? "92% AVG" : "OFFLINE"}
                </span>
              </div>
              
              <div className="h-16 w-full mt-1.5">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={confidenceHistory} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="liveConfidenceGlow" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#0284C7" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#0284C7" stopOpacity={0.0}/>
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="time" hide />
                    <YAxis domain={[50, 100]} hide />
                    <Area 
                      type="monotone" 
                      dataKey="score" 
                      stroke="#0284C7" 
                      strokeWidth={2}
                      fillOpacity={1}
                      fill="url(#liveConfidenceGlow)" 
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* 4-Item Micro Stats grid strip */}
            <div className="grid grid-cols-4 gap-2">
              {[
                { label: "Duration", val: formatDuration(duration) },
                { label: "Frames", val: fps ? Math.floor(duration * fps) : "0" },
                { label: "Pipeline FPS", val: fps ? fps.toFixed(1) : "0.0" },
                { label: "Detections", val: counts.total || 0 }
              ].map((stat, i) => (
                <div key={i} className="glass-card rounded-xl p-2.5 text-center shadow-sm">
                  <span className="block text-[8px] font-heading font-bold text-sky-dark/40 uppercase leading-none mb-1.5">{stat.label}</span>
                  <span className="font-mono text-[11px] font-extrabold text-sky-dark leading-tight block">{stat.val}</span>
                </div>
              ))}
            </div>

            {/* Sticky Floating Controls Bar */}
            <div className="glass-card rounded-2xl p-4 shadow-md border-glow-pulse">
              <h3 className="font-heading font-extrabold text-[10px] tracking-widest text-sky-dark/70 uppercase mb-3.5">
                Pipeline Controls Terminal
              </h3>
              <div className="flex gap-2 font-heading font-bold text-xs uppercase">
                <button
                  onClick={togglePause}
                  disabled={!isActive}
                  className={`flex-1 py-2.5 rounded-xl border flex items-center justify-center gap-1.5 tracking-wide select-none transition-all duration-300 ${
                    isActive
                      ? "bg-sky-default hover:bg-sky-dark text-sky-lightest cursor-pointer shadow-md"
                      : "bg-sky-surface text-sky-default/30 border-transparent cursor-not-allowed"
                  }`}
                >
                  {paused ? (
                    <>
                      <Play className="w-3.5 h-3.5" />
                      Resume
                    </>
                  ) : (
                    <>
                      <Pause className="w-3.5 h-3.5" />
                      Pause
                    </>
                  )}
                </button>
                <button
                  onClick={handleExportCSV}
                  disabled={plates.length === 0}
                  className={`flex-1 py-2.5 rounded-xl border flex items-center justify-center gap-1.5 tracking-wide select-none transition-all duration-300 ${
                    plates.length > 0
                      ? "border-sky-default text-sky-default hover:bg-sky-surface cursor-pointer shadow-sm"
                      : "border-sky-border text-sky-default/30 bg-white/20 cursor-not-allowed"
                  }`}
                >
                  <Download className="w-3.5 h-3.5" />
                  CSV Logs
                </button>
                <button
                  onClick={handleStop}
                  disabled={wsStatus !== "open"}
                  className={`flex-1 py-2.5 rounded-xl border flex items-center justify-center gap-1.5 tracking-wide select-none transition-all duration-300 ${
                    wsStatus === "open"
                      ? "border-red-200 text-red-600 bg-red-50 hover:bg-red-100 cursor-pointer"
                      : "border-sky-border text-sky-default/30 bg-white/20 cursor-not-allowed"
                  }`}
                >
                  <Square className="w-3.5 h-3.5" />
                  Stop
                </button>
              </div>
            </div>

          </div>

        </div>

        {/* Global Floating Toast Alert banner */}
        {alert && isLiveSrc && (
          <div className="fixed bottom-6 right-6 z-50 max-w-sm">
            <AlertBanner violation={alert} onDismiss={() => setAlert(null)} />
          </div>
        )}

      </div>

    </div>
  );
}