import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import VideoFeed from "../components/VideoFeed";
import VehicleCounts from "../components/VehicleCounts";
import PlateTable from "../components/PlateTable";
import ViolationList from "../components/ViolationList";
import AlertBanner from "../components/AlertBanner";

const WS_URL  = "ws://localhost:8000/video-feed";
const API_URL = "http://localhost:8000";

const RECONNECT_DELAY_MS = 3000;
const MAX_RECONNECTS     = 10;

const VEHICLE_ICONS = {
  car: "🚗", truck: "🚛", bus: "🚌",
  "auto-rickshaw": "🛺", motorcycle: "🏍️",
  scooter: "🛵", bicycle: "🚲",
};

// ── Video Summary Panel ────────────────────────────────────────────────────
function VideoSummaryPanel({ summary, isProcessing }) {
  if (isProcessing) {
    return (
      <div
        className="rounded-panel p-5"
        style={{ backgroundColor: "#E0F2FE", border: "1px solid #BAE6FD", borderRadius: "12px" }}
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="w-2 h-2 rounded-full live-pulse" style={{ backgroundColor: "#0284C7" }} />
          <h3 className="text-sm font-semibold" style={{ color: "#0C4A6E" }}>
            Analysing Video…
          </h3>
        </div>
        <p className="text-xs mb-4" style={{ color: "#475569" }}>
          Processing all frames. Final overall count will appear when complete.
        </p>
        <div className="grid grid-cols-2 gap-2">
          {["Cars", "Trucks", "Motorcycles", "Buses"].map((v) => (
            <div
              key={v}
              className="flex items-center gap-2 p-2"
              style={{ backgroundColor: "#F0F9FF", border: "1px solid #BAE6FD", borderRadius: "8px" }}
            >
              <div style={{ width: 28, height: 28, background: "#BAE6FD", borderRadius: 6 }} />
              <div>
                <div style={{ width: 32, height: 8, background: "#BAE6FD", borderRadius: 4, marginBottom: 4 }} />
                <div style={{ width: 48, height: 10, background: "#7DD3FC", borderRadius: 4 }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!summary) return null;

  const { counts, plates } = summary;
  const vehicleRows = Object.entries(counts || {}).filter(
    ([k]) => k !== "total" && (counts[k] ?? 0) > 0
  );

  return (
    <div
      className="rounded-panel p-5"
      style={{ backgroundColor: "#E0F2FE", border: "1px solid #BAE6FD", borderRadius: "12px" }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span style={{ fontSize: 18 }}>✅</span>
          <h3 className="text-sm font-semibold" style={{ color: "#0C4A6E" }}>
            Video Analysis Complete
          </h3>
        </div>
        <span
          className="text-xs px-2 py-1 rounded-full font-bold"
          style={{ backgroundColor: "#0284C7", color: "#fff" }}
        >
          {counts?.total ?? 0} vehicles total
        </span>
      </div>

      {/* Vehicle breakdown */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        {vehicleRows.length > 0 ? vehicleRows.map(([cls, count]) => (
          <div
            key={cls}
            className="flex items-center gap-2 p-2"
            style={{ backgroundColor: "#F0F9FF", border: "1px solid #BAE6FD", borderRadius: "8px" }}
          >
            <span style={{ fontSize: 22 }}>{VEHICLE_ICONS[cls] || "🚗"}</span>
            <div>
              <div className="text-xl font-bold" style={{ color: "#0284C7", lineHeight: 1 }}>
                {count}
              </div>
              <div className="text-xs capitalize" style={{ color: "#475569" }}>{cls}</div>
            </div>
          </div>
        )) : (
          <div className="col-span-2 text-xs text-center py-4" style={{ color: "#475569" }}>
            No vehicles detected
          </div>
        )}
      </div>

      {/* Plates */}
      <div style={{ borderTop: "1px solid #BAE6FD", paddingTop: "12px" }}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold" style={{ color: "#0C4A6E" }}>
            Plates Detected
          </span>
          <span
            className="text-xs px-2 py-0.5 rounded-full font-bold"
            style={{ backgroundColor: "#0284C7", color: "#fff" }}
          >
            {plates?.length ?? 0}
          </span>
        </div>
        {plates && plates.length > 0 ? (
          <div className="flex flex-col gap-1" style={{ maxHeight: "140px", overflowY: "auto" }}>
            {plates.map((p, i) => (
              <div
                key={i}
                className="px-2 py-1 rounded"
                style={{ backgroundColor: "#F0F9FF", borderRadius: "6px" }}
              >
                <span className="font-mono text-xs font-bold" style={{ color: "#0284C7", letterSpacing: "0.06em" }}>
                  {p.plate}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs" style={{ color: "#94A3B8" }}>No plates read yet</p>
        )}
      </div>
    </div>
  );
}

// ── Live Per-Second Panel ─────────────────────────────────────────────────
function LivePanel({ counts, plates, violations }) {
  return (
    <>
      <div className="flex items-center gap-2 mb-2">
        <span className="w-2 h-2 rounded-full live-pulse" style={{ backgroundColor: "#EF4444", display: "inline-block" }} />
        <span className="text-xs font-semibold" style={{ color: "#EF4444" }}>
          LIVE — updating every second
        </span>
      </div>

      <div
        className="rounded-panel p-4"
        style={{ backgroundColor: "#E0F2FE", border: "1px solid #BAE6FD", borderRadius: "12px" }}
      >
        <VehicleCounts counts={counts} />
      </div>

      <div style={{ height: "1px", backgroundColor: "#BAE6FD", margin: "8px 0" }} />

      <div
        className="rounded-panel p-4"
        style={{ backgroundColor: "#E0F2FE", border: "1px solid #BAE6FD", borderRadius: "12px" }}
      >
        <PlateTable plates={plates} />
      </div>

      <div style={{ height: "1px", backgroundColor: "#BAE6FD", margin: "8px 0" }} />

      <div
        className="rounded-panel p-4"
        style={{ backgroundColor: "#E0F2FE", border: "1px solid #BAE6FD", borderRadius: "12px" }}
      >
        <ViolationList violations={violations} />
      </div>
    </>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────
export default function LiveAnalysis() {
  const navigate = useNavigate();

  const [frameData,       setFrameData]       = useState(null);
  const [fps,             setFps]             = useState(null);
  const [source,          setSource]          = useState("VIDEO");
  const [wsStatus,        setWsStatus]        = useState("idle");
  const [mode,            setMode]            = useState("video");  // "video" | "live"

  // Live mode state
  const [counts,     setCounts]     = useState({});
  const [plates,     setPlates]     = useState([]);
  const [violations, setViolations] = useState([]);

  // Video mode state
  const [videoSummary,    setVideoSummary]    = useState(null);
  const [videoProcessing, setVideoProcessing] = useState(false);

  const [paused,  setPaused]  = useState(false);
  const [alert,   setAlert]   = useState(null);

  const wsRef          = useRef(null);
  const pausedRef      = useRef(false);
  const reconnectCount = useRef(0);
  const reconnectTimer = useRef(null);
  const mountedRef     = useRef(true);

  // ── WebSocket connect ──────────────────────────────────────────────────
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
    };

    ws.onmessage = (e) => {
      if (!mountedRef.current || pausedRef.current) return;
      try {
        const data = JSON.parse(e.data);

        if (data.frame)  setFrameData(data.frame);
        if (data.fps !== undefined) setFps(data.fps);
        if (data.source) setSource(data.source.toUpperCase());

        const isLive = data.mode === "live";
        setMode(isLive ? "live" : "video");

        if (isLive) {
          // ── LIVE: update every message ──────────────────────────
          setVideoProcessing(false);
          setVideoSummary(null);

          if (data.counts) setCounts(data.counts);

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
              if (!fresh.length) return prev;
              return [...fresh, ...prev].slice(0, 200);
            });
          }

          if (Array.isArray(data.violations)) {
            setViolations(data.violations);
            if (data.violations.length > 0)
              setAlert(data.violations[data.violations.length - 1]);
          }

        } else {
          // ── VIDEO: show progress then final summary ─────────────
          if (data.video_done && data.video_summary) {
            setVideoProcessing(false);
            setVideoSummary(data.video_summary);
          } else if (!data.video_done) {
            setVideoProcessing(true);
          }
        }
      } catch { /* skip malformed */ }
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
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const togglePause = () => {
    pausedRef.current = !pausedRef.current;
    setPaused(pausedRef.current);
  };

  const handleStop = async () => {
    try { await fetch(`${API_URL}/stop`, { method: "POST" }); } catch { /* ignore */ }
    wsRef.current?.close();
    setFrameData(null); setCounts({}); setPlates([]);
    setViolations([]); setFps(null); setWsStatus("idle");
    setVideoSummary(null); setVideoProcessing(false);
  };

  const handleExportCSV = () => {
    const data = mode === "video" && videoSummary ? videoSummary.plates : plates;
    if (!data?.length) return;
    const rows = [
      ["#", "Plate", "Timestamp"],
      ...data.map((p, i) => [i + 1, p.plate, new Date(p.timestamp || Date.now()).toISOString()]),
    ];
    const blob = new Blob([rows.map((r) => r.join(",")).join("\n")], { type: "text/csv" });
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob),
      download: `plates_${Date.now()}.csv`,
    });
    a.click();
  };

  const isActive = wsStatus === "open" && !!frameData;
  const isLiveSource = source === "LIVE" || source === "WEBCAM";

  const statusLabel = {
    idle: "○ Idle", connecting: "◌ Connecting…",
    open: "● Connected", closed: "○ Disconnected", error: "○ Error",
  }[wsStatus] ?? "○ Unknown";

  return (
    <div className="min-h-screen" style={{ backgroundColor: "#F0F9FF" }}>
      <div className="max-w-screen-xl mx-auto px-6 py-6">

        {/* ── Header ──────────────────────────────────────────────── */}
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-bold" style={{ color: "#0C4A6E" }}>Live Analysis</h2>
            <span
              className="text-xs px-3 py-1 rounded-full font-semibold"
              style={
                isLiveSource
                  ? { backgroundColor: "#FEE2E2", color: "#DC2626", border: "1px solid #FCA5A5" }
                  : { backgroundColor: "#E0F2FE", color: "#0284C7", border: "1px solid #BAE6FD" }
              }
            >
              {isLiveSource ? "🔴 LIVE" : "📹 VIDEO"}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span
              className="text-xs px-3 py-1 rounded-full font-medium"
              style={{
                backgroundColor: "#E0F2FE",
                color: wsStatus === "open" ? "#0284C7" : "#475569",
                border: "1px solid #BAE6FD",
              }}
            >
              {statusLabel}
            </span>

            {wsStatus !== "open" && wsStatus !== "connecting" && (
              <button
                onClick={() => { reconnectCount.current = 0; connect(); }}
                className="text-xs px-3 py-1.5 rounded-full font-semibold"
                style={{ backgroundColor: "#0284C7", color: "#F0F9FF" }}
              >
                Reconnect
              </button>
            )}

            <button
              onClick={() => navigate("/")}
              className="text-xs px-3 py-1.5 rounded-full font-semibold"
              style={{ border: "1px solid #BAE6FD", color: "#0284C7", backgroundColor: "transparent" }}
            >
              ← Upload New
            </button>
          </div>
        </div>

        {/* ── Mode explanation banner ────────────────────────────── */}
        {isActive && (
          <div
            className="mb-4 px-4 py-2 text-xs flex items-center gap-2"
            style={{
              backgroundColor: mode === "live" ? "#FEF2F2" : "#EFF6FF",
              border: `1px solid ${mode === "live" ? "#FCA5A5" : "#BFDBFE"}`,
              color: mode === "live" ? "#DC2626" : "#1D4ED8",
              borderRadius: "8px",
            }}
          >
            {mode === "live" ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full live-pulse" style={{ backgroundColor: "#EF4444", display: "inline-block" }} />
                <span><strong>Live mode:</strong> counts and plates update in real time every second.</span>
              </>
            ) : (
              <>
                <span>📊</span>
                <span>
                  <strong>Video mode:</strong> overall totals across the entire video will appear when processing is complete.
                  {videoProcessing && " Processing…"}
                </span>
              </>
            )}
          </div>
        )}

        {/* ── Two-column layout ─────────────────────────────────── */}
        <div className="flex gap-5 items-start">

          {/* LEFT 60% */}
          <div style={{ flex: "0 0 60%" }}>
            <VideoFeed frameData={frameData} fps={fps} source={source} isActive={isActive} />

            {mode === "live" && fps !== null && (
              <div
                className="mt-2 flex items-center gap-2 px-3 py-1.5 text-xs"
                style={{ backgroundColor: "#E0F2FE", border: "1px solid #BAE6FD", borderRadius: "8px", color: "#0284C7" }}
              >
                <span className="font-mono font-bold">{fps} FPS</span>
                <span style={{ color: "#94A3B8" }}>· live processing speed</span>
              </div>
            )}
          </div>

          {/* RIGHT 40% */}
          <div className="flex flex-col gap-4" style={{ flex: "0 0 40%", minWidth: 0 }}>

            {mode === "video" ? (
              <VideoSummaryPanel
                summary={videoSummary}
                isProcessing={videoProcessing}
              />
            ) : (
              <LivePanel
                counts={counts}
                plates={plates}
                violations={violations}
              />
            )}

            <div style={{ height: "1px", backgroundColor: "#BAE6FD" }} />

            {/* Controls */}
            <div
              className="rounded-panel p-4"
              style={{ backgroundColor: "#E0F2FE", border: "1px solid #BAE6FD", borderRadius: "12px" }}
            >
              <h3 className="text-sm font-semibold mb-3" style={{ color: "#0C4A6E" }}>Controls</h3>
              <div className="flex gap-2">
                <button
                  onClick={togglePause}
                  className="flex-1 py-2 text-sm font-semibold transition-all duration-200"
                  style={{ backgroundColor: "#0284C7", color: "#F0F9FF", borderRadius: "8px" }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#0C4A6E")}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#0284C7")}
                >
                  {paused ? "▶ Resume" : "⏸ Pause"}
                </button>
                <button
                  onClick={handleExportCSV}
                  className="flex-1 py-2 text-sm font-semibold transition-all duration-200"
                  style={{ border: "1px solid #0284C7", color: "#0284C7", backgroundColor: "transparent", borderRadius: "8px" }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = "#0284C7"; e.currentTarget.style.color = "#fff"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = "transparent"; e.currentTarget.style.color = "#0284C7"; }}
                >
                  Export CSV
                </button>
                <button
                  onClick={handleStop}
                  className="flex-1 py-2 text-sm font-semibold transition-all duration-200"
                  style={{ border: "1px solid #FCA5A5", color: "#DC2626", backgroundColor: "transparent", borderRadius: "8px" }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#FEE2E2")}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                >
                  ■ Stop
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* ── Alert banner — live mode only ────────────────────── */}
        {alert && mode === "live" && (
          <div className="mt-5">
            <AlertBanner violation={alert} onDismiss={() => setAlert(null)} />
          </div>
        )}
      </div>
    </div>
  );
}