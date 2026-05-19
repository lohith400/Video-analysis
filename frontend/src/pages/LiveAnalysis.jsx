import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import VideoFeed from "../components/VideoFeed";
import VehicleCounts from "../components/VehicleCounts";
import PlateTable from "../components/PlateTable";
import ViolationList from "../components/ViolationList";
import AlertBanner from "../components/AlertBanner";

const WS_URL  = "ws://localhost:8000/video-feed";
const API_URL = "http://localhost:8000";

// ── Reconnect parameters ───────────────────────────────────────────────────
const RECONNECT_DELAY_MS = 3000;
const MAX_RECONNECTS     = 10;

export default function LiveAnalysis() {
  const navigate = useNavigate();

  // Video & stream
  const [frameData,   setFrameData]   = useState(null);   // base64 string
  const [fps,         setFps]         = useState(null);
  const [source,      setSource]      = useState("VIDEO");
  const [wsStatus,    setWsStatus]    = useState("idle"); // idle|connecting|open|closed|error

  // Analysis data
  const [counts,     setCounts]     = useState({});
  const [plates,     setPlates]     = useState([]);
  const [violations, setViolations] = useState([]);

  // UI state
  const [paused,   setPaused]   = useState(false);
  const [alert,    setAlert]    = useState(null);

  // Refs (never stale inside callbacks)
  const wsRef          = useRef(null);
  const pausedRef      = useRef(false);
  const reconnectCount = useRef(0);
  const reconnectTimer = useRef(null);
  const mountedRef     = useRef(true);

  // ── WebSocket connect ────────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    if (wsRef.current && wsRef.current.readyState < 2) {
      wsRef.current.close();
    }

    setWsStatus("connecting");
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      console.log("[WS] Connected");
      setWsStatus("open");
      reconnectCount.current = 0;
    };

    ws.onmessage = (e) => {
      if (!mountedRef.current || pausedRef.current) return;
      try {
        const data = JSON.parse(e.data);

        // ── Frame ────────────────────────────────────────────────────
        if (data.frame) setFrameData(data.frame);

        // ── FPS & source ─────────────────────────────────────────────
        if (data.fps  !== undefined) setFps(data.fps);
        if (data.source)             setSource(data.source.toUpperCase());

        // ── Counts ───────────────────────────────────────────────────
        if (data.counts) setCounts(data.counts);

        // ── Plates (de-dup by plate string) ──────────────────────────
        if (Array.isArray(data.plates) && data.plates.length > 0) {
          setPlates((prev) => {
            const seen = new Set(prev.map((p) => p.plate));
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

        // ── Violations ───────────────────────────────────────────────
        if (Array.isArray(data.violations)) {
          setViolations(data.violations);
          if (data.violations.length > 0) {
            setAlert(data.violations[data.violations.length - 1]);
          }
        }
      } catch {
        /* malformed packet — skip */
      }
    };

    ws.onclose = (e) => {
      if (!mountedRef.current) return;
      console.log(`[WS] Closed (code=${e.code})`);
      setWsStatus("closed");
      // Auto-reconnect
      if (reconnectCount.current < MAX_RECONNECTS) {
        reconnectCount.current += 1;
        console.log(`[WS] Reconnecting in ${RECONNECT_DELAY_MS}ms (attempt ${reconnectCount.current})`);
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
      }
    };

    ws.onerror = () => {
      if (!mountedRef.current) return;
      setWsStatus("error");
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Mount / unmount ──────────────────────────────────────────────────────
  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // ── Controls ─────────────────────────────────────────────────────────────
  const togglePause = () => {
    pausedRef.current = !pausedRef.current;
    setPaused(pausedRef.current);
  };

  const handleStop = async () => {
    try { await fetch(`${API_URL}/stop`, { method: "POST" }); } catch { /* ignore */ }
    wsRef.current?.close();
    setFrameData(null);
    setCounts({});
    setPlates([]);
    setViolations([]);
    setFps(null);
    setWsStatus("idle");
  };

  const handleExportCSV = () => {
    if (!plates.length) return;
    const rows = [
      ["#", "Plate", "Timestamp"],
      ...plates.map((p, i) => [i + 1, p.plate, new Date(p.timestamp).toISOString()]),
    ];
    const blob = new Blob([rows.map((r) => r.join(",")).join("\n")], { type: "text/csv" });
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob),
      download: `plates_${Date.now()}.csv`,
    });
    a.click();
  };

  const isActive = wsStatus === "open" && !!frameData;

  // ── Status pill ───────────────────────────────────────────────────────────
  const statusLabel = {
    idle:       "○ Idle",
    connecting: "◌ Connecting…",
    open:       "● Connected",
    closed:     "○ Disconnected",
    error:      "○ Error",
  }[wsStatus] ?? "○ Unknown";

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen" style={{ backgroundColor: "#F0F9FF" }}>
      <div className="max-w-screen-xl mx-auto px-6 py-6">

        {/* ── Header bar ──────────────────────────────────────────────── */}
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-bold" style={{ color: "#0C4A6E" }}>
            Live Analysis
          </h2>
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
                className="text-xs px-3 py-1.5 rounded-full font-semibold transition-all duration-200"
                style={{ backgroundColor: "#0284C7", color: "#F0F9FF" }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#0C4A6E")}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#0284C7")}
              >
                Reconnect
              </button>
            )}

            <button
              onClick={() => navigate("/")}
              className="text-xs px-3 py-1.5 rounded-full font-semibold transition-all duration-200"
              style={{ border: "1px solid #BAE6FD", color: "#0284C7", backgroundColor: "transparent" }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#E0F2FE")}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
            >
              ← Upload New
            </button>
          </div>
        </div>

        {/* ── Info banner when no frame yet ───────────────────────────── */}
        {!frameData && wsStatus !== "idle" && (
          <div
            className="mb-4 px-4 py-3 rounded-card text-sm flex items-center gap-3"
            style={{ backgroundColor: "#E0F2FE", border: "1px solid #BAE6FD", color: "#0C4A6E", borderRadius: "8px" }}
          >
            <span className="w-2 h-2 rounded-full live-pulse shrink-0" style={{ backgroundColor: "#38BDF8" }} />
            <span>
              {wsStatus === "connecting"
                ? "Connecting to backend… Make sure the server is running."
                : wsStatus === "open"
                ? "Connected — upload a video on the home page to start streaming frames."
                : "Backend disconnected. Will retry automatically."}
            </span>
          </div>
        )}

        {wsStatus === "error" && (
          <div
            className="mb-4 px-4 py-3 rounded-card text-sm"
            style={{ backgroundColor: "#E0F2FE", border: "1px solid #7DD3FC", color: "#0C4A6E", borderRadius: "8px" }}
          >
            WebSocket error — ensure the FastAPI backend is running at{" "}
            <code className="font-mono text-xs" style={{ color: "#0284C7" }}>
              ws://localhost:8000/video-feed
            </code>
            . Run:{" "}
            <code className="font-mono text-xs" style={{ color: "#0284C7" }}>
              uvicorn server:app --port 8000
            </code>
          </div>
        )}

        {/* ── Two-column layout ────────────────────────────────────────── */}
        <div className="flex gap-5 items-start">

          {/* LEFT 60% — Video feed */}
          <div style={{ flex: "0 0 60%" }}>
            <VideoFeed
              frameData={frameData}
              fps={fps}
              source={source}
              isActive={isActive}
            />
          </div>

          {/* RIGHT 40% — Panels */}
          <div className="flex flex-col gap-4" style={{ flex: "0 0 40%", minWidth: 0 }}>

            {/* Vehicle Counts */}
            <div
              className="rounded-panel p-4"
              style={{ backgroundColor: "#E0F2FE", border: "1px solid #BAE6FD", borderRadius: "12px" }}
            >
              <VehicleCounts counts={counts} />
            </div>

            <div style={{ height: "1px", backgroundColor: "#BAE6FD" }} />

            {/* Plates */}
            <div
              className="rounded-panel p-4"
              style={{ backgroundColor: "#E0F2FE", border: "1px solid #BAE6FD", borderRadius: "12px" }}
            >
              <PlateTable plates={plates} />
            </div>

            <div style={{ height: "1px", backgroundColor: "#BAE6FD" }} />

            {/* Violations */}
            <div
              className="rounded-panel p-4"
              style={{ backgroundColor: "#E0F2FE", border: "1px solid #BAE6FD", borderRadius: "12px" }}
            >
              <ViolationList violations={violations} />
            </div>

            <div style={{ height: "1px", backgroundColor: "#BAE6FD" }} />

            {/* Controls */}
            <div
              className="rounded-panel p-4"
              style={{ backgroundColor: "#E0F2FE", border: "1px solid #BAE6FD", borderRadius: "12px" }}
            >
              <h3 className="text-sm font-semibold mb-3" style={{ color: "#0C4A6E" }}>
                Controls
              </h3>
              <div className="flex gap-2">
                <button
                  onClick={togglePause}
                  className="flex-1 py-2 text-sm font-semibold rounded-card transition-all duration-200"
                  style={{ backgroundColor: "#0284C7", color: "#F0F9FF", borderRadius: "8px" }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#0C4A6E")}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#0284C7")}
                >
                  {paused ? "▶ Resume" : "⏸ Pause"}
                </button>
                <button
                  onClick={handleExportCSV}
                  className="flex-1 py-2 text-sm font-semibold rounded-card transition-all duration-200"
                  style={{ border: "1px solid #0284C7", color: "#0284C7", backgroundColor: "transparent", borderRadius: "8px" }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = "#0284C7"; e.currentTarget.style.color = "#F0F9FF"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = "transparent"; e.currentTarget.style.color = "#0284C7"; }}
                >
                  Export CSV
                </button>
                <button
                  onClick={handleStop}
                  className="flex-1 py-2 text-sm font-semibold rounded-card transition-all duration-200"
                  style={{ border: "1px solid #7DD3FC", color: "#0284C7", backgroundColor: "transparent", borderRadius: "8px" }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#E0F2FE")}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                >
                  ■ Stop
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* ── Alert banner ─────────────────────────────────────────────── */}
        {alert && (
          <div className="mt-5">
            <AlertBanner violation={alert} onDismiss={() => setAlert(null)} />
          </div>
        )}
      </div>
    </div>
  );
}