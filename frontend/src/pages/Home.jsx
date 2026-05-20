import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const API = "http://localhost:8000";

// ── Icons ──────────────────────────────────────────────────────────────────
function VideoIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0284C7" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 10l4.553-2.069A1 1 0 0121 8.82V15.18a1 1 0 01-1.447.89L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
    </svg>
  );
}

function ImageIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0284C7" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <path d="M21 15l-5-5L5 21" />
    </svg>
  );
}

function BroadcastIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="2" />
      <path d="M16.24 7.76a6 6 0 010 8.49M7.76 7.76a6 6 0 000 8.49M19.07 4.93a10 10 0 010 14.14M4.93 4.93a10 10 0 000 14.14" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#7DD3FC" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" />
    </svg>
  );
}

// ── Upload Video Card ──────────────────────────────────────────────────────
function UploadVideoCard() {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef();
  const navigate = useNavigate();

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("video/")) setFile(f);
  };

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (f) setFile(f);
  };

  const handleAnalyse = async () => {
    if (!file) return;
    setUploading(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      await axios.post(`${API}/upload`, form, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => setProgress(Math.round((e.loaded / e.total) * 100)),
      });
      navigate("/live");
    } catch {
      setError("Upload failed. Is the backend running?");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div
      className="rounded-panel p-6 flex flex-col gap-4 transition-all duration-200 hover:cursor-default"
      style={{ backgroundColor: "#c7e8fd", border: "1px solid #BAE6FD", borderRadius: "12px" }}
      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#BAE6FD")}
      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#c7e8fd")}
    >
      <div className="flex items-center gap-3">
        <VideoIcon />
        <div>
          <h3 className="font-semibold text-base" style={{ color: "#0C4A6E" }}>Upload Video</h3>
          <p className="text-xs mt-0.5" style={{ color: "#475569" }}>MP4, AVI, MOV supported</p>
        </div>
      </div>

      {/* Drop Zone */}
      <div
        className="rounded-card flex flex-col items-center justify-center gap-2 py-8 px-4 transition-all duration-200 cursor-pointer"
        style={{
          border: `2px dashed ${dragging ? "#0284C7" : "#7DD3FC"}`,
          backgroundColor: "#F0F9FF",
          borderRadius: "8px",
        }}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current.click()}
      >
        <UploadIcon />
        {file ? (
          <p className="text-sm font-medium text-center" style={{ color: "#0284C7" }}>{file.name}</p>
        ) : (
          <>
            <p className="text-sm" style={{ color: "#475569" }}>Drag & drop video here</p>
            <p className="text-xs" style={{ color: "#7DD3FC" }}>or click to browse</p>
          </>
        )}
        <input ref={inputRef} type="file" accept="video/*" className="hidden" onChange={handleFile} />
      </div>

      <button
        onClick={() => inputRef.current.click()}
        className="w-full py-2 rounded-card text-sm font-medium transition-all duration-200"
        style={{ border: "1px solid #0284C7", color: "#0284C7", backgroundColor: "transparent", borderRadius: "8px" }}
        onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = "#0284C7"; e.currentTarget.style.color = "#F0F9FF"; }}
        onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = "transparent"; e.currentTarget.style.color = "#0284C7"; }}
      >
        Browse File
      </button>

      {/* Progress */}
      {uploading && (
        <div>
          <div className="flex justify-between text-xs mb-1" style={{ color: "#475569" }}>
            <span>Uploading…</span>
            <span>{progress}%</span>
          </div>
          <div className="h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: "#BAE6FD" }}>
            <div className="h-full rounded-full transition-all duration-300" style={{ width: `${progress}%`, backgroundColor: "#0284C7" }} />
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-card px-3 py-2 text-xs" style={{ backgroundColor: "#c7e8fd", border: "1px solid #7DD3FC", color: "#0C4A6E", borderRadius: "8px" }}>
          {error}
        </div>
      )}

      <button
        onClick={handleAnalyse}
        disabled={!file || uploading}
        className="w-full py-2.5 rounded-card text-sm font-semibold transition-all duration-200"
        style={{
          backgroundColor: file && !uploading ? "#0284C7" : "#BAE6FD",
          color: "#F0F9FF",
          borderRadius: "8px",
          cursor: file && !uploading ? "pointer" : "not-allowed",
        }}
        onMouseEnter={(e) => { if (file && !uploading) e.currentTarget.style.backgroundColor = "#0C4A6E"; }}
        onMouseLeave={(e) => { if (file && !uploading) e.currentTarget.style.backgroundColor = "#0284C7"; }}
      >
        {uploading ? "Uploading…" : "Analyse Video"}
      </button>
    </div>
  );
}

// ── Upload Image Card ──────────────────────────────────────────────────────
function UploadImageCard() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef();
  const navigate = useNavigate();

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handleAnalyse = async () => {
    if (!file) return;
    setUploading(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      await axios.post(`${API}/upload`, form, { headers: { "Content-Type": "multipart/form-data" } });
      navigate("/live");
    } catch {
      setError("Upload failed. Is the backend running?");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div
      className="rounded-panel p-6 flex flex-col gap-4 transition-all duration-200"
      style={{ backgroundColor: "#c7e8fd", border: "1px solid #BAE6FD", borderRadius: "12px" }}
      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#BAE6FD")}
      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#c7e8fd")}
    >
      <div className="flex items-center gap-3">
        <ImageIcon />
        <div>
          <h3 className="font-semibold text-base" style={{ color: "#0C4A6E" }}>Upload Image</h3>
          <p className="text-xs mt-0.5" style={{ color: "#475569" }}>JPG, PNG, WEBP supported</p>
        </div>
      </div>

      <div
        className="rounded-card flex flex-col items-center justify-center gap-2 py-6 px-4 transition-all duration-200 cursor-pointer overflow-hidden"
        style={{ border: `2px dashed ${dragging ? "#0284C7" : "#7DD3FC"}`, backgroundColor: "#F0F9FF", borderRadius: "8px", minHeight: "140px" }}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
        onClick={() => inputRef.current.click()}
      >
        {preview ? (
          <img src={preview} alt="preview" className="max-h-28 max-w-full object-contain rounded" />
        ) : (
          <>
            <UploadIcon />
            <p className="text-sm" style={{ color: "#475569" }}>Drag & drop image here</p>
            <p className="text-xs" style={{ color: "#7DD3FC" }}>or click to browse</p>
          </>
        )}
        <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={(e) => handleFile(e.target.files[0])} />
      </div>

      {file && (
        <p className="text-xs truncate" style={{ color: "#0284C7" }}>📎 {file.name}</p>
      )}

      {error && (
        <div className="rounded-card px-3 py-2 text-xs" style={{ backgroundColor: "#c7e8fd", border: "1px solid #7DD3FC", color: "#0C4A6E", borderRadius: "8px" }}>
          {error}
        </div>
      )}

      <button
        onClick={handleAnalyse}
        disabled={!file || uploading}
        className="w-full py-2.5 rounded-card text-sm font-semibold transition-all duration-200 mt-auto"
        style={{
          backgroundColor: file && !uploading ? "#0284C7" : "#BAE6FD",
          color: "#F0F9FF",
          borderRadius: "8px",
          cursor: file && !uploading ? "pointer" : "not-allowed",
        }}
        onMouseEnter={(e) => { if (file && !uploading) e.currentTarget.style.backgroundColor = "#0C4A6E"; }}
        onMouseLeave={(e) => { if (file && !uploading) e.currentTarget.style.backgroundColor = "#0284C7"; }}
      >
        {uploading ? "Uploading…" : "Analyse Image"}
      </button>
    </div>
  );
}

// ── Live CCTV Card ─────────────────────────────────────────────────────────
function LiveCCTVCard() {
  const [rtspUrl, setRtspUrl] = useState("");
  const [useWebcam, setUseWebcam] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleConnect = async () => {
    const source = useWebcam ? "webcam" : rtspUrl;
    if (!source) return;
    setConnecting(true);
    setError("");
    try {
      await axios.post(`${API}/connect`, { source });
      setIsLive(true);
      setTimeout(() => navigate("/live"), 800);
    } catch {
      setError("Connection failed. Check URL or backend.");
    } finally {
      setConnecting(false);
    }
  };

  return (
    <div
      className="rounded-panel p-6 flex flex-col gap-4 transition-all duration-200"
      style={{ backgroundColor: "#c7e8fd", border: "1px solid #BAE6FD", borderRadius: "12px" }}
      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#BAE6FD")}
      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#c7e8fd")}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <BroadcastIcon />
          <div>
            <h3 className="font-semibold text-base" style={{ color: "#0C4A6E" }}>Live CCTV / Webcam</h3>
            <p className="text-xs mt-0.5" style={{ color: "#475569" }}>RTSP stream or local webcam</p>
          </div>
        </div>
        {isLive && (
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full live-pulse" style={{ backgroundColor: "#38BDF8" }} />
            <span className="text-xs font-semibold" style={{ color: "#0284C7" }}>LIVE</span>
          </div>
        )}
      </div>

      {/* Webcam toggle */}
      <div className="flex items-center justify-between">
        <label className="text-sm" style={{ color: "#475569" }}>Use local webcam</label>
        <button
          onClick={() => setUseWebcam((v) => !v)}
          className="relative inline-flex h-5 w-9 items-center rounded-full transition-all duration-200"
          style={{ backgroundColor: useWebcam ? "#0284C7" : "#BAE6FD" }}
        >
          <span
            className="inline-block h-3.5 w-3.5 rounded-full transition-all duration-200"
            style={{ backgroundColor: "#F0F9FF", transform: useWebcam ? "translateX(18px)" : "translateX(2px)" }}
          />
        </button>
      </div>

      {/* RTSP URL */}
      {!useWebcam && (
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium" style={{ color: "#475569" }}>RTSP Stream URL</label>
          <input
            type="text"
            placeholder="rtsp://user:pass@192.168.1.1:554/stream"
            value={rtspUrl}
            onChange={(e) => setRtspUrl(e.target.value)}
            className="w-full px-3 py-2 text-sm rounded-card outline-none transition-all duration-200"
            style={{
              backgroundColor: "#F0F9FF",
              border: "1px solid #BAE6FD",
              color: "#0C4A6E",
              borderRadius: "8px",
            }}
            onFocus={(e) => (e.target.style.boxShadow = "0 0 0 2px #38BDF8")}
            onBlur={(e) => (e.target.style.boxShadow = "none")}
          />
        </div>
      )}

      {error && (
        <div className="rounded-card px-3 py-2 text-xs" style={{ backgroundColor: "#c7e8fd", border: "1px solid #7DD3FC", color: "#0C4A6E", borderRadius: "8px" }}>
          {error}
        </div>
      )}

      <button
        onClick={handleConnect}
        disabled={(!rtspUrl && !useWebcam) || connecting}
        className="w-full py-2.5 rounded-card text-sm font-semibold transition-all duration-200 mt-auto"
        style={{
          backgroundColor: (rtspUrl || useWebcam) && !connecting ? "#0284C7" : "#BAE6FD",
          color: "#F0F9FF",
          borderRadius: "8px",
          cursor: (rtspUrl || useWebcam) && !connecting ? "pointer" : "not-allowed",
        }}
        onMouseEnter={(e) => { if ((rtspUrl || useWebcam) && !connecting) e.currentTarget.style.backgroundColor = "#0C4A6E"; }}
        onMouseLeave={(e) => { if ((rtspUrl || useWebcam) && !connecting) e.currentTarget.style.backgroundColor = "#0284C7"; }}
      >
        {connecting ? (
          <span className="flex items-center justify-center gap-2">
            <span className="spinner" style={{ width: 14, height: 14 }} />
            Connecting…
          </span>
        ) : "Connect & Analyse"}
      </button>
    </div>
  );
}

// ── Home Page ──────────────────────────────────────────────────────────────
export default function Home() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: "#F0F9FF" }}>
      <div className="max-w-screen-xl mx-auto px-6 py-12">
        {/* Hero */}
        <div className="text-center mb-12 fade-in">
          <div className="flex items-center justify-center gap-3 mb-5">
            <span
              className="px-3 py-1 text-xs font-medium rounded-full"
              style={{ backgroundColor: "#c7e8fd", color: "#0284C7", border: "1px solid #BAE6FD" }}
            >
              ✦ System Ready
            </span>
            <span
              className="px-3 py-1 text-xs font-medium rounded-full"
              style={{ backgroundColor: "#c7e8fd", color: "#0284C7", border: "1px solid #BAE6FD" }}
            >
              ✦ Models Loaded
            </span>
          </div>

          <h1 className="text-4xl font-extrabold tracking-tight mb-4" style={{ color: "#0C4A6E" }}>
            Indian Road Intelligence System
          </h1>
          <p className="text-base max-w-2xl mx-auto leading-relaxed" style={{ color: "#475569" }}>
            Upload a video, image or connect a live CCTV camera for real-time vehicle detection,
            ANPR, and helmet violation analysis.
          </p>
        </div>

        {/* Divider */}
        <div className="mb-8" style={{ height: "1px", backgroundColor: "#BAE6FD" }} />

        {/* Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <UploadVideoCard />
          <UploadImageCard />
          <LiveCCTVCard />
        </div>

        {/* Footer note */}
        <p className="text-center text-xs mt-10" style={{ color: "#7DD3FC" }}>
          Backend at{" "}
          <code className="font-mono" style={{ color: "#0284C7" }}>
            http://localhost:8000
          </code>{" "}
          · WebSocket at{" "}
          <code className="font-mono" style={{ color: "#0284C7" }}>
            ws://localhost:8000/video-feed
          </code>
        </p>
      </div>
    </div>
  );
}
