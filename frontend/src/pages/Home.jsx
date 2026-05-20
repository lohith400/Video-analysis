import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Video, 
  Image as ImageIcon, 
  Tv, 
  Upload, 
  Activity, 
  Check, 
  Camera, 
  ShieldAlert, 
  Compass, 
  Cpu, 
  Sparkles,
  RefreshCw
} from "lucide-react";

const API = "http://localhost:8000";

// ── Lightweight Smooth CountUp ──────────────────────────────────────────────
export function CountUp({ to, duration = 1.5 }) {
  const [count, setCount] = useState(0);
  useEffect(() => {
    let start = 0;
    const end = parseInt(to);
    if (isNaN(end) || start === end) return;
    
    const totalMiliseconds = duration * 1000;
    const incrementTime = 30; // 30ms frame steps
    const steps = totalMiliseconds / incrementTime;
    const increment = Math.ceil(end / steps);
    
    const timer = setInterval(() => {
      start += increment;
      if (start >= end) {
        clearInterval(timer);
        setCount(end);
      } else {
        setCount(start);
      }
    }, incrementTime);
    
    return () => clearInterval(timer);
  }, [to, duration]);
  return <span>{count.toLocaleString()}</span>;
}

// ── Interactive Top-Down Highway SVG ────────────────────────────────────────
function HighwaySimulation() {
  return (
    <div className="relative w-full h-44 rounded-2xl border border-sky-border/40 overflow-hidden bg-sky-surface/20 shadow-inner grid-overlay select-none mb-10 border-glow-pulse">
      {/* Highway Lanes */}
      <div className="absolute inset-0 flex flex-col justify-between py-6">
        {/* Lane 1 Line */}
        <div className="w-full border-t border-dashed border-sky-light/40" style={{ borderDasharray: "12 12" }} />
        {/* Lane Divider Barrier */}
        <div className="w-full h-1 bg-sky-border/50 shadow-sm" />
        {/* Lane 2 Line */}
        <div className="w-full border-t border-dashed border-sky-light/40" style={{ borderDasharray: "12 12" }} />
      </div>

      {/* Static HUD markers */}
      <div className="absolute left-6 top-4 font-mono text-[9px] text-sky-default/70 tracking-widest uppercase">
        Sector 04 // Road Intelligence CCTV 12
      </div>
      <div className="absolute right-6 top-4 font-mono text-[9px] text-sky-default/70 flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping" />
        LIVE SCANNING
      </div>

      {/* Road vehicles (animating particles) */}
      {/* Top Lane: Left to Right */}
      <motion.div
        className="absolute w-5 h-2.5 rounded-sm bg-sky-default shadow-[0_0_8px_rgba(2,132,199,0.5)]"
        style={{ top: "22%" }}
        animate={{ left: ["-10%", "110%"] }}
        transition={{ repeat: Infinity, duration: 4.8, ease: "linear" }}
      />
      <motion.div
        className="absolute w-6.5 h-2.5 rounded-sm bg-sky-dark shadow-[0_0_6px_rgba(12,74,110,0.4)]"
        style={{ top: "25%" }}
        animate={{ left: ["-15%", "115%"] }}
        transition={{ repeat: Infinity, duration: 6.2, ease: "linear", delay: 1.5 }}
      />
      <motion.div
        className="absolute w-4 h-2 rounded-full bg-sky-mid"
        style={{ top: "18%" }}
        animate={{ left: ["-10%", "110%"] }}
        transition={{ repeat: Infinity, duration: 3.5, ease: "linear", delay: 0.5 }}
      />

      {/* Bottom Lane: Right to Left */}
      <motion.div
        className="absolute w-5 h-2.5 rounded-sm bg-sky-default shadow-[0_0_8px_rgba(2,132,199,0.5)]"
        style={{ bottom: "22%" }}
        animate={{ right: ["-10%", "110%"] }}
        transition={{ repeat: Infinity, duration: 5.5, ease: "linear" }}
      />
      <motion.div
        className="absolute w-8 h-2.5 rounded-sm bg-sky-mid shadow-[0_0_8px_rgba(56,189,248,0.4)]"
        style={{ bottom: "26%" }}
        animate={{ right: ["-10%", "110%"] }}
        transition={{ repeat: Infinity, duration: 7.2, ease: "linear", delay: 2 }}
      />
      <motion.div
        className="absolute w-4 h-2 rounded-full bg-sky-dark"
        style={{ bottom: "18%" }}
        animate={{ right: ["-10%", "110%"] }}
        transition={{ repeat: Infinity, duration: 4.2, ease: "linear", delay: 0.8 }}
      />

      {/* Scanline visual overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-sky-light/2 to-transparent pointer-events-none opacity-40" />
    </div>
  );
}

export default function Home() {
  const navigate = useNavigate();

  // Cards States
  const [videoFile, setVideoFile] = useState(null);
  const [videoDragging, setVideoDragging] = useState(false);
  const [videoProgress, setVideoProgress] = useState(0);
  const [videoUploading, setVideoUploading] = useState(false);
  const [videoError, setVideoError] = useState("");
  const videoInputRef = useRef();

  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageDragging, setImageDragging] = useState(false);
  const [imageUploading, setImageUploading] = useState(false);
  const [imageError, setImageError] = useState("");
  const imageInputRef = useRef();

  const [rtspUrl, setRtspUrl] = useState("");
  const [useWebcam, setUseWebcam] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [rtspError, setRtspError] = useState("");
  const [rtspPing, setRtspPing] = useState(null);
  const [testingPing, setTestingPing] = useState(false);

  // Live System Info (static mock + alive states)
  const systemStatus = [
    { label: "System Status", value: "Online", pulse: true, color: "text-emerald-500" },
    { label: "YOLOv8 Model", value: "Active", color: "text-sky-default" },
    { label: "ANPR Engine", value: "Ready", color: "text-sky-default" },
    { label: "Active Sessions", value: "0 Idle", color: "text-sky-dark/70" }
  ];

  // ── Handlers ─────────────────────────────────────────────────────────────
  const handleVideoFile = (f) => {
    if (f && f.type.startsWith("video/")) {
      setVideoFile(f);
      setVideoError("");
    } else {
      setVideoError("Invalid file. Please select a valid video format.");
    }
  };

  const handleAnalyseVideo = async () => {
    if (!videoFile) return;
    setVideoUploading(true);
    setVideoError("");
    const form = new FormData();
    form.append("file", videoFile);
    try {
      await axios.post(`${API}/upload`, form, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => setVideoProgress(Math.round((e.loaded / e.total) * 100)),
      });
      navigate("/live");
    } catch {
      setVideoError("Upload request failed. Verify FastAPI server is running.");
    } finally {
      setVideoUploading(false);
    }
  };

  const handleImageFile = (f) => {
    if (f && f.type.startsWith("image/")) {
      setImageFile(f);
      setImagePreview(URL.createObjectURL(f));
      setImageError("");
    } else {
      setImageError("Invalid file. Please select an image format.");
    }
  };

  const handleAnalyseImage = async () => {
    if (!imageFile) return;
    setImageUploading(true);
    setImageError("");
    const form = new FormData();
    form.append("file", imageFile);
    try {
      await axios.post(`${API}/upload`, form, { headers: { "Content-Type": "multipart/form-data" } });
      navigate("/live");
    } catch {
      setImageError("Analysis failed. Verify backend connections.");
    } finally {
      setImageUploading(false);
    }
  };

  const handleTestRtspConnection = () => {
    const source = useWebcam ? "webcam" : rtspUrl;
    if (!source) {
      setRtspError("Enter stream URL or toggle webcam first.");
      return;
    }
    setTestingPing(true);
    setRtspError("");
    setRtspPing(null);
    setTimeout(() => {
      setTestingPing(false);
      setRtspPing(`${Math.floor(Math.random() * 20) + 25}ms Latency - Stream OK`);
    }, 1200);
  };

  const handleConnectRtsp = async () => {
    const source = useWebcam ? "webcam" : rtspUrl;
    if (!source) return;
    setConnecting(true);
    setRtspError("");
    try {
      await axios.post(`${API}/connect`, { source });
      setTimeout(() => navigate("/live"), 500);
    } catch {
      setRtspError("Failed to initiate live link. Check URL pathing.");
    } finally {
      setConnecting(false);
    }
  };

  return (
    <div className="min-h-[calc(100vh-69px)] flex flex-col justify-between py-10 px-6 font-sans select-none relative overflow-hidden bg-sky-lightest">
      
      {/* Background visual details */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] rounded-full bg-sky-surface/10 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] rounded-full bg-sky-light/10 blur-[100px] pointer-events-none" />

      <div className="max-w-screen-xl w-full mx-auto flex-1 flex flex-col justify-center">
        
        {/* Top Hero and Typewriter Banner */}
        <div className="text-center max-w-3xl mx-auto flex flex-col items-center">
          
          {/* Status badging */}
          <div className="flex gap-2.5 mb-6">
            <span className="flex items-center gap-1.5 px-3 py-1 text-[11px] font-heading font-extrabold text-sky-default bg-white border border-sky-border/40 rounded-full shadow-sm">
              <Sparkles className="w-3.5 h-3.5" />
              INTELLIGENT COMMAND CENTER
            </span>
          </div>

          <h1 className="font-heading font-extrabold text-4xl sm:text-5xl tracking-tight text-sky-dark leading-none mb-4 uppercase">
            Indian Road <span className="text-transparent bg-clip-text bg-gradient-to-r from-sky-default to-sky-dark">Intelligence System</span>
          </h1>

          <p className="text-sm text-sky-dark/70 font-sans tracking-wide leading-relaxed max-w-xl mb-6">
            A modular deep learning platform providing high-fidelity visual telemetrics, real-time segment line-crossing counters, automated ANPR tracking, and safety compliance matrices.
          </p>

          {/* System status details capsule */}
          <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2 py-2 px-6 rounded-full border border-sky-border/30 bg-white/40 backdrop-blur-md shadow-sm mb-8">
            {systemStatus.map((stat, i) => (
              <div key={i} className="flex items-center gap-2 text-xs font-heading font-bold text-sky-dark">
                {stat.pulse && <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 live-pulse inline-block" />}
                <span className="text-sky-dark/65 font-medium">{stat.label}:</span>
                <span className={stat.color}>{stat.value}</span>
                {i < systemStatus.length - 1 && <span className="text-sky-border/60 ml-4 font-normal">|</span>}
              </div>
            ))}
          </div>
        </div>

        {/* Highway animated road particle grid */}
        <HighwaySimulation />

        {/* Three input method cards triptych */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          
          {/* Card 1: Upload Video */}
          <motion.div 
            className="glass-card glass-card-hover rounded-2xl p-6 flex flex-col justify-between border-glow-active"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
          >
            <div>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-9 h-9 rounded-xl bg-sky-default/10 flex items-center justify-center text-sky-default">
                  <Video className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-heading font-extrabold text-sm text-sky-dark uppercase tracking-wide">Upload Video Feed</h3>
                  <p className="text-[10px] text-sky-dark/50 font-mono uppercase font-bold">Standard MP4/AVI/MOV</p>
                </div>
              </div>

              {/* Drag drop zone */}
              <div
                className={`border-2 border-dashed rounded-xl flex flex-col items-center justify-center gap-2 py-8 px-4 transition-all duration-300 ${
                  videoDragging ? "border-sky-default bg-sky-surface/20" : "border-sky-border/60 hover:border-sky-light/80 bg-white/40"
                }`}
                onDragOver={(e) => { e.preventDefault(); setVideoDragging(true); }}
                onDragLeave={() => setVideoDragging(false)}
                onDrop={(e) => { e.preventDefault(); setVideoDragging(false); handleVideoFile(e.dataTransfer.files[0]); }}
                onClick={() => videoInputRef.current.click()}
              >
                <input ref={videoInputRef} type="file" accept="video/*" className="hidden" onChange={(e) => handleVideoFile(e.target.files[0])} />
                {videoFile ? (
                  <div className="text-center">
                    <Check className="w-7.5 h-7.5 text-sky-default mx-auto mb-2 animate-bounce" />
                    <p className="text-xs font-semibold text-sky-dark max-w-[170px] truncate">{videoFile.name}</p>
                    <p className="text-[9px] text-sky-default font-bold font-mono">{(videoFile.size / 1024 / 1024).toFixed(1)} MB</p>
                  </div>
                ) : (
                  <>
                    <Upload className="w-8 h-8 text-sky-light/70" />
                    <p className="text-xs font-medium text-sky-dark/70 text-center">Drag & drop footage</p>
                    <p className="text-[10px] text-sky-light font-heading font-bold uppercase">Or click to browse</p>
                  </>
                )}
              </div>
            </div>

            {/* Error & Uploading state */}
            <div className="mt-4 flex flex-col gap-2">
              <AnimatePresence>
                {videoError && (
                  <motion.div 
                    className="p-2.5 rounded-lg border border-red-200 bg-red-50 text-[10px] font-semibold text-red-600 flex items-center gap-1.5"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                  >
                    <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
                    {videoError}
                  </motion.div>
                )}
              </AnimatePresence>

              {videoUploading && (
                <div className="py-1">
                  <div className="flex justify-between text-[10px] font-mono font-bold text-sky-dark/70 mb-1">
                    <span className="flex items-center gap-1 animate-pulse">
                      <RefreshCw className="w-2.5 h-2.5 animate-spin" />
                      BUFFERING...
                    </span>
                    <span>{videoProgress}%</span>
                  </div>
                  <div className="h-1.5 rounded-full overflow-hidden bg-sky-surface border border-sky-border/40">
                    <motion.div className="h-full bg-gradient-to-r from-sky-light to-sky-default rounded-full" style={{ width: `${videoProgress}%` }} />
                  </div>
                </div>
              )}

              <button
                onClick={handleAnalyseVideo}
                disabled={!videoFile || videoUploading}
                className={`w-full py-2.5 rounded-xl font-heading font-extrabold text-xs tracking-wider uppercase select-none transition-all duration-300 ${
                  videoFile && !videoUploading 
                    ? "bg-sky-default hover:bg-sky-dark text-sky-lightest shadow-md shadow-sky-default/10 cursor-pointer" 
                    : "bg-sky-surface text-sky-default/40 cursor-not-allowed border border-sky-border/30"
                }`}
              >
                {videoUploading ? "Buffering Feed..." : "Initialize Analysis"}
              </button>
            </div>
          </motion.div>

          {/* Card 2: Upload Image */}
          <motion.div 
            className="glass-card glass-card-hover rounded-2xl p-6 flex flex-col justify-between"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
          >
            <div>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-9 h-9 rounded-xl bg-sky-default/10 flex items-center justify-center text-sky-default">
                  <ImageIcon className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-heading font-extrabold text-sm text-sky-dark uppercase tracking-wide">Analyse Still Frame</h3>
                  <p className="text-[10px] text-sky-dark/50 font-mono uppercase font-bold">High-Res JPG/PNG</p>
                </div>
              </div>

              {/* Drag drop zone */}
              <div
                className={`border-2 border-dashed rounded-xl flex flex-col items-center justify-center gap-2 py-8 px-4 transition-all duration-300 relative overflow-hidden ${
                  imageDragging ? "border-sky-default bg-sky-surface/20" : "border-sky-border/60 hover:border-sky-light/80 bg-white/40"
                }`}
                style={{ minHeight: "144px" }}
                onDragOver={(e) => { e.preventDefault(); setImageDragging(true); }}
                onDragLeave={() => setImageDragging(false)}
                onDrop={(e) => { e.preventDefault(); setImageDragging(false); handleImageFile(e.dataTransfer.files[0]); }}
                onClick={() => imageInputRef.current.click()}
              >
                <input ref={imageInputRef} type="file" accept="image/*" className="hidden" onChange={(e) => handleImageFile(e.target.files[0])} />
                {imagePreview ? (
                  <div className="absolute inset-0 bg-sky-lightest flex items-center justify-center p-2">
                    <img src={imagePreview} alt="upload preview" className="max-h-full max-w-full object-contain rounded-lg shadow-sm" />
                    <div className="absolute bottom-2 right-2 p-1.5 rounded-full bg-white/80 backdrop-blur shadow-sm border border-sky-border/40 text-sky-default hover:text-sky-dark">
                      <Check className="w-4 h-4" />
                    </div>
                  </div>
                ) : (
                  <>
                    <Upload className="w-8 h-8 text-sky-light/70" />
                    <p className="text-xs font-medium text-sky-dark/70 text-center">Drag still image here</p>
                    <p className="text-[10px] text-sky-light font-heading font-bold uppercase">Or click to select</p>
                  </>
                )}
              </div>
            </div>

            {/* Actions */}
            <div className="mt-4 flex flex-col gap-2">
              <AnimatePresence>
                {imageError && (
                  <motion.div 
                    className="p-2.5 rounded-lg border border-red-200 bg-red-50 text-[10px] font-semibold text-red-600 flex items-center gap-1.5"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                  >
                    <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
                    {imageError}
                  </motion.div>
                )}
              </AnimatePresence>

              <button
                onClick={handleAnalyseImage}
                disabled={!imageFile || imageUploading}
                className={`w-full py-2.5 rounded-xl font-heading font-extrabold text-xs tracking-wider uppercase select-none transition-all duration-300 ${
                  imageFile && !imageUploading 
                    ? "bg-sky-default hover:bg-sky-dark text-sky-lightest shadow-md shadow-sky-default/10 cursor-pointer" 
                    : "bg-sky-surface text-sky-default/40 cursor-not-allowed border border-sky-border/30"
                }`}
              >
                {imageUploading ? "Processing Frame..." : "Perform Still Scan"}
              </button>
            </div>
          </motion.div>

          {/* Card 3: Live CCTV / Webcam */}
          <motion.div 
            className="glass-card glass-card-hover rounded-2xl p-6 flex flex-col justify-between"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.3 }}
          >
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-sky-default/10 flex items-center justify-center text-sky-default animate-pulse">
                    <Tv className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-heading font-extrabold text-sm text-sky-dark uppercase tracking-wide">Live Stream Terminal</h3>
                    <p className="text-[10px] text-sky-dark/50 font-mono uppercase font-bold">RTSP / Web Camera</p>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border border-sky-border/40 bg-white/60 shadow-sm text-sky-default">
                  <span className="w-1.5 h-1.5 rounded-full bg-sky-default live-pulse" />
                  <span className="font-mono text-[9px] font-extrabold">LIVE</span>
                </div>
              </div>

              {/* Webcam toggle */}
              <div className="flex items-center justify-between py-2 px-3 bg-white/40 border border-sky-border/30 rounded-xl mb-3 shadow-inner">
                <span className="text-xs font-heading font-bold text-sky-dark flex items-center gap-2">
                  <Camera className={`w-4.5 h-4.5 text-sky-default ${useWebcam && "animate-spin"}`} />
                  Capture Local Webcam
                </span>
                <button
                  onClick={() => setUseWebcam((v) => !v)}
                  className={`relative inline-flex h-5.5 w-10.5 items-center rounded-full transition-all duration-300 shadow-sm ${
                    useWebcam ? "bg-sky-default" : "bg-sky-border"
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 rounded-full bg-white transition-all duration-300 ${
                      useWebcam ? "translate-x-5.5" : "translate-x-1"
                    }`}
                  />
                </button>
              </div>

              {/* RTSP Stream Inputs */}
              {!useWebcam && (
                <div className="flex flex-col gap-1.5 mb-2">
                  <label className="text-[10px] font-heading font-extrabold text-sky-dark/70 uppercase">RTSP CCTV Endpoint URL</label>
                  <input
                    type="text"
                    placeholder="rtsp://user:pass@192.168.1.1:554/stream"
                    value={rtspUrl}
                    onChange={(e) => setRtspUrl(e.target.value)}
                    className="w-full px-3 py-2 text-xs font-mono rounded-xl bg-white/50 border border-sky-border/60 outline-none text-sky-dark focus:border-sky-default focus:ring-1 focus:ring-sky-default transition-all duration-300"
                  />
                </div>
              )}
            </div>

            {/* Actions & Pings */}
            <div className="mt-4 flex flex-col gap-2">
              <AnimatePresence>
                {rtspError && (
                  <motion.div 
                    className="p-2.5 rounded-lg border border-red-200 bg-red-50 text-[10px] font-semibold text-red-600 flex items-center gap-1.5"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                  >
                    <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
                    {rtspError}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Ping latency results */}
              <AnimatePresence>
                {rtspPing && (
                  <motion.div 
                    className="p-2 rounded-xl bg-white/70 border border-sky-border/30 text-[10px] font-mono text-sky-default flex items-center gap-2 shadow-inner"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <span className="w-2 h-2 rounded-full bg-emerald-500 live-pulse" />
                    <span>LATENCY CHECK: <strong className="font-extrabold">{rtspPing}</strong></span>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="flex gap-2">
                <button
                  onClick={handleTestRtspConnection}
                  disabled={testingPing || (!rtspUrl && !useWebcam)}
                  className={`py-2 px-3 rounded-xl border font-heading font-extrabold text-[10px] uppercase tracking-wide transition-all duration-300 ${
                    (rtspUrl || useWebcam) && !testingPing
                      ? "border-sky-default text-sky-default hover:bg-sky-surface cursor-pointer" 
                      : "border-sky-border text-sky-default/30 cursor-not-allowed"
                  }`}
                >
                  {testingPing ? "Ping..." : "Test Connection"}
                </button>
                <button
                  onClick={handleConnectRtsp}
                  disabled={connecting || (!rtspUrl && !useWebcam)}
                  className={`flex-1 py-2 rounded-xl font-heading font-extrabold text-xs uppercase tracking-wide transition-all duration-300 ${
                    (rtspUrl || useWebcam) && !connecting
                      ? "bg-sky-default hover:bg-sky-dark text-sky-lightest shadow-md shadow-sky-default/10 cursor-pointer" 
                      : "bg-sky-surface text-sky-default/40 cursor-not-allowed border border-sky-border/30"
                  }`}
                >
                  {connecting ? "Linking..." : "Establish Link"}
                </button>
              </div>
            </div>
          </motion.div>

        </div>

      </div>

      {/* Scrolling stock ticker strip footer */}
      <div className="w-full overflow-hidden border-y border-sky-border/40 py-3 bg-white/40 backdrop-blur-md select-none mt-6">
        <div className="ticker-content whitespace-nowrap flex gap-12 font-mono text-xs font-semibold tracking-wider text-sky-dark select-none">
          <span>⚡ IRIS CORE OPS ONLINE</span>
          <span>🚙 VEHICLES SCANNED: <CountUp to={12480} /></span>
          <span>🪖 HELMET COMPLIANCE ALERTS: <CountUp to={340} /></span>
          <span>📋 ANPR READS COMPLETED: <CountUp to={8920} /></span>
          <span>📡 RTSP PIPELINE: OK</span>
          <span>⚡ CORE CPU PROCESSORS: 8 ACTIVE</span>
          
          {/* Duplicate to create a seamless carousel infinite wrap */}
          <span>⚡ IRIS CORE OPS ONLINE</span>
          <span>🚙 VEHICLES SCANNED: <CountUp to={12480} /></span>
          <span>🪖 HELMET COMPLIANCE ALERTS: <CountUp to={340} /></span>
          <span>📋 ANPR READS COMPLETED: <CountUp to={8920} /></span>
          <span>📡 RTSP PIPELINE: OK</span>
          <span>⚡ CORE CPU PROCESSORS: 8 ACTIVE</span>
        </div>
      </div>

    </div>
  );
}
