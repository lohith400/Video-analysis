function CameraIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#0284C7" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
      <circle cx="12" cy="13" r="4" />
    </svg>
  );
}

export default function VideoFeed({ frameData, fps, source, isActive }) {
  return (
    <div
      className="relative rounded-panel overflow-hidden"
      style={{
        backgroundColor: "#E0F2FE",
        border: isActive ? "1px solid #38BDF8" : "1px solid #BAE6FD",
        borderRadius: "12px",
        aspectRatio: "16/9",
      }}
    >
      {frameData ? (
        <img
          src={`data:image/jpeg;base64,${frameData}`}
          alt="Live feed"
          className="w-full h-full object-contain"
        />
      ) : (
        <div className="w-full h-full flex flex-col items-center justify-center gap-3">
          <CameraIcon />
          <div className="text-center">
            <p className="text-sm font-medium" style={{ color: "#0284C7" }}>
              Waiting for video feed
            </p>
            <p className="text-xs mt-1" style={{ color: "#7DD3FC" }}>
              Connect a source to begin analysis
            </p>
          </div>
        </div>
      )}

      {/* FPS overlay */}
      {isActive && fps !== null && (
        <div
          className="absolute top-3 left-3 px-2.5 py-1 rounded-card text-xs font-mono font-semibold"
          style={{ backgroundColor: "#0284C7", color: "#F0F9FF", borderRadius: "6px" }}
        >
          {fps.toFixed(1)} FPS
        </div>
      )}

      {/* Source badge */}
      {source && (
        <div
          className="absolute top-3 right-3 px-2.5 py-1 rounded-card text-xs font-semibold uppercase tracking-wider"
          style={{ backgroundColor: "#0284C7", color: "#F0F9FF", borderRadius: "6px" }}
        >
          {source}
        </div>
      )}

      {/* Live pulse indicator */}
      {isActive && (
        <div className="absolute bottom-3 left-3 flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full live-pulse" style={{ backgroundColor: "#38BDF8" }} />
          <span className="text-xs font-semibold" style={{ color: "#F0F9FF", textShadow: "0 1px 4px rgba(0,0,0,0.4)" }}>
            LIVE
          </span>
        </div>
      )}
    </div>
  );
}
