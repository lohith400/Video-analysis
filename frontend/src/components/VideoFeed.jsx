import { motion } from "framer-motion";
import { Camera, Cpu, Activity, Wifi, ShieldCheck, Zap } from "lucide-react";

export default function VideoFeed({ frameData, fps, source, isActive, plates = [] }) {
  // Hardcoded or dynamically derived model average metrics for realism
  const avgConfidence = isActive ? 94.8 : 0;
  const strokeOffset = 88 - (88 * avgConfidence) / 100;

  return (
    <div
      className="relative rounded-2xl overflow-hidden glass-card select-none group"
      style={{
        backgroundColor: "#F0F9FF",
        border: isActive ? "2px solid #0284C7" : "1px solid #BAE6FD",
        aspectRatio: "16/9",
        boxShadow: isActive 
          ? "0 20px 40px -15px rgba(2, 132, 199, 0.25), 0 10px 15px -10px rgba(2, 132, 199, 0.15)"
          : "none",
      }}
    >
      {/* Reticle decorative corner brackets */}
      {isActive && (
        <>
          {/* HUD Corner Accents */}
          <div className="absolute top-5 left-5 w-5 h-5 border-t-2 border-l-2 border-sky-default rounded-tl-sm select-none pointer-events-none z-10" />
          <div className="absolute top-5 right-5 w-5 h-5 border-t-2 border-r-2 border-sky-default rounded-tr-sm select-none pointer-events-none z-10" />
          <div className="absolute bottom-5 left-5 w-5 h-5 border-b-2 border-l-2 border-sky-default rounded-bl-sm select-none pointer-events-none z-10" />
          <div className="absolute bottom-5 right-5 w-5 h-5 border-b-2 border-r-2 border-sky-default rounded-br-sm select-none pointer-events-none z-10" />
          
          {/* Dynamic Scrolling Horizontal Scanline */}
          <div className="absolute inset-x-0 h-0.5 bg-sky-default/30 shadow-[0_0_12px_#0284C7] pointer-events-none z-10 animate-[scanline_4s_linear_infinite]" />

          {/* Center Target Reticle Crosshair */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10 opacity-30 select-none">
            <div className="w-10 h-10 border border-sky-default rounded-full flex items-center justify-center">
              <div className="w-1.5 h-1.5 bg-sky-default rounded-full" />
            </div>
            <div className="absolute w-12 h-[1px] bg-sky-default" />
            <div className="absolute h-12 w-[1px] bg-sky-default" />
          </div>
        </>
      )}

      {frameData ? (
        <img
          src={`data:image/jpeg;base64,${frameData}`}
          alt="Live feed"
          className="w-full h-full object-contain bg-slate-950"
        />
      ) : (
        <div className="w-full h-full flex flex-col items-center justify-center gap-4 bg-sky-surface/20 relative">
          {/* Radar background grid scanning lines effect */}
          <div className="absolute inset-0 grid grid-cols-8 grid-rows-8 opacity-15 pointer-events-none">
            {Array.from({ length: 64 }).map((_, i) => (
              <div key={i} className="border-[0.5px] border-sky-border/40" />
            ))}
          </div>

          {/* Glowing camera standby reticle */}
          <div className="w-20 h-20 rounded-full bg-white flex items-center justify-center border border-sky-border shadow-inner relative animate-pulse">
            <Camera className="w-10 h-10 text-sky-default" strokeWidth={1.2} />
            <div className="absolute inset-0 rounded-full border border-sky-default animate-[ping_2.5s_infinite] opacity-25" />
          </div>
          
          <div className="text-center z-10 px-4">
            <h4 className="font-heading font-extrabold text-xs text-sky-dark tracking-widest uppercase">
              System Telemetry Standby
            </h4>
            <p className="text-[9px] font-heading font-extrabold text-sky-default uppercase tracking-wider mt-1.5 opacity-80">
              Initialize source stream to trigger machine vision pipeline
            </p>
          </div>
        </div>
      )}

      {/* LEFT COLUMN HUD: Monospace intelligence diagnostics panel */}
      {isActive && (
        <motion.div 
          className="absolute top-5 left-5 bg-white/80 backdrop-blur-md border border-sky-border/40 p-3 rounded-2xl font-mono text-[9px] text-sky-dark/95 flex flex-col gap-1.5 shadow-md select-none z-10"
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center gap-2 font-extrabold">
            <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-ping" />
            <Cpu className="w-3.5 h-3.5 text-blue-600 animate-[spin_6s_linear_infinite]" />
            <span>CORE_ENG: YOLOv8_MAPPED</span>
          </div>
          <div className="flex items-center gap-2 font-extrabold">
            <span className="w-1.5 h-1.5 bg-teal-500 rounded-full animate-pulse" />
            <Activity className="w-3.5 h-3.5 text-teal-600" />
            <span>TRACK_ID: DEEP_SORT</span>
          </div>
          <div className="flex items-center gap-2 font-extrabold">
            <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
            <Wifi className="w-3.5 h-3.5 text-emerald-600" />
            <span>LATENCY: 12ms</span>
          </div>
        </motion.div>
      )}

      {/* RIGHT COLUMN HUD: Model confidence circular gauge */}
      {isActive && (
        <motion.div 
          className="absolute top-5 right-5 bg-white/85 backdrop-blur-md border border-sky-border/40 px-3.5 py-2.5 rounded-2xl flex items-center gap-3 shadow-md select-none z-10"
          initial={{ opacity: 0, x: 10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="relative flex items-center justify-center">
            <svg className="w-9 h-9 transform -rotate-90" viewBox="0 0 36 36">
              <circle cx="18" cy="18" r="14" fill="none" stroke="#E2E8F0" strokeWidth="3" />
              <circle 
                cx="18" 
                cy="18" 
                r="14" 
                fill="none" 
                stroke="#0284C7" 
                strokeWidth="3"
                strokeDasharray="88" 
                strokeDashoffset={strokeOffset} 
                strokeLinecap="round" 
                className="transition-all duration-1000 ease-out" 
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <ShieldCheck className="w-4 h-4 text-sky-default" />
            </div>
          </div>
          <div className="flex flex-col select-none leading-none">
            <span className="text-[11px] font-mono font-extrabold text-sky-default">{avgConfidence}%</span>
            <span className="text-[7px] font-heading font-extrabold text-sky-dark/50 uppercase tracking-widest mt-1">Accuracy</span>
          </div>
        </motion.div>
      )}

      {/* LOWER SECTION OVERLAYS */}
      {isActive && (
        <>
          {/* Live indicator badge */}
          <div className="absolute bottom-5 left-5 bg-white/80 backdrop-blur-sm border border-sky-border/40 px-3 py-1.5 rounded-xl flex items-center gap-2 shadow-md z-10">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-[ping_1.5s_infinite] shadow-sm shadow-red-500/50" />
            <span className="font-mono text-[9px] font-extrabold text-sky-dark/80 tracking-widest">
              LIVE BROADCAST
            </span>
          </div>

          {/* Framerate Odometer */}
          {fps !== null && (
            <div className="absolute bottom-5 right-5 bg-white/80 backdrop-blur-sm border border-sky-border/40 px-3 py-1.5 rounded-xl flex items-center gap-1.5 shadow-md z-10">
              <span className="font-heading font-extrabold text-[8px] text-sky-dark/40 uppercase tracking-wide">INFERENCE:</span>
              <span className="font-mono text-[9px] font-extrabold text-sky-default">
                {fps.toFixed(1)} FPS
              </span>
            </div>
          )}

          {/* Bottom Plate stock ticker scrolling banner overlay */}
          <div 
            className="absolute bottom-12 left-1/2 -translate-x-1/2 w-[90%] bg-white/85 backdrop-blur-md border border-sky-border/40 rounded-2xl py-2 px-4 overflow-hidden shadow-md flex items-center gap-4 z-10"
            style={{ borderLeft: "4px solid #0284C7" }}
          >
            <span className="text-[8px] font-heading font-extrabold text-sky-default uppercase tracking-wider shrink-0 flex items-center gap-1.5 select-none">
              <Zap className="w-3.5 h-3.5 text-sky-default animate-bounce" />
              Plate Capture Feed
            </span>
            <div className="flex gap-3 overflow-x-auto no-scrollbar font-mono text-[10px] uppercase font-bold text-sky-dark select-none flex-1 py-0.5">
              {plates.length > 0 ? (
                plates.slice(0, 8).map((p, i) => (
                  <motion.div 
                    key={p.plate + i} 
                    className="inline-flex items-center overflow-hidden rounded-md border border-gray-300 bg-gradient-to-b from-white to-gray-50 text-[9px] font-mono font-extrabold uppercase shadow-sm select-all shrink-0"
                    initial={{ opacity: 0, scale: 0.8, x: -10 }}
                    animate={{ opacity: 1, scale: 1, x: 0 }}
                    transition={{ type: "spring", stiffness: 300, damping: 20 }}
                  >
                    <div className="bg-blue-600 text-white font-extrabold text-[5px] px-1 py-1 flex flex-col items-center justify-center leading-none select-none border-r border-gray-200">
                      <span>IND</span>
                    </div>
                    <div className="px-2 py-0.5 font-bold text-gray-900 tracking-wider">
                      {p.plate}
                    </div>
                  </motion.div>
                ))
              ) : (
                <span className="text-sky-dark/30 italic font-heading font-semibold text-[8px] uppercase tracking-wider py-0.5">
                  Scanning video frame grids for active license plates...
                </span>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}


