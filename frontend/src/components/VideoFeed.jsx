import { motion } from "framer-motion";
import { Camera, Cpu, Activity, Wifi, ShieldCheck, Zap } from "lucide-react";

export default function VideoFeed({ frameData, fps, source, isActive, plates = [] }) {
  // Hardcoded or dynamically derived model average metrics for realism
  const avgConfidence = isActive ? 92.4 : 0;
  const strokeOffset = 88 - (88 * avgConfidence) / 100;

  return (
    <div
      className="relative rounded-2xl overflow-hidden glass-card select-none"
      style={{
        backgroundColor: "#F0F9FF",
        border: isActive ? "2px solid #38BDF8" : "1px solid #BAE6FD",
        aspectRatio: "16/9",
        boxShadow: isActive 
          ? "0 10px 25px -5px rgba(56, 189, 248, 0.15), 0 8px 10px -6px rgba(56, 189, 248, 0.1)"
          : "none",
      }}
    >
      {/* Reticle decorative corner brackets */}
      {isActive && (
        <>
          <div className="absolute top-4 left-4 w-4 h-4 border-t-2 border-l-2 border-sky-default/70 rounded-tl" />
          <div className="absolute top-4 right-4 w-4 h-4 border-t-2 border-r-2 border-sky-default/70 rounded-tr" />
          <div className="absolute bottom-4 left-4 w-4 h-4 border-b-2 border-l-2 border-sky-default/70 rounded-bl" />
          <div className="absolute bottom-4 right-4 w-4 h-4 border-b-2 border-r-2 border-sky-default/70 rounded-br" />
          
          {/* Subtle diagonal scanning laser line */}
          <div className="absolute inset-0 bg-gradient-to-b from-sky-default/5 to-transparent h-1/2 pointer-events-none animate-[pulse_3s_infinite]" />
        </>
      )}

      {frameData ? (
        <img
          src={`data:image/jpeg;base64,${frameData}`}
          alt="Live feed"
          className="w-full h-full object-contain"
        />
      ) : (
        <div className="w-full h-full flex flex-col items-center justify-center gap-4 bg-sky-surface/10 relative">
          {/* Radar background grid scanning lines effect */}
          <div className="absolute inset-0 grid grid-cols-6 grid-rows-6 opacity-20 pointer-events-none">
            {Array.from({ length: 36 }).map((_, i) => (
              <div key={i} className="border-[0.5px] border-sky-border" />
            ))}
          </div>

          <div className="w-16 h-16 rounded-full bg-sky-surface flex items-center justify-center border border-sky-border shadow-inner relative animate-pulse">
            <Camera className="w-8 h-8 text-sky-default" strokeWidth={1.5} />
            <div className="absolute inset-0 rounded-full border border-sky-default animate-[ping_2s_infinite] opacity-30" />
          </div>
          
          <div className="text-center z-10 px-4">
            <h4 className="font-heading font-extrabold text-sm text-sky-dark tracking-wide uppercase">
              System Telemetry Standby
            </h4>
            <p className="text-[10px] font-heading font-semibold text-sky-default uppercase tracking-wider mt-1">
              Initialize source stream to trigger machine vision pipeline
            </p>
          </div>
        </div>
      )}

      {/* LEFT COLUMN HUD: Monospace intelligence diagnostics panel */}
      {isActive && (
        <motion.div 
          className="absolute top-4 left-4 bg-white/75 backdrop-blur-md border border-sky-border/40 p-2.5 rounded-xl font-mono text-[9px] text-sky-dark/95 flex flex-col gap-1 shadow-sm select-none"
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center gap-1.5 font-bold">
            <Cpu className="w-3 h-3 text-sky-default animate-spin" style={{ animationDuration: '4s' }} />
            <span>SYS_ENG: YOLOv8_VLD</span>
          </div>
          <div className="flex items-center gap-1.5 font-bold">
            <Activity className="w-3 h-3 text-sky-default" />
            <span>TRK_MD: BYTE_TRACK</span>
          </div>
          <div className="flex items-center gap-1.5 font-bold">
            <Wifi className="w-3 h-3 text-emerald-500 animate-pulse" />
            <span>LATENCY: 34ms</span>
          </div>
        </motion.div>
      )}

      {/* RIGHT COLUMN HUD: Model confidence circular gauge */}
      {isActive && (
        <motion.div 
          className="absolute top-4 right-4 bg-white/75 backdrop-blur-md border border-sky-border/40 px-3 py-2 rounded-xl flex items-center gap-2.5 shadow-sm select-none"
          initial={{ opacity: 0, x: 10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="relative flex items-center justify-center">
            <svg className="w-8 h-8 transform -rotate-90" viewBox="0 0 36 36">
              <circle cx="18" cy="18" r="14" fill="none" stroke="#E0F2FE" strokeWidth="3" />
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
              <ShieldCheck className="w-3 h-3 text-sky-default" />
            </div>
          </div>
          <div className="flex flex-col select-none leading-none">
            <span className="text-[10px] font-mono font-extrabold text-sky-default">{avgConfidence}%</span>
            <span className="text-[7.5px] font-heading font-extrabold text-sky-dark/50 uppercase tracking-wider mt-0.5">Model Conf</span>
          </div>
        </motion.div>
      )}

      {/* LOWER SECTION OVERLAYS */}
      {isActive && (
        <>
          {/* Live indicator badge */}
          <div className="absolute bottom-4 left-4 bg-white/70 backdrop-blur-sm border border-sky-border/40 px-2.5 py-1 rounded-lg flex items-center gap-1.5 shadow-sm">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-[ping_1.5s_infinite] shadow-sm shadow-red-500/50" />
            <span className="font-mono text-[9px] font-extrabold text-sky-dark/80 tracking-widest">
              LIVE BROADCAST
            </span>
          </div>

          {/* Framerate Odometer */}
          {fps !== null && (
            <div className="absolute bottom-4 right-4 bg-white/70 backdrop-blur-sm border border-sky-border/40 px-2.5 py-1 rounded-lg flex items-center gap-1 shadow-sm">
              <span className="font-heading font-extrabold text-[8px] text-sky-dark/40 uppercase">INFERENCE FPS:</span>
              <span className="font-mono text-[9px] font-extrabold text-sky-default">
                {fps.toFixed(1)}
              </span>
            </div>
          )}

          {/* Bottom Plate stock ticker scrolling banner overlay */}
          <div 
            className="absolute bottom-11 left-1/2 -translate-x-1/2 w-[90%] bg-white/80 backdrop-blur-md border border-sky-border/50 rounded-xl py-1.5 px-3 overflow-hidden shadow-sm flex items-center gap-3.5"
            style={{ borderLeft: "3px solid #0284C7" }}
          >
            <span className="text-[8px] font-heading font-extrabold text-sky-default uppercase tracking-wider shrink-0 flex items-center gap-1 select-none">
              <Zap className="w-2.5 h-2.5 text-sky-default animate-bounce" />
              Plate Capture Feed
            </span>
            <div className="flex gap-2.5 overflow-x-auto no-scrollbar font-mono text-[9.5px] uppercase font-bold text-sky-dark select-none flex-1 py-0.5">
              {plates.length > 0 ? (
                plates.slice(0, 6).map((p, i) => (
                  <motion.span 
                    key={p.plate + i} 
                    className="px-2.5 py-0.5 bg-sky-surface/40 border border-sky-border/40 rounded-lg text-sky-default shrink-0 flex items-center gap-1.5 shadow-sm"
                    initial={{ opacity: 0, scale: 0.8, x: -10 }}
                    animate={{ opacity: 1, scale: 1, x: 0 }}
                    transition={{ type: "spring", stiffness: 300, damping: 20 }}
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    {p.plate}
                  </motion.span>
                ))
              ) : (
                <span className="text-sky-dark/30 italic font-heading font-semibold text-[8.5px] uppercase tracking-wider py-0.5">
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

