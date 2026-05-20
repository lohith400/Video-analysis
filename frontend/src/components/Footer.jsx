import { Activity, Shield, Eye, MapPin } from "lucide-react";

export default function Footer() {
  return (
    <footer className="w-full border-t border-[#BAE6FD]/40 bg-white/50 backdrop-blur-md select-none mt-auto">
      <div className="max-w-screen-xl mx-auto px-6 py-8">
        {/* Top row */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 mb-6">
          {/* Brand block */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl overflow-hidden shadow-sm border border-[#BAE6FD]/40">
              <img src="/logo.png" alt="IRIS Logo" className="w-full h-full object-cover" />
            </div>
            <div>
              <h4 className="font-heading font-extrabold text-sm text-[#0C4A6E] tracking-tight leading-tight">
                IRIS
              </h4>
              <p className="text-[9px] font-mono font-bold text-[#0284C7] tracking-widest uppercase leading-none">
                Indian Road Intelligence System
              </p>
            </div>
          </div>

          {/* Feature highlights */}
          <div className="flex flex-wrap items-center gap-6">
            {[
              { icon: Eye, label: "Real-Time Detection" },
              { icon: Shield, label: "ANPR Scanning" },
              { icon: Activity, label: "Traffic Analytics" },
              { icon: MapPin, label: "GPS Mapping" },
            ].map(({ icon: Icon, label }) => (
              <div key={label} className="flex items-center gap-1.5 text-[10px] font-heading font-bold text-[#0C4A6E]/60 uppercase tracking-wider">
                <Icon className="w-3.5 h-3.5 text-[#0284C7]" strokeWidth={1.8} />
                {label}
              </div>
            ))}
          </div>
        </div>

        {/* Divider */}
        <div className="w-full h-px bg-[#BAE6FD]/40 mb-4" />

        {/* Bottom row */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-2">
          <p className="text-[10px] font-heading font-semibold text-[#0C4A6E]/45 tracking-wide">
            © {new Date().getFullYear()} IRIS — Indian Road Intelligence System. Built with YOLOv8 + ByteTrack.
          </p>
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[9px] font-mono font-bold text-[#0284C7] tracking-widest uppercase">
              System Operational
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
