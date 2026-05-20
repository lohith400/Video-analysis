import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import { Activity } from "lucide-react";

export default function Navbar() {
  const links = [
    { to: "/", label: "Upload Control" },
    { to: "/live", label: "Live Command Center" },
    { to: "/analytics", label: "Intelligence Analytics" },
  ];

  return (
    <nav className="sticky top-0 z-50 glass-card border-b border-sky-border/40 backdrop-blur-md bg-white/60">
      <div className="max-w-screen-xl mx-auto px-6 py-3.5 flex items-center justify-between">
        {/* Left Logo and Title */}
        <div className="flex items-center gap-3 select-none">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl overflow-hidden shadow-sm border border-sky-border/40">
            <img src="/logo.png" alt="IRIS Logo" className="w-full h-full object-cover" />
          </div>
          <div className="flex flex-col">
            <span className="font-heading font-extrabold text-base tracking-tight text-sky-dark leading-tight">
              IRIS
            </span>
            <span className="text-[10px] text-[#0284C7] tracking-widest uppercase font-mono font-bold leading-none">
              Indian Road Intelligence
            </span>
          </div>
        </div>

        {/* Right Nav Links with framer-motion sliding layout indicator */}
        <div className="flex items-center gap-1.5 p-1 bg-sky-surface/30 border border-sky-border/30 rounded-full">
          {links.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className="relative px-5 py-2 text-xs font-semibold tracking-wide rounded-full transition-all duration-300 select-none"
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.div
                      layoutId="active-nav-indicator"
                      className="absolute inset-0 bg-[#0284C7] rounded-full shadow-md shadow-[#0284C7]/20"
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                  <span
                    className={`relative z-10 font-heading transition-colors duration-300 ${
                      isActive ? "text-white font-bold" : "text-sky-dark hover:text-[#0284C7]"
                    }`}
                  >
                    {label}
                  </span>
                </>
              )}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  );
}
