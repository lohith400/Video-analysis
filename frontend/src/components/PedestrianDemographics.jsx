import { motion } from "framer-motion";
import { Users, User, Smile } from "lucide-react";

export default function PedestrianDemographics({ data = {} }) {
  const total = data.total || 0;
  const males = data.males || 0;
  const females = data.females || 0;
  const children = data.children || 0;
  const unknown = data.unknown || (total - males - females - children > 0 ? total - males - females - children : 0);

  const malePct = total > 0 ? Math.round((males / total) * 100) : 0;
  const femalePct = total > 0 ? Math.round((females / total) * 100) : 0;
  const childPct = total > 0 ? Math.round((children / total) * 100) : 0;
  const unknownPct = total > 0 ? Math.max(0, 100 - malePct - femalePct - childPct) : 0;

  return (
    <div className="select-none">
      {/* Title Header */}
      <div className="flex items-center justify-between mb-3 border-b border-sky-border/30 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-sky-default/10 flex items-center justify-center text-sky-default">
            <Users className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider">
              Pedestrian Demographics
            </h3>
            <p className="text-[9px] text-sky-dark/45 font-mono uppercase">
              YOLOv8 + Deep Demographic Classification
            </p>
          </div>
        </div>
        <span className="font-mono text-[10px] font-extrabold text-sky-default bg-sky-surface border border-sky-border/40 px-2.5 py-1 rounded-xl shadow-sm">
          {total} HUMANS LOGGED
        </span>
      </div>

      {/* Demographic Proportion Segmented Bar */}
      {total > 0 && (
        <div className="mb-3.5">
          <div className="flex items-center justify-between text-[9px] font-heading font-bold text-sky-dark/60 mb-1">
            <span>Demographic Distribution</span>
            <span>100% of tracked pedestrians</span>
          </div>
          <div className="h-2 rounded-full overflow-hidden flex bg-sky-surface/30 p-0.5 border border-sky-border/30">
            {malePct > 0 && (
              <div 
                className="h-full bg-blue-500 rounded-l-full transition-all duration-500" 
                style={{ width: `${malePct}%` }}
                title={`Males: ${malePct}%`}
              />
            )}
            {femalePct > 0 && (
              <div 
                className="h-full bg-purple-500 transition-all duration-500" 
                style={{ width: `${femalePct}%` }}
                title={`Females: ${femalePct}%`}
              />
            )}
            {childPct > 0 && (
              <div 
                className="h-full bg-amber-400 transition-all duration-500" 
                style={{ width: `${childPct}%` }}
                title={`Children: ${childPct}%`}
              />
            )}
            {unknownPct > 0 && (
              <div 
                className="h-full bg-sky-300 rounded-r-full transition-all duration-500" 
                style={{ width: `${unknownPct}%` }}
                title={`General: ${unknownPct}%`}
              />
            )}
          </div>
        </div>
      )}

      {/* Four Demographic Stat Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        {/* Males */}
        <div className="p-3 rounded-xl bg-blue-50/30 border border-blue-200/40 shadow-sm flex flex-col items-center text-center">
          <span className="text-xl mb-1">👨</span>
          <span className="text-[9px] font-heading font-extrabold text-blue-700 uppercase tracking-wider">
            Adult Males
          </span>
          <span className="font-mono text-lg font-extrabold text-blue-600 leading-tight mt-0.5">
            {males}
          </span>
          <span className="text-[9px] font-mono text-blue-500/70 font-semibold mt-0.5">
            {malePct}% of total
          </span>
        </div>

        {/* Females */}
        <div className="p-3 rounded-xl bg-purple-50/30 border border-purple-200/40 shadow-sm flex flex-col items-center text-center">
          <span className="text-xl mb-1">👩</span>
          <span className="text-[9px] font-heading font-extrabold text-purple-700 uppercase tracking-wider">
            Adult Females
          </span>
          <span className="font-mono text-lg font-extrabold text-purple-600 leading-tight mt-0.5">
            {females}
          </span>
          <span className="text-[9px] font-mono text-purple-500/70 font-semibold mt-0.5">
            {femalePct}% of total
          </span>
        </div>

        {/* Children */}
        <div className={`p-3 rounded-xl border shadow-sm flex flex-col items-center text-center transition-all ${
          children > 0 
            ? "bg-amber-50/60 border-amber-300" 
            : "bg-yellow-50/20 border-yellow-200/30"
        }`}>
          <span className="text-xl mb-1">🧒</span>
          <span className="text-[9px] font-heading font-extrabold text-amber-700 uppercase tracking-wider">
            Children
          </span>
          <span className="font-mono text-lg font-extrabold text-amber-600 leading-tight mt-0.5">
            {children}
          </span>
          <span className="text-[9px] font-mono text-amber-500/70 font-semibold mt-0.5">
            {childPct}% of total
          </span>
        </div>

        {/* General / Unclassified */}
        <div className="p-3 rounded-xl bg-white/40 border border-sky-border/30 shadow-sm flex flex-col items-center text-center">
          <span className="text-xl mb-1">🚶</span>
          <span className="text-[9px] font-heading font-extrabold text-sky-dark/70 uppercase tracking-wider">
            General
          </span>
          <span className="font-mono text-lg font-extrabold text-sky-dark leading-tight mt-0.5">
            {unknown}
          </span>
          <span className="text-[9px] font-mono text-sky-dark/50 font-semibold mt-0.5">
            {unknownPct}% of total
          </span>
        </div>
      </div>
    </div>
  );
}
