import { useEffect } from "react";

function InfoIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0284C7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}

const KANNADA_MAP = {
  "No Helmet": "ಹೆಲ್ಮೆಟ್ ಇಲ್ಲ",
  "Wrong Way": "ತಪ್ಪು ದಿಕ್ಕು",
  "Speeding": "ವೇಗ ಮೀರಿದೆ",
  "Triple Riding": "ಮೂವರು ಸವಾರಿ",
};

export default function AlertBanner({ violation, onDismiss }) {
  useEffect(() => {
    if (!violation) return;
    const t = setTimeout(onDismiss, 8000);
    return () => clearTimeout(t);
  }, [violation, onDismiss]);

  if (!violation) return null;

  const kannada = KANNADA_MAP[violation.type] || violation.type;

  return (
    <div
      className="slide-up flex items-center gap-4 px-5 py-3.5 rounded-card"
      style={{
        backgroundColor: "#c7e8fd",
        border: "1px solid #0284C7",
        borderRadius: "8px",
      }}
    >
      <InfoIcon />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold" style={{ color: "#0C4A6E" }}>
          Violation Detected: {violation.type}
          {violation.plate && (
            <span className="ml-2 font-mono text-xs" style={{ color: "#0284C7" }}>
              [{violation.plate}]
            </span>
          )}
        </p>
        <p className="text-xs mt-0.5" style={{ color: "#475569" }}>
          {kannada} · ಉಲ್ಲಂಘನೆ ಪತ್ತೆಯಾಗಿದೆ
        </p>
      </div>
      <button
        onClick={onDismiss}
        className="text-xs font-semibold px-3 py-1.5 rounded-card transition-all duration-200 shrink-0"
        style={{ color: "#0284C7", border: "1px solid #0284C7", backgroundColor: "transparent", borderRadius: "6px" }}
        onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = "#0284C7"; e.currentTarget.style.color = "#F0F9FF"; }}
        onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = "transparent"; e.currentTarget.style.color = "#0284C7"; }}
      >
        Dismiss
      </button>
    </div>
  );
}
