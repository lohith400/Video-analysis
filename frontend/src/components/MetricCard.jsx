function TrendArrow({ direction }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="#38BDF8"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {direction === "up" ? (
        <>
          <line x1="12" y1="19" x2="12" y2="5" />
          <polyline points="5 12 12 5 19 12" />
        </>
      ) : (
        <>
          <line x1="12" y1="5" x2="12" y2="19" />
          <polyline points="19 12 12 19 5 12" />
        </>
      )}
    </svg>
  );
}

export default function MetricCard({ label, value, trend, suffix = "" }) {
  return (
    <div
      className="rounded-panel p-5 flex flex-col gap-1 transition-all duration-200"
      style={{ backgroundColor: "#c7e8fd", border: "1px solid #BAE6FD", borderRadius: "12px" }}
      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#BAE6FD")}
      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#c7e8fd")}
    >
      <div className="flex items-start justify-between">
        <span className="text-xs font-medium" style={{ color: "#475569" }}>
          {label}
        </span>
        {trend && <TrendArrow direction={trend} />}
      </div>
      <div className="text-3xl font-extrabold font-mono mt-1" style={{ color: "#0284C7" }}>
        {typeof value === "number" ? value.toLocaleString() : value}
        {suffix && <span className="text-base font-medium ml-1" style={{ color: "#38BDF8" }}>{suffix}</span>}
      </div>
    </div>
  );
}
