function CheckIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}

export default function ViolationList({ violations = [] }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold" style={{ color: "#0C4A6E" }}>
          Violations
        </h3>
        <span
          className="text-xs font-semibold px-2 py-0.5 rounded-full"
          style={{ backgroundColor: "#7DD3FC", color: "#0C4A6E" }}
        >
          {violations.length}
        </span>
      </div>

      <div className="flex flex-col gap-2">
        {violations.length === 0 ? (
          <div
            className="flex items-center justify-center gap-2 py-5 rounded-card text-sm"
            style={{ backgroundColor: "#c7e8fd", borderRadius: "8px" }}
          >
            <CheckIcon />
            <span style={{ color: "#0284C7" }}>No violations detected</span>
          </div>
        ) : (
          violations.map((v, i) => (
            <div
              key={i}
              className="flex items-center justify-between px-3 py-2.5 rounded-card"
              style={{
                backgroundColor: "#c7e8fd",
                borderLeft: "3px solid #0284C7",
                borderRadius: "8px",
              }}
            >
              <div className="flex flex-col gap-0.5">
                <span className="text-xs font-semibold" style={{ color: "#0284C7" }}>
                  {v.type}
                </span>
                {v.plate && (
                  <span className="text-xs font-mono uppercase" style={{ color: "#475569" }}>
                    {v.plate}
                  </span>
                )}
              </div>
              {v.timestamp && (
                <span className="text-xs font-mono" style={{ color: "#475569" }}>
                  {new Date(v.timestamp).toTimeString().slice(0, 8)}
                </span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
