function formatTime(ts) {
  if (!ts) return "--:--:--";
  const d = new Date(ts);
  return d.toTimeString().slice(0, 8);
}

export default function PlateTable({ plates = [] }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold" style={{ color: "#0C4A6E" }}>
          Plates Detected
        </h3>
        <span className="text-xs font-mono" style={{ color: "#0284C7" }}>
          {plates.length} plate{plates.length !== 1 ? "s" : ""}
        </span>
      </div>

      <div
        className="overflow-hidden rounded-card"
        style={{ border: "1px solid #BAE6FD", borderRadius: "8px" }}
      >
        {/* Table Header */}
        <div
          className="grid text-xs font-semibold px-3 py-2"
          style={{ backgroundColor: "#c7e8fd", color: "#0C4A6E", gridTemplateColumns: "1fr 1fr 1fr" }}
        >
          <span>#</span>
          <span>Plate No.</span>
          <span>Time</span>
        </div>

        {/* Scrollable rows */}
        <div className="overflow-y-auto" style={{ maxHeight: "140px" }}>
          {plates.length === 0 ? (
            <div
              className="px-3 py-6 text-center text-xs"
              style={{ backgroundColor: "#F0F9FF", color: "#475569" }}
            >
              No plates detected yet
            </div>
          ) : (
            plates.map((p, i) => (
              <div
                key={i}
                className="grid text-xs px-3 py-2 items-center"
                style={{
                  backgroundColor: i % 2 === 0 ? "#F0F9FF" : "#c7e8fd",
                  gridTemplateColumns: "1fr 1fr 1fr",
                  borderTop: "1px solid #BAE6FD",
                }}
              >
                <span style={{ color: "#475569" }}>{i + 1}</span>
                <span
                  className="font-mono font-semibold uppercase tracking-widest"
                  style={{ color: "#0284C7" }}
                >
                  {p.plate}
                </span>
                <span className="font-mono" style={{ color: "#475569" }}>
                  {formatTime(p.timestamp)}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
