const VEHICLE_TYPES = [
  { key: "car", label: "Car" },
  { key: "truck", label: "Truck" },
  { key: "bus", label: "Bus" },
  { key: "auto-rickshaw", label: "Auto-Rickshaw" },
  { key: "motorcycle", label: "Motorcycle" },
  { key: "scooter", label: "Scooter" },
  { key: "bicycle", label: "Bicycle" },
];

export default function VehicleCounts({ counts = {} }) {
  const total = counts.total || 0;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full live-pulse" style={{ backgroundColor: "#38BDF8" }} />
          <h3 className="text-sm font-semibold" style={{ color: "#0C4A6E" }}>
            Vehicle Counts
          </h3>
        </div>
        <span className="text-xs font-semibold font-mono" style={{ color: "#0284C7" }}>
          Total: {total.toLocaleString()} vehicles
        </span>
      </div>

      {/* Count Cards */}
      <div className="grid grid-cols-4 gap-2">
        {VEHICLE_TYPES.map(({ key, label }) => (
          <div
            key={key}
            className="rounded-card px-2 py-2.5 text-center transition-all duration-200"
            style={{ backgroundColor: "#c7e8fd", border: "1px solid #BAE6FD", borderRadius: "8px" }}
          >
            <div className="text-xl font-bold font-mono" style={{ color: "#0284C7" }}>
              {(counts[key] || 0).toLocaleString()}
            </div>
            <div className="text-xs mt-0.5 leading-tight" style={{ color: "#475569" }}>
              {label}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
