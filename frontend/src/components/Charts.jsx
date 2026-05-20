import {
  PieChart,
  Pie,
  Cell,
  Tooltip as PieTooltip,
  Legend,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as AreaTooltip,
  ResponsiveContainer,
} from "recharts";

const VEHICLE_COLORS = {
  Car: "#0284C7",
  Truck: "#38BDF8",
  Bus: "#7DD3FC",
  Motorcycle: "#0C4A6E",
  "Auto-Rickshaw": "#BAE6FD",
  Scooter: "#c7e8fd",
  Bicycle: "#94A3B8",
};

const CustomTooltipPie = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card rounded-xl p-2.5 border border-sky-border/40 font-heading text-xs text-sky-dark">
        <p className="font-mono text-[9px] text-sky-dark/50 uppercase">CLASSIFICATION</p>
        <p className="font-extrabold text-sky-dark mt-0.5">{payload[0].name}</p>
        <p className="font-mono font-extrabold text-sky-default text-sm mt-1">
          {payload[0].value.toLocaleString()} units
        </p>
      </div>
    );
  }
  return null;
};

const CustomTooltipArea = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card rounded-xl p-2.5 border border-sky-border/40 font-heading text-xs text-sky-dark">
        <p className="font-mono text-[9px] text-sky-dark/50 uppercase">TIME INTERVAL</p>
        <p className="font-extrabold text-sky-dark mt-0.5">{label}</p>
        <p className="font-mono font-extrabold text-sky-default text-sm mt-1">
          {payload[0].value.toLocaleString()} vehicles
        </p>
      </div>
    );
  }
  return null;
};

export function VehicleDistributionChart({ data = [] }) {
  const chartData = data.length > 0 ? data : [
    { name: "Car", value: 412 },
    { name: "Truck", value: 87 },
    { name: "Bus", value: 53 },
    { name: "Motorcycle", value: 298 },
    { name: "Auto-Rickshaw", value: 201 },
    { name: "Scooter", value: 164 },
    { name: "Bicycle", value: 32 },
  ];

  return (
    <div className="glass-card rounded-2xl p-5 shadow-sm border border-sky-border/30">
      <h3 className="font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider mb-4">
        Vehicle Type Breakdown
      </h3>
      <ResponsiveContainer width="100%" height={260}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="45%"
            innerRadius={60}
            outerRadius={95}
            paddingAngle={2.5}
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={VEHICLE_COLORS[entry.name] || "#0284C7"}
                stroke="#F0F9FF"
                strokeWidth={2}
              />
            ))}
          </Pie>
          <PieTooltip content={<CustomTooltipPie />} />
          <Legend
            iconType="circle"
            iconSize={6}
            verticalAlign="bottom"
            formatter={(value) => (
              <span className="font-heading font-bold text-[10px] text-sky-dark/75 uppercase tracking-wide">
                {value}
              </span>
            )}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

export function VehiclesOverTimeChart({ data = [] }) {
  const chartData = data.length > 0 ? data : [
    { time: "08:00", vehicles: 42 },
    { time: "09:00", vehicles: 78 },
    { time: "10:00", vehicles: 95 },
    { time: "11:00", vehicles: 130 },
    { time: "12:00", vehicles: 110 },
    { time: "13:00", vehicles: 88 },
    { time: "14:00", vehicles: 105 },
    { time: "15:00", vehicles: 143 },
    { time: "16:00", vehicles: 187 },
    { time: "17:00", vehicles: 210 },
    { time: "18:00", vehicles: 176 },
    { time: "19:00", vehicles: 122 },
  ];

  return (
    <div className="glass-card rounded-2xl p-5 shadow-sm border border-sky-border/30">
      <h3 className="font-heading font-extrabold text-xs text-sky-dark uppercase tracking-wider mb-4">
        Detections Over Timeline
      </h3>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={chartData} margin={{ top: 10, right: 16, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="areaChartGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#0284C7" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#0284C7" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(186, 230, 253, 0.35)" strokeDasharray="4 4" vertical={false} />
          <XAxis
            dataKey="time"
            tick={{ fill: "#475569", fontSize: 10, fontFamily: "Inter" }}
            axisLine={{ stroke: "rgba(186, 230, 253, 0.4)" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "#475569", fontSize: 10, fontFamily: "Inter" }}
            axisLine={false}
            tickLine={false}
          />
          <AreaTooltip content={<CustomTooltipArea />} />
          <Area
            type="monotone"
            dataKey="vehicles"
            stroke="#0284C7"
            strokeWidth={2.5}
            fillOpacity={1}
            fill="url(#areaChartGlow)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
