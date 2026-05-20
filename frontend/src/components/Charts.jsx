import {
  PieChart,
  Pie,
  Cell,
  Tooltip as PieTooltip,
  Legend,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as LineTooltip,
  ResponsiveContainer,
} from "recharts";

const VEHICLE_COLORS = {
  Car: "#0284C7",
  Truck: "#38BDF8",
  Bus: "#7DD3FC",
  Motorcycle: "#0C4A6E",
  "Auto-Rickshaw": "#BAE6FD",
  Scooter: "#c7e8fd",
  Bicycle: "#475569",
};

const CustomTooltipPie = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{ backgroundColor: "#c7e8fd", border: "1px solid #BAE6FD", borderRadius: "8px", padding: "8px 12px" }}>
        <p style={{ color: "#0C4A6E", fontSize: 12, fontWeight: 600 }}>{payload[0].name}</p>
        <p style={{ color: "#0284C7", fontSize: 13, fontWeight: 700 }}>{payload[0].value.toLocaleString()}</p>
      </div>
    );
  }
  return null;
};

const CustomTooltipLine = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{ backgroundColor: "#c7e8fd", border: "1px solid #BAE6FD", borderRadius: "8px", padding: "8px 12px" }}>
        <p style={{ color: "#475569", fontSize: 11 }}>{label}</p>
        <p style={{ color: "#0284C7", fontSize: 13, fontWeight: 700 }}>{payload[0].value.toLocaleString()} vehicles</p>
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
    <div
      className="rounded-panel p-5"
      style={{ backgroundColor: "#c7e8fd", border: "1px solid #BAE6FD", borderRadius: "12px" }}
    >
      <h3 className="text-sm font-semibold mb-4" style={{ color: "#0C4A6E" }}>
        Vehicle Distribution
      </h3>
      <ResponsiveContainer width="100%" height={260}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            innerRadius={65}
            outerRadius={100}
            paddingAngle={2}
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={VEHICLE_COLORS[entry.name] || "#0284C7"}
                stroke="#c7e8fd"
                strokeWidth={2}
              />
            ))}
          </Pie>
          <PieTooltip content={<CustomTooltipPie />} />
          <Legend
            iconType="circle"
            iconSize={8}
            formatter={(value) => (
              <span style={{ color: "#0C4A6E", fontSize: 11 }}>{value}</span>
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
    <div
      className="rounded-panel p-5"
      style={{ backgroundColor: "#c7e8fd", border: "1px solid #BAE6FD", borderRadius: "12px" }}
    >
      <h3 className="text-sm font-semibold mb-4" style={{ color: "#0C4A6E" }}>
        Vehicles Over Time
      </h3>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData} margin={{ top: 4, right: 16, left: -10, bottom: 0 }}>
          <CartesianGrid stroke="#BAE6FD" strokeDasharray="4 4" vertical={false} />
          <XAxis
            dataKey="time"
            tick={{ fill: "#475569", fontSize: 11 }}
            axisLine={{ stroke: "#BAE6FD" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "#475569", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <LineTooltip content={<CustomTooltipLine />} />
          <Line
            type="monotone"
            dataKey="vehicles"
            stroke="#0284C7"
            strokeWidth={2.5}
            dot={{ fill: "#0284C7", r: 3, strokeWidth: 0 }}
            activeDot={{ fill: "#0C4A6E", r: 5, strokeWidth: 0 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
