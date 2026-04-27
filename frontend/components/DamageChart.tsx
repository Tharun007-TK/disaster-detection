"use client";

import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";

const CLASS_COLORS: Record<string, string> = {
  "No Damage": "#16a34a",
  Minor: "#ca8a04",
  Major: "#ea580c",
  Destroyed: "#dc2626",
};

interface Props {
  damagePct: Record<string, number>;
}

export default function DamageChart({ damagePct }: Props) {
  const data = Object.entries(damagePct).map(([name, value]) => ({ name, value }));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          outerRadius={80}
          label={({ name, value }) => `${name}: ${(value as number).toFixed(1)}%`}
          labelLine={false}
        >
          {data.map((entry) => (
            <Cell key={entry.name} fill={CLASS_COLORS[entry.name] ?? "#6b7280"} />
          ))}
        </Pie>
        <Tooltip formatter={(v) => typeof v === "number" ? `${v.toFixed(2)}%` : String(v)} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}
