"use client";

import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import { DAMAGE_CLASSES, colorFor } from "@/lib/severity";

interface Props {
  damagePct: Record<string, number>;
  /** Render as a donut (default) or solid pie. */
  donut?: boolean;
  height?: number;
}

export default function DamageChart({ damagePct, donut = true, height = 240 }: Props) {
  // Order classes consistently so colors stay stable across re-renders.
  const data = DAMAGE_CLASSES
    .filter((cls) => damagePct[cls] !== undefined)
    .map((cls) => ({ name: cls, value: damagePct[cls] ?? 0 }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          innerRadius={donut ? 56 : 0}
          outerRadius={88}
          paddingAngle={donut ? 2 : 0}
          stroke="#0a0a0a"
          strokeWidth={2}
          isAnimationActive={false}
        >
          {data.map((entry) => (
            <Cell key={entry.name} fill={colorFor(entry.name)} />
          ))}
        </Pie>
        <Tooltip
          cursor={{ fill: "transparent" }}
          contentStyle={{
            backgroundColor: "#18181b",
            border: "1px solid #27272a",
            borderRadius: 8,
            color: "#e4e4e7",
            fontSize: 12,
            padding: "6px 10px",
          }}
          itemStyle={{ color: "#e4e4e7" }}
          labelStyle={{ color: "#a1a1aa" }}
          formatter={(v) =>
            typeof v === "number" ? `${v.toFixed(2)}%` : String(v)
          }
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
