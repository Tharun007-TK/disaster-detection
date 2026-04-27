"use client";

import { useEffect, useState } from "react";

interface PrecautionInfo {
  damage_class: number;
  name: string;
  color: string;
  precautions: string[];
  actions: string[];
  evacuation: string;
}

const BORDER: Record<number, string> = {
  0: "border-green-700",
  1: "border-yellow-600",
  2: "border-orange-600",
  3: "border-red-700",
};

export default function PrecautionsPage() {
  const [data, setData] = useState<PrecautionInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/precautions")
      .then((r) => r.json())
      .then((d) => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Precaution Protocols</h1>
        <p className="text-zinc-400 text-sm mt-1">
          Recommended actions by damage classification
        </p>
      </div>

      {loading ? (
        <p className="text-zinc-400">Loading...</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {data.map((item) => (
            <div
              key={item.damage_class}
              className={`rounded-lg border bg-zinc-900 p-5 space-y-4 ${BORDER[item.damage_class] ?? "border-zinc-700"}`}
            >
              <div className="flex items-center gap-3">
                <span
                  className="w-4 h-4 rounded-full shrink-0"
                  style={{ backgroundColor: item.color }}
                />
                <h2 className="text-lg font-semibold">{item.name}</h2>
                <span className="text-xs text-zinc-500 ml-auto">Class {item.damage_class}</span>
              </div>

              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2">
                  Precautions
                </h3>
                <ul className="space-y-1">
                  {item.precautions.map((p, i) => (
                    <li key={i} className="text-sm text-zinc-300 flex gap-2">
                      <span className="text-zinc-500 shrink-0">·</span>{p}
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2">
                  Required Actions
                </h3>
                <ul className="space-y-1">
                  {item.actions.map((a, i) => (
                    <li key={i} className="text-sm text-zinc-300 flex gap-2">
                      <span className="text-zinc-500 shrink-0">{i + 1}.</span>{a}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="rounded bg-zinc-800 px-3 py-2">
                <span className="text-xs font-semibold text-zinc-400">Evacuation: </span>
                <span className="text-xs text-zinc-200">{item.evacuation}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
