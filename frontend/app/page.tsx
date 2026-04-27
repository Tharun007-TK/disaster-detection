"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import DamageChart from "@/components/DamageChart";

interface Result {
  id: string;
  stem: string;
  damage_pct: Record<string, number>;
  mask_url?: string;
  timestamp?: string;
}

const CLASS_COLORS: Record<string, string> = {
  "No Damage": "text-green-400",
  Minor: "text-yellow-400",
  Major: "text-orange-400",
  Destroyed: "text-red-400",
};

function aggregate(results: Result[]): Record<string, number> {
  if (!results.length) return {};
  const sum: Record<string, number> = {};
  for (const r of results) {
    for (const [k, v] of Object.entries(r.damage_pct)) {
      sum[k] = (sum[k] ?? 0) + v;
    }
  }
  return Object.fromEntries(
    Object.entries(sum).map(([k, v]) => [k, v / results.length])
  );
}

export default function Dashboard() {
  const [results, setResults] = useState<Result[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/results")
      .then((r) => r.json())
      .then((data) => { setResults(Array.isArray(data) ? data : []); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const avg = aggregate(results);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Damage Assessment Dashboard</h1>
          <p className="text-zinc-400 text-sm mt-1">
            {results.length} area{results.length !== 1 ? "s" : ""} analyzed
          </p>
        </div>
        <Link
          href="/upload"
          className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded text-sm font-medium transition-colors"
        >
          Run Inference
        </Link>
      </div>

      {loading ? (
        <p className="text-zinc-400">Loading results...</p>
      ) : results.length === 0 ? (
        <div className="rounded-lg border border-zinc-800 p-8 text-center text-zinc-400">
          No results yet.{" "}
          <Link href="/upload" className="text-blue-400 hover:underline">
            Upload images to run inference.
          </Link>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
              <h2 className="text-sm font-semibold text-zinc-300 mb-3">
                Average Damage Distribution
              </h2>
              <DamageChart damagePct={avg} />
            </div>
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
              <h2 className="text-sm font-semibold text-zinc-300 mb-3">Average Breakdown</h2>
              <div className="space-y-2 mt-4">
                {Object.entries(avg).map(([cls, pct]) => (
                  <div key={cls} className="flex items-center gap-3">
                    <span className={`text-sm w-24 shrink-0 ${CLASS_COLORS[cls] ?? "text-zinc-300"}`}>
                      {cls}
                    </span>
                    <div className="flex-1 bg-zinc-800 rounded-full h-2 overflow-hidden">
                      <div
                        className="h-2 rounded-full"
                        style={{
                          width: `${Math.min(pct, 100)}%`,
                          backgroundColor:
                            cls === "No Damage" ? "#16a34a"
                            : cls === "Minor" ? "#ca8a04"
                            : cls === "Major" ? "#ea580c"
                            : "#dc2626",
                        }}
                      />
                    </div>
                    <span className="text-xs text-zinc-400 w-12 text-right">
                      {pct.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
            <div className="px-4 py-3 border-b border-zinc-800">
              <h2 className="text-sm font-semibold text-zinc-300">Recent Results</h2>
            </div>
            <ul className="divide-y divide-zinc-800">
              {results.slice(0, 20).map((r) => (
                <li key={r.id} className="px-4 py-3 flex items-center gap-4">
                  {r.mask_url && (
                    <img
                      src={`${apiUrl}${r.mask_url}`}
                      alt="mask"
                      className="w-12 h-12 object-cover rounded shrink-0"
                    />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{r.stem}</p>
                    <p className="text-xs text-zinc-500">
                      {r.timestamp?.slice(0, 19).replace("T", " ")}
                    </p>
                  </div>
                  <div className="flex gap-3 text-xs">
                    {Object.entries(r.damage_pct).map(([cls, pct]) => (
                      <span key={cls} className={CLASS_COLORS[cls] ?? "text-zinc-400"}>
                        {pct.toFixed(1)}%
                      </span>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
}
