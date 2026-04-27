"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import Link from "next/link";
import DamageChart from "@/components/DamageChart";
import StatCard from "@/components/StatCard";
import SeverityBadge from "@/components/SeverityBadge";
import DamageBar from "@/components/DamageBar";
import { DAMAGE_CLASSES, colorFor, relativeTime } from "@/lib/severity";

interface Result {
  id: string;
  stem: string;
  damage_pct: Record<string, number>;
  mask_url?: string;
  timestamp?: string;
}

function aggregate(results: Result[]): Record<string, number> {
  if (!results.length) return Object.fromEntries(DAMAGE_CLASSES.map((c) => [c, 0]));
  const sum: Record<string, number> = {};
  for (const cls of DAMAGE_CLASSES) sum[cls] = 0;
  for (const r of results) {
    for (const cls of DAMAGE_CLASSES) {
      sum[cls] += r.damage_pct[cls] ?? 0;
    }
  }
  const out: Record<string, number> = {};
  for (const cls of DAMAGE_CLASSES) out[cls] = sum[cls] / results.length;
  return out;
}

function formatTimestamp(iso?: string): string {
  if (!iso) return "—";
  return iso.slice(0, 19).replace("T", " ");
}

export default function Dashboard() {
  const [results, setResults] = useState<Result[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(() => {
    fetch("/api/results", { cache: "no-store" })
      .then((r) => r.json())
      .then((data) => {
        setResults(Array.isArray(data) ? data : []);
        setLoading(false);
        setRefreshing(false);
      })
      .catch(() => {
        setResults([]);
        setLoading(false);
        setRefreshing(false);
      });
  }, []);

  useEffect(() => {
    fetch("/api/results", { cache: "no-store" })
      .then((r) => r.json())
      .then((data) => {
        setResults(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    load();
  }, [load]);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  const sorted = useMemo(() => {
    return [...results].sort((a, b) => {
      const ta = a.timestamp ? Date.parse(a.timestamp) : 0;
      const tb = b.timestamp ? Date.parse(b.timestamp) : 0;
      return tb - ta;
    });
  }, [results]);

  const avg = useMemo(() => aggregate(results), [results]);
  const avgDestroyed = avg["Destroyed"] ?? 0;
  const criticalCount = useMemo(
    () => results.filter((r) => (r.damage_pct["Destroyed"] ?? 0) > 20).length,
    [results]
  );
  const latestTs = sorted[0]?.timestamp;

  return (
    <div className="p-6 lg:p-8 space-y-6">
      {/* Header */}
      <header className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight text-zinc-100">
            Damage Assessment
          </h1>
          <p className="text-sm text-zinc-400">
            <span className="tabular-nums">{results.length}</span> site
            {results.length === 1 ? "" : "s"} analyzed
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onRefresh}
            disabled={refreshing || loading}
            className="rounded-md border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 disabled:opacity-50 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {refreshing ? "Refreshing..." : "Refresh"}
          </button>
          <Link
            href="/upload"
            className="rounded-md bg-blue-600 hover:bg-blue-500 px-3 py-1.5 text-sm font-medium text-white transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Run Inference
          </Link>
        </div>
      </header>

      {/* KPI Row */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Sites Analyzed"
          value={results.length}
          hint={loading ? "Loading..." : "Total analyzed"}
        />
        <StatCard
          label="Avg Destroyed"
          value={`${avgDestroyed.toFixed(1)}%`}
          color={colorFor("Destroyed")}
          hint="Across all sites"
        />
        <StatCard
          label="Critical Sites"
          value={criticalCount}
          hint="Destroyed > 20%"
        />
        <StatCard
          label="Last Inference"
          value={relativeTime(latestTs)}
          hint={formatTimestamp(latestTs)}
        />
      </section>

      {loading ? (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-8 text-center text-sm text-zinc-400">
          Loading results...
        </div>
      ) : results.length === 0 ? (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-12 text-center">
          <p className="text-sm font-medium text-zinc-300">No results yet</p>
          <p className="mt-1 text-sm text-zinc-500">
            Upload pre/post imagery to generate your first damage assessment.
          </p>
          <Link
            href="/upload"
            className="mt-4 inline-flex rounded-md bg-blue-600 hover:bg-blue-500 px-3 py-1.5 text-sm font-medium text-white transition-colors"
          >
            Run Inference
          </Link>
        </div>
      ) : (
        <>
          {/* Two-column charts */}
          <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-sm font-medium text-zinc-300">
                  Average Damage Distribution
                </h2>
                <span className="text-xs text-zinc-500 tabular-nums">
                  n={results.length}
                </span>
              </div>
              <DamageChart damagePct={avg} />
              <div className="mt-3 flex flex-wrap gap-3">
                {DAMAGE_CLASSES.map((cls) => (
                  <div
                    key={cls}
                    className="flex items-center gap-1.5 text-xs text-zinc-400"
                  >
                    <span
                      aria-hidden="true"
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: colorFor(cls) }}
                    />
                    {cls}
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
              <h2 className="text-sm font-medium text-zinc-300 mb-3">
                Average Breakdown
              </h2>
              <div className="space-y-3 mt-2">
                {DAMAGE_CLASSES.map((cls) => (
                  <DamageBar key={cls} cls={cls} pct={avg[cls] ?? 0} />
                ))}
              </div>
            </div>
          </section>

          {/* Recent Results table */}
          <section className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
            <div className="px-4 py-3 border-b border-zinc-800 flex items-center justify-between">
              <h2 className="text-sm font-medium text-zinc-300">Recent Results</h2>
              <span className="text-xs text-zinc-500 tabular-nums">
                Showing {Math.min(20, sorted.length)} of {sorted.length}
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-zinc-950/50">
                  <tr className="text-left text-xs uppercase tracking-wide text-zinc-500">
                    <th className="px-4 py-2.5 font-medium">Site</th>
                    <th className="px-4 py-2.5 font-medium">Timestamp</th>
                    {DAMAGE_CLASSES.map((cls) => (
                      <th
                        key={cls}
                        className="px-3 py-2.5 font-medium text-right"
                      >
                        {cls}
                      </th>
                    ))}
                    <th className="px-4 py-2.5 font-medium">Severity</th>
                    <th className="px-4 py-2.5 font-medium text-right"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800">
                  {sorted.slice(0, 20).map((r) => (
                    <tr
                      key={r.id}
                      className="hover:bg-zinc-800/50 transition-colors"
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          {r.mask_url ? (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img
                              src={`${apiUrl}${r.mask_url}`}
                              alt=""
                              className="w-9 h-9 rounded border border-zinc-800 object-cover shrink-0"
                            />
                          ) : (
                            <div className="w-9 h-9 rounded border border-zinc-800 bg-zinc-800/50 shrink-0" />
                          )}
                          <span className="font-medium text-zinc-200 truncate max-w-45">
                            {r.stem}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-zinc-400 text-xs tabular-nums whitespace-nowrap">
                        {formatTimestamp(r.timestamp)}
                      </td>
                      {DAMAGE_CLASSES.map((cls) => {
                        const pct = r.damage_pct[cls] ?? 0;
                        return (
                          <td
                            key={cls}
                            className="px-3 py-3 text-right tabular-nums text-xs"
                            style={{
                              color: pct > 0 ? colorFor(cls) : "#52525b",
                            }}
                          >
                            {pct.toFixed(1)}%
                          </td>
                        );
                      })}
                      <td className="px-4 py-3">
                        <SeverityBadge damagePct={r.damage_pct} />
                      </td>
                      <td className="px-4 py-3 text-right">
                        <Link
                          href="/map"
                          className="text-xs text-blue-400 hover:text-blue-300 hover:underline"
                        >
                          View
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
