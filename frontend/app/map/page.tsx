"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import {
  colorFor,
  dominantClass,
  worstClass,
  DAMAGE_CLASSES,
} from "@/lib/severity";
import SeverityBadge from "@/components/SeverityBadge";

const MapView = dynamic(() => import("@/components/MapView"), { ssr: false });

interface Result {
  id: string;
  stem: string;
  damage_pct: Record<string, number>;
  mask_url?: string;
  timestamp?: string;
  lat?: number;
  lng?: number;
  bounds?: { left: number; bottom: number; right: number; top: number };
}

export default function MapPage() {
  const [results, setResults] = useState<Result[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  useEffect(() => {
    fetch("/api/results", { cache: "no-store" })
      .then((r) => r.json())
      .then((data) => {
        setResults(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const sorted = useMemo(() => {
    return [...results].sort((a, b) => {
      const ta = a.timestamp ? Date.parse(a.timestamp) : 0;
      const tb = b.timestamp ? Date.parse(b.timestamp) : 0;
      return tb - ta;
    });
  }, [results]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return sorted;
    return sorted.filter((r) => r.stem.toLowerCase().includes(q));
  }, [sorted, search]);

  const geoCount = useMemo(
    () => results.filter((r) => r.lat != null && r.lng != null).length,
    [results]
  );

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* Page header */}
      <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between px-6 lg:px-8 py-4 border-b border-zinc-800">
        <div className="space-y-0.5">
          <h1 className="text-xl font-semibold tracking-tight text-zinc-100">
            Damage Map
          </h1>
          <p className="text-xs text-zinc-500">
            <span className="tabular-nums">{geoCount}</span> of{" "}
            <span className="tabular-nums">{results.length}</span> sites plotted ·
            positions derived from GeoTIFF geotransform.
          </p>
        </div>
      </header>

      {/* Two-column body — sidebar + map */}
      <div className="flex-1 flex flex-col lg:flex-row min-h-0">
        {/* In-page sidebar */}
        <aside className="lg:w-80 lg:shrink-0 border-b lg:border-b-0 lg:border-r border-zinc-800 bg-zinc-950 flex flex-col min-h-0">
          <div className="px-4 py-3 border-b border-zinc-800">
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search sites..."
              className="w-full rounded-md border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              aria-label="Search sites"
            />
          </div>

          <div className="flex-1 overflow-y-auto">
            {loading ? (
              <div className="p-4 text-sm text-zinc-500">Loading...</div>
            ) : filtered.length === 0 ? (
              <div className="p-6 text-center">
                <p className="text-sm text-zinc-300 font-medium">
                  {results.length === 0 ? "No results yet" : "No matching sites"}
                </p>
                {results.length === 0 && (
                  <>
                    <p className="mt-1 text-xs text-zinc-500">
                      Run inference to populate the map.
                    </p>
                    <Link
                      href="/upload"
                      className="mt-3 inline-flex rounded-md bg-blue-600 hover:bg-blue-500 px-3 py-1.5 text-xs font-medium text-white"
                    >
                      Run Inference
                    </Link>
                  </>
                )}
              </div>
            ) : (
              <ul className="divide-y divide-zinc-800">
                {filtered.map((r) => {
                  const dom = dominantClass(r.damage_pct);
                  const worst = worstClass(r.damage_pct);
                  const active = selectedId === r.id;
                  const hasGeo = r.lat != null && r.lng != null;
                  return (
                    <li key={r.id}>
                      <button
                        type="button"
                        onClick={() => setSelectedId(r.id)}
                        disabled={!hasGeo}
                        className={[
                          "w-full text-left px-4 py-3 transition-colors focus:outline-none",
                          active
                            ? "bg-zinc-800/70"
                            : "hover:bg-zinc-800/50",
                          !hasGeo && "opacity-50 cursor-not-allowed",
                        ]
                          .filter(Boolean)
                          .join(" ")}
                        aria-pressed={active}
                      >
                        <div className="flex items-center gap-2">
                          <span
                            aria-hidden="true"
                            className="w-2 h-2 rounded-full shrink-0"
                            style={{ backgroundColor: colorFor(dom) }}
                          />
                          <span className="text-sm text-zinc-200 font-medium truncate flex-1">
                            {r.stem}
                          </span>
                        </div>
                        <div className="mt-1.5 flex items-center justify-between gap-2 pl-4">
                          <span className="text-[10px] uppercase tracking-wider text-zinc-500 tabular-nums">
                            {hasGeo
                              ? `${r.lat!.toFixed(3)}, ${r.lng!.toFixed(3)}`
                              : "no geolocation"}
                          </span>
                          <SeverityBadge damagePct={r.damage_pct} />
                        </div>
                        <div className="mt-1.5 flex gap-2 pl-4 text-[10px] tabular-nums">
                          {DAMAGE_CLASSES.map((cls) => {
                            const pct = r.damage_pct[cls] ?? 0;
                            const isWorst = cls === worst;
                            return (
                              <span
                                key={cls}
                                className={isWorst ? "font-semibold" : ""}
                                style={{
                                  color: pct > 0 ? colorFor(cls) : "#52525b",
                                }}
                              >
                                {pct.toFixed(0)}%
                              </span>
                            );
                          })}
                        </div>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </aside>

        {/* Map */}
        <div className="flex-1 min-h-100 lg:min-h-0 bg-zinc-950 relative">
          {loading ? (
            <div className="h-full flex items-center justify-center text-sm text-zinc-500">
              Loading map...
            </div>
          ) : (
            <MapView
              results={results}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          )}
        </div>
      </div>
    </div>
  );
}
