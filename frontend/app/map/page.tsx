"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";

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

  useEffect(() => {
    fetch("/api/results")
      .then((r) => r.json())
      .then((data) => { setResults(Array.isArray(data) ? data : []); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Damage Map</h1>
        <p className="text-zinc-400 text-sm mt-1">
          {results.length} result{results.length !== 1 ? "s" : ""} plotted
        </p>
      </div>

      {loading ? (
        <div className="h-[500px] rounded-lg border border-zinc-800 bg-zinc-900 flex items-center justify-center text-zinc-400">
          Loading...
        </div>
      ) : (
        <div className="rounded-lg overflow-hidden border border-zinc-800">
          <MapView results={results} />
        </div>
      )}

      <p className="text-xs text-zinc-500">
        Positions derived from GeoTIFF geotransform. Only results with valid CRS are plotted.
      </p>
    </div>
  );
}
