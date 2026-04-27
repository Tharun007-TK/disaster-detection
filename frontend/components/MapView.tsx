"use client";

import { useEffect } from "react";
import { MapContainer, TileLayer, Popup, CircleMarker, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const CLASS_COLORS: Record<string, string> = {
  "No Damage": "#16a34a",
  Minor: "#ca8a04",
  Major: "#ea580c",
  Destroyed: "#dc2626",
};

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

function FitBounds({ results }: { results: Result[] }) {
  const map = useMap();
  useEffect(() => {
    const t = setTimeout(() => {
      map.invalidateSize();
      const geo = results.filter((r) => r.lat != null && r.lng != null);
      if (geo.length === 0) return;
      if (geo.length === 1) {
        map.setView([geo[0].lat!, geo[0].lng!], 14);
        return;
      }
      const lats = geo.map((r) => r.lat!);
      const lngs = geo.map((r) => r.lng!);
      map.fitBounds(
        [[Math.min(...lats), Math.min(...lngs)], [Math.max(...lats), Math.max(...lngs)]],
        { padding: [40, 40] }
      );
    }, 100);
    return () => clearTimeout(t);
  }, [map, results]);
  return null;
}

function dominantClass(damage_pct: Record<string, number>): string {
  return Object.entries(damage_pct).reduce(
    (a, b) => (b[1] > a[1] ? b : a),
    ["No Damage", 0] as [string, number]
  )[0];
}

export default function MapView({ results }: { results: Result[] }) {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  return (
    <MapContainer
      center={[20, 0]}
      zoom={2}
      style={{ height: "500px", width: "100%" }}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <FitBounds results={results} />
      {results.filter((r) => r.lat != null && r.lng != null).map((r) => {
        const dom = dominantClass(r.damage_pct);
        return (
          <CircleMarker
            key={r.id}
            center={[r.lat!, r.lng!]}
            radius={12}
            pathOptions={{ color: CLASS_COLORS[dom] ?? "#6b7280", fillOpacity: 0.7 }}
          >
            <Popup>
              <div className="text-sm">
                <p className="font-bold mb-1">{r.stem}</p>
                {Object.entries(r.damage_pct).map(([cls, pct]) => (
                  <p key={cls}>
                    <span style={{ color: CLASS_COLORS[cls] }}>{cls}</span>: {pct.toFixed(1)}%
                  </p>
                ))}
                {r.mask_url && (
                  <img
                    src={`${apiUrl}${r.mask_url}`}
                    alt="damage mask"
                    className="mt-2 w-32 h-32 object-cover rounded"
                  />
                )}
              </div>
            </Popup>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
