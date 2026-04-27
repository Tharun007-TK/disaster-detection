"use client";

import { useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  Popup,
  CircleMarker,
  ImageOverlay,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { colorFor, dominantClass, DAMAGE_CLASSES } from "@/lib/severity";

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

interface Props {
  results: Result[];
  selectedId?: string | null;
  onSelect?: (id: string) => void;
}

function FlyToSelected({
  results,
  selectedId,
}: {
  results: Result[];
  selectedId?: string | null;
}) {
  const map = useMap();
  useEffect(() => {
    if (!selectedId) return;
    const r = results.find((x) => x.id === selectedId);
    if (!r) return;
    if (r.bounds) {
      map.flyToBounds(
        [
          [r.bounds.bottom, r.bounds.left],
          [r.bounds.top, r.bounds.right],
        ],
        { padding: [40, 40], duration: 0.6 }
      );
    } else if (r.lat != null && r.lng != null) {
      map.flyTo([r.lat, r.lng], 14, { duration: 0.6 });
    }
  }, [map, results, selectedId]);
  return null;
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
        [
          [Math.min(...lats), Math.min(...lngs)],
          [Math.max(...lats), Math.max(...lngs)],
        ],
        { padding: [40, 40] }
      );
    }, 100);
    return () => clearTimeout(t);
  }, [map, results]);
  return null;
}

export default function MapView({ results, selectedId, onSelect }: Props) {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  const geo = results.filter((r) => r.lat != null && r.lng != null);

  return (
    <div className="relative h-full w-full">
      <MapContainer
        center={[20, 0]}
        zoom={2}
        style={{ height: "100%", width: "100%", backgroundColor: "#09090b" }}
        zoomControl={true}
      >
        <TileLayer
          attribution='&copy; OpenStreetMap'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <FitBounds results={geo} />
        <FlyToSelected results={results} selectedId={selectedId} />

        {/* Damage mask image overlays for results that have bounds */}
        {results
          .filter((r) => r.mask_url && r.bounds)
          .filter((r) => !selectedId || r.id === selectedId)
          .map((r) => (
            <ImageOverlay
              key={`overlay-${r.id}`}
              url={`${apiUrl}${r.mask_url}`}
              bounds={[
                [r.bounds!.bottom, r.bounds!.left],
                [r.bounds!.top, r.bounds!.right],
              ]}
              opacity={0.65}
            />
          ))}

        {geo.map((r) => {
          const dom = dominantClass(r.damage_pct);
          const color = colorFor(dom);
          const active = selectedId === r.id;
          return (
            <CircleMarker
              key={r.id}
              center={[r.lat!, r.lng!]}
              radius={active ? 14 : 10}
              pathOptions={{
                color,
                weight: active ? 3 : 2,
                fillColor: color,
                fillOpacity: active ? 0.85 : 0.6,
              }}
              eventHandlers={{
                click: () => onSelect?.(r.id),
              }}
            >
              <Popup>
                <div className="text-xs">
                  <p className="font-semibold mb-1">{r.stem}</p>
                  {DAMAGE_CLASSES.filter(
                    (cls) => r.damage_pct[cls] !== undefined
                  ).map((cls) => (
                    <p key={cls} className="leading-tight">
                      <span style={{ color: colorFor(cls) }}>{cls}</span>:{" "}
                      {(r.damage_pct[cls] ?? 0).toFixed(1)}%
                    </p>
                  ))}
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>

      {/* Legend */}
      <div className="absolute bottom-4 right-4 z-[1000] rounded-lg border border-zinc-800 bg-zinc-900/95 px-3 py-2 shadow-lg">
        <p className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5">
          Severity
        </p>
        <div className="space-y-1">
          {DAMAGE_CLASSES.map((cls) => (
            <div key={cls} className="flex items-center gap-2 text-xs text-zinc-300">
              <span
                aria-hidden="true"
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: colorFor(cls) }}
              />
              <span>{cls}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
