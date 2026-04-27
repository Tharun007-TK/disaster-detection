"use client";

import { useState, useRef, FormEvent } from "react";
import DamageChart from "@/components/DamageChart";

type Mode = "pair" | "single";

interface InferenceResult {
  height: number;
  width: number;
  damage_pct: Record<string, number>;
  pixel_counts: Record<string, number>;
  outputs?: { mask_png: string; mask_tif: string; overlay_png?: string };
  crs?: string;
  mode?: string;
}

const CLASS_BG: Record<string, string> = {
  "No Damage": "bg-green-900/40 border-green-700",
  Minor: "bg-yellow-900/40 border-yellow-700",
  Major: "bg-orange-900/40 border-orange-700",
  Destroyed: "bg-red-900/40 border-red-700",
};

export default function UploadPage() {
  const [mode, setMode] = useState<Mode>("pair");
  const preRef = useRef<HTMLInputElement>(null);
  const postRef = useRef<HTMLInputElement>(null);
  const singleRef = useRef<HTMLInputElement>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const fd = new FormData();
    let endpoint: string;

    if (mode === "pair") {
      const pre = preRef.current?.files?.[0];
      const post = postRef.current?.files?.[0];
      if (!pre || !post) { setError("Select both pre and post files."); setLoading(false); return; }
      fd.append("pre", pre);
      fd.append("post", post);
      endpoint = "/api/inference";
    } else {
      const img = singleRef.current?.files?.[0];
      if (!img) { setError("Select a disaster image."); setLoading(false); return; }
      fd.append("image", img);
      endpoint = "/api/inference/single";
    }

    try {
      const res = await fetch(endpoint, { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed");
    } finally {
      setLoading(false);
    }
  }

  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 space-y-8">
      <h1 className="text-2xl font-bold">Run Inference</h1>

      <div className="flex rounded-lg overflow-hidden border border-zinc-700 w-fit">
        {(["pair", "single"] as Mode[]).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => { setMode(m); setResult(null); setError(null); }}
            className={`px-5 py-2 text-sm font-medium transition-colors ${
              mode === m
                ? "bg-blue-600 text-white"
                : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
            }`}
          >
            {m === "pair" ? "Pre + Post Pair" : "Single Image"}
          </button>
        ))}
      </div>

      <form
        onSubmit={handleSubmit}
        className="rounded-lg border border-zinc-800 bg-zinc-900 p-6 space-y-5"
      >
        {mode === "pair" ? (
          <>
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-1">
                Pre-event GeoTIFF (3-band RGB)
              </label>
              <input
                ref={preRef}
                type="file"
                accept=".tif,.tiff,.png,.jpg,.jpeg"
                required
                className="block w-full text-sm text-zinc-400 file:mr-3 file:rounded file:border-0 file:bg-zinc-700 file:px-3 file:py-1.5 file:text-sm file:text-zinc-200 file:cursor-pointer hover:file:bg-zinc-600"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-1">
                Post-event GeoTIFF (1-band SAR)
              </label>
              <input
                ref={postRef}
                type="file"
                accept=".tif,.tiff,.png,.jpg,.jpeg"
                required
                className="block w-full text-sm text-zinc-400 file:mr-3 file:rounded file:border-0 file:bg-zinc-700 file:px-3 file:py-1.5 file:text-sm file:text-zinc-200 file:cursor-pointer hover:file:bg-zinc-600"
              />
            </div>
          </>
        ) : (
          <div>
            <label className="block text-sm font-medium text-zinc-300 mb-1">
              Disaster Image GeoTIFF (any band count)
            </label>
            <input
              ref={singleRef}
              type="file"
              accept=".tif,.tiff,.png,.jpg,.jpeg"
              required
              className="block w-full text-sm text-zinc-400 file:mr-3 file:rounded file:border-0 file:bg-zinc-700 file:px-3 file:py-1.5 file:text-sm file:text-zinc-200 file:cursor-pointer hover:file:bg-zinc-600"
            />
            <p className="text-xs text-zinc-500 mt-1">
              Single-image mode classifies damage from absolute pixel intensity — no temporal comparison needed.
            </p>
          </div>
        )}

        {error && <p className="text-sm text-red-400">{error}</p>}
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white py-2 rounded font-medium transition-colors"
        >
          {loading ? "Running inference..." : "Analyze Damage"}
        </button>
      </form>

      {result && (
        <div className="space-y-6">
          <div className="flex items-center gap-4">
            <h2 className="text-lg font-semibold">Result</h2>
            <span className="text-xs text-zinc-500">
              {result.height} x {result.width} px
              {result.crs ? ` · ${result.crs}` : ""}
              {result.mode === "single" ? " · single-image mode" : ""}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {Object.entries(result.damage_pct).map(([cls, pct]) => (
              <div
                key={cls}
                className={`rounded-lg border p-4 ${CLASS_BG[cls] ?? "bg-zinc-800 border-zinc-700"}`}
              >
                <p className="text-xs text-zinc-400 uppercase tracking-wider">{cls}</p>
                <p className="text-2xl font-bold mt-1">{pct.toFixed(2)}%</p>
                <p className="text-xs text-zinc-500 mt-0.5">
                  {result.pixel_counts[cls]?.toLocaleString()} px
                </p>
              </div>
            ))}
          </div>

          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
            <DamageChart damagePct={result.damage_pct} />
          </div>

          {(result.outputs?.overlay_png || result.outputs?.mask_png) && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 space-y-4">
              {result.outputs?.overlay_png && (
                <div>
                  <p className="text-sm font-medium text-zinc-300 mb-2">Damage Segmentation Overlay</p>
                  <img
                    src={`${apiUrl}/outputs/${result.outputs.overlay_png.split(/[\\/]/).pop()}`}
                    alt="damage overlay"
                    className="rounded w-full max-h-[500px] object-contain"
                  />
                  <p className="text-xs text-zinc-500 mt-1">
                    Green = No Damage · Yellow = Minor · Orange = Major · Red = Destroyed
                  </p>
                </div>
              )}
              {result.outputs?.mask_png && (
                <details className="group">
                  <summary className="text-sm text-zinc-400 cursor-pointer hover:text-zinc-200 select-none">
                    Raw damage mask
                  </summary>
                  <img
                    src={`${apiUrl}/outputs/${result.outputs.mask_png.split(/[\\/]/).pop()}`}
                    alt="damage mask"
                    className="rounded mt-2 max-h-96 object-contain"
                  />
                </details>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
