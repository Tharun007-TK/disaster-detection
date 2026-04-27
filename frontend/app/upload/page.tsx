"use client";

import { useState, useRef, FormEvent, useCallback, DragEvent } from "react";
import DamageChart from "@/components/DamageChart";
import DamageBar from "@/components/DamageBar";
import { DAMAGE_CLASSES, colorFor } from "@/lib/severity";

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

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

interface DropZoneProps {
  label: string;
  hint: string;
  file: File | null;
  onFile: (f: File | null) => void;
}

function DropZone({ label, hint, file, onFile }: DropZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const onDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };

  return (
    <div className="space-y-1.5">
      <p className="text-xs uppercase tracking-wide text-zinc-400">{label}</p>
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={[
          "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-8 cursor-pointer transition-colors",
          dragOver
            ? "border-blue-500 bg-blue-500/5"
            : file
            ? "border-zinc-700 bg-zinc-900"
            : "border-zinc-800 bg-zinc-900 hover:border-zinc-700 hover:bg-zinc-800/50",
        ].join(" ")}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          className="sr-only"
          onChange={(e) => onFile(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <div className="text-center space-y-1">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="w-6 h-6 mx-auto text-emerald-500"
              aria-hidden="true"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <polyline points="9 14 11 16 15 12" />
            </svg>
            <p className="text-sm text-zinc-200 font-medium truncate max-w-xs">
              {file.name}
            </p>
            <p className="text-xs text-zinc-500 tabular-nums">
              {formatBytes(file.size)}
            </p>
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                onFile(null);
                if (inputRef.current) inputRef.current.value = "";
              }}
              className="text-xs text-zinc-400 hover:text-zinc-200 underline-offset-2 hover:underline"
            >
              Replace
            </button>
          </div>
        ) : (
          <div className="text-center space-y-1">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="w-6 h-6 mx-auto text-zinc-500"
              aria-hidden="true"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <p className="text-sm text-zinc-300 font-medium">
              Drop file or click to browse
            </p>
            <p className="text-xs text-zinc-500">{hint}</p>
          </div>
        )}
      </label>
    </div>
  );
}

function Spinner() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      className="w-4 h-4 animate-spin"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path
        d="M22 12a10 10 0 0 1-10 10"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

export default function UploadPage() {
  const [mode, setMode] = useState<Mode>("pair");
  const [preFile, setPreFile] = useState<File | null>(null);
  const [postFile, setPostFile] = useState<File | null>(null);
  const [singleFile, setSingleFile] = useState<File | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  const ready =
    mode === "pair" ? Boolean(preFile && postFile) : Boolean(singleFile);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!ready) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const fd = new FormData();
    let endpoint: string;

    if (mode === "pair") {
      if (!preFile || !postFile) {
        setError("Select both pre and post files.");
        setLoading(false);
        return;
      }
      fd.append("pre", preFile);
      fd.append("post", postFile);
      endpoint = `${apiUrl}/api/inference`;
    } else {
      if (!singleFile) {
        setError("Select a disaster image.");
        setLoading(false);
        return;
      }
      fd.append("image", singleFile);
      endpoint = `${apiUrl}/api/inference/single`;
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

  return (
    <div className="p-6 lg:p-8 space-y-6 max-w-5xl">
      {/* Header */}
      <header className="space-y-1">
        <h1 className="text-xl font-semibold tracking-tight text-zinc-100">
          Run Inference
        </h1>
        <p className="text-sm text-zinc-400">
          Upload pre/post GeoTIFF imagery to generate a damage assessment.
        </p>
      </header>

      {/* Mode toggle */}
      <div
        role="tablist"
        aria-label="Inference mode"
        className="inline-flex rounded-md border border-zinc-800 bg-zinc-900 p-1"
      >
        {(
          [
            { id: "pair" as const, label: "Pre + Post Pair" },
            { id: "single" as const, label: "Single Image" },
          ]
        ).map((m) => (
          <button
            key={m.id}
            type="button"
            role="tab"
            aria-selected={mode === m.id}
            onClick={() => {
              setMode(m.id);
              reset();
            }}
            className={[
              "px-3 py-1.5 text-sm rounded transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500",
              mode === m.id
                ? "bg-zinc-800 text-zinc-100"
                : "text-zinc-400 hover:text-zinc-200",
            ].join(" ")}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Upload form */}
      <form
        onSubmit={handleSubmit}
        className="rounded-lg border border-zinc-800 bg-zinc-900 p-6 space-y-5"
      >
        {mode === "pair" ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <DropZone
              label="Pre-event"
              hint=".tif / .tiff (3-band RGB)"
              file={preFile}
              onFile={(f) => {
                setPreFile(f);
                reset();
              }}
            />
            <DropZone
              label="Post-event"
              hint=".tif / .tiff (1-band SAR)"
              file={postFile}
              onFile={(f) => {
                setPostFile(f);
                reset();
              }}
            />
          </div>
        ) : (
          <div className="space-y-2">
            <DropZone
              label="Disaster Image"
              hint=".tif / .tiff (any band count)"
              file={singleFile}
              onFile={(f) => {
                setSingleFile(f);
                reset();
              }}
            />
            <p className="text-xs text-zinc-500">
              Single-image mode classifies damage from absolute pixel intensity — no
              temporal comparison required.
            </p>
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="rounded-md border border-red-900/60 bg-red-950/40 px-3 py-2 text-sm text-red-300"
          >
            {error}
          </div>
        )}

        <div className="flex items-center justify-between gap-4">
          <p className="text-xs text-zinc-500">
            {ready
              ? "Ready to analyze."
              : mode === "pair"
              ? "Select both pre and post files."
              : "Select a disaster image."}
          </p>
          <button
            type="submit"
            disabled={loading || !ready}
            className="inline-flex items-center gap-2 rounded-md bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white px-4 py-1.5 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {loading && <Spinner />}
            {loading ? "Running inference..." : "Analyze Damage"}
          </button>
        </div>
      </form>

      {/* Result panel */}
      {result && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-zinc-300">Result</h2>
            <p className="text-xs text-zinc-500 tabular-nums">
              {result.height} × {result.width} px
              {result.crs ? ` · ${result.crs}` : ""}
              {result.mode === "single" ? " · single-image mode" : ""}
            </p>
          </div>

          {/* KPI tiles per class */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {DAMAGE_CLASSES.map((cls) => {
              const pct = result.damage_pct[cls] ?? 0;
              const px = result.pixel_counts[cls] ?? 0;
              return (
                <div
                  key={cls}
                  className="rounded-lg border border-zinc-800 bg-zinc-900 p-4"
                  style={{ borderLeftColor: colorFor(cls), borderLeftWidth: 3 }}
                >
                  <p className="text-xs uppercase tracking-wide text-zinc-400">
                    {cls}
                  </p>
                  <p
                    className="mt-2 text-2xl font-semibold tabular-nums"
                    style={{ color: colorFor(cls) }}
                  >
                    {pct.toFixed(2)}%
                  </p>
                  <p className="mt-1 text-xs text-zinc-500 tabular-nums">
                    {px.toLocaleString()} px
                  </p>
                </div>
              );
            })}
          </div>

          {/* Chart + breakdown */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
              <h3 className="text-sm font-medium text-zinc-300 mb-2">
                Damage Distribution
              </h3>
              <DamageChart damagePct={result.damage_pct} />
            </div>
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
              <h3 className="text-sm font-medium text-zinc-300 mb-3">
                Class Breakdown
              </h3>
              <div className="space-y-3">
                {DAMAGE_CLASSES.map((cls) => (
                  <DamageBar
                    key={cls}
                    cls={cls}
                    pct={result.damage_pct[cls] ?? 0}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Outputs */}
          {(result.outputs?.overlay_png || result.outputs?.mask_png) && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 space-y-4">
              {result.outputs?.overlay_png && (
                <div>
                  <h3 className="text-sm font-medium text-zinc-300 mb-2">
                    Segmentation Overlay
                  </h3>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`${apiUrl}/outputs/${result.outputs.overlay_png
                      .split(/[\\/]/)
                      .pop()}`}
                    alt="damage overlay"
                    className="rounded-md w-full max-h-125 object-contain bg-zinc-950 border border-zinc-800"
                  />
                  <div className="mt-2 flex flex-wrap gap-3">
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
              )}
              {result.outputs?.mask_png && (
                <details className="group">
                  <summary className="text-sm text-zinc-400 cursor-pointer hover:text-zinc-200 select-none list-none">
                    <span className="inline-flex items-center gap-1.5">
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="w-3.5 h-3.5 transition-transform group-open:rotate-90"
                        aria-hidden="true"
                      >
                        <polyline points="9 18 15 12 9 6" />
                      </svg>
                      Raw damage mask
                    </span>
                  </summary>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`${apiUrl}/outputs/${result.outputs.mask_png
                      .split(/[\\/]/)
                      .pop()}`}
                    alt="damage mask"
                    className="rounded-md mt-2 max-h-96 object-contain border border-zinc-800"
                  />
                </details>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
