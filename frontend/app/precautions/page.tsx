"use client";

import { useEffect, useMemo, useState } from "react";
import { CLASS_NAME_BY_ID, SEVERITY_COLOR } from "@/lib/severity";

interface PrecautionInfo {
  damage_class: number;
  name: string;
  color: string;
  precautions: string[];
  actions: string[];
  evacuation: string;
}

function colorForClassId(id: number, fallback?: string): string {
  const name = CLASS_NAME_BY_ID[id];
  return (name && SEVERITY_COLOR[name]) || fallback || "#71717a";
}

function sectionId(id: number) {
  return `damage-class-${id}`;
}

export default function PrecautionsPage() {
  const [data, setData] = useState<PrecautionInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/precautions", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => {
        setData(Array.isArray(d) ? d : []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const sorted = useMemo(
    () => [...data].sort((a, b) => a.damage_class - b.damage_class),
    [data]
  );

  return (
    <div className="p-6 lg:p-8 space-y-6 max-w-7xl">
      {/* Header */}
      <header className="space-y-1">
        <h1 className="text-xl font-semibold tracking-tight text-zinc-100">
          Precaution Protocols
        </h1>
        <p className="text-sm text-zinc-400">
          Recommended actions, safety protocols, and evacuation guidance per damage
          classification.
        </p>
      </header>

      {loading ? (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-8 text-center text-sm text-zinc-400">
          Loading...
        </div>
      ) : sorted.length === 0 ? (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-8 text-center text-sm text-zinc-400">
          No precaution data available.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-[200px_minmax(0,1fr)] gap-6">
          {/* Sticky class navigator */}
          <nav
            aria-label="Damage classes"
            className="hidden lg:block self-start sticky top-6"
          >
            <p className="text-xs uppercase tracking-wide text-zinc-500 mb-2 px-3">
              Classes
            </p>
            <ul className="space-y-1">
              {sorted.map((item) => {
                const color = colorForClassId(item.damage_class, item.color);
                return (
                  <li key={item.damage_class}>
                    <a
                      href={`#${sectionId(item.damage_class)}`}
                      className="group flex items-center gap-2 rounded-md px-3 py-2 text-sm text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 transition-colors"
                    >
                      <span
                        aria-hidden="true"
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: color }}
                      />
                      <span className="flex-1">{item.name}</span>
                      <span className="text-xs text-zinc-600 tabular-nums">
                        {item.damage_class}
                      </span>
                    </a>
                  </li>
                );
              })}
            </ul>
          </nav>

          {/* Cards stack */}
          <div className="space-y-4">
            {sorted.map((item) => {
              const color = colorForClassId(item.damage_class, item.color);
              return (
                <article
                  id={sectionId(item.damage_class)}
                  key={item.damage_class}
                  className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden scroll-mt-6"
                  style={{ borderLeftColor: color, borderLeftWidth: 3 }}
                >
                  {/* Card header */}
                  <header className="flex items-center gap-3 px-5 py-4 border-b border-zinc-800">
                    <span
                      aria-hidden="true"
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <h2 className="text-base font-semibold text-zinc-100">
                      {item.name}
                    </h2>
                    <span
                      className="ml-auto inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium tabular-nums"
                      style={{
                        color,
                        backgroundColor: `${color}1f`,
                        boxShadow: `inset 0 0 0 1px ${color}40`,
                      }}
                    >
                      Class {item.damage_class}
                    </span>
                  </header>

                  <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-zinc-800">
                    {/* Precautions / Safety Protocols */}
                    <div className="p-5 space-y-2">
                      <h3 className="text-xs font-medium uppercase tracking-wide text-zinc-400">
                        Safety Protocols
                      </h3>
                      <ul className="space-y-1.5">
                        {item.precautions.map((p, i) => (
                          <li
                            key={i}
                            className="text-sm text-zinc-300 flex gap-2 leading-relaxed"
                          >
                            <span
                              aria-hidden="true"
                              className="mt-1.5 w-1 h-1 rounded-full shrink-0"
                              style={{ backgroundColor: color }}
                            />
                            <span className="flex-1">{p}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Recommended Actions */}
                    <div className="p-5 space-y-2">
                      <h3 className="text-xs font-medium uppercase tracking-wide text-zinc-400">
                        Recommended Actions
                      </h3>
                      <ol className="space-y-1.5">
                        {item.actions.map((a, i) => (
                          <li
                            key={i}
                            className="text-sm text-zinc-300 flex gap-2 leading-relaxed"
                          >
                            <span
                              className="text-xs font-medium tabular-nums shrink-0 w-5 text-right"
                              style={{ color }}
                            >
                              {i + 1}.
                            </span>
                            <span className="flex-1">{a}</span>
                          </li>
                        ))}
                      </ol>
                    </div>
                  </div>

                  {/* Evacuation */}
                  <footer className="px-5 py-3 bg-zinc-800/30 border-t border-zinc-800">
                    <p className="text-xs uppercase tracking-wide text-zinc-400 mb-1">
                      Evacuation
                    </p>
                    <p className="text-sm text-zinc-200">{item.evacuation}</p>
                  </footer>
                </article>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
