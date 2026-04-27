// Centralized severity tokens for damage classes.
// Use everywhere damage classes appear so colors stay consistent.

export type DamageClass = "No Damage" | "Minor" | "Major" | "Destroyed";

export const DAMAGE_CLASSES: DamageClass[] = [
  "No Damage",
  "Minor",
  "Major",
  "Destroyed",
];

export const SEVERITY_COLOR: Record<DamageClass, string> = {
  "No Damage": "#22c55e", // green-500
  Minor: "#eab308", // yellow-500
  Major: "#f97316", // orange-500
  Destroyed: "#ef4444", // red-500
};

// Numeric class id (0..3) -> name
export const CLASS_NAME_BY_ID: Record<number, DamageClass> = {
  0: "No Damage",
  1: "Minor",
  2: "Major",
  3: "Destroyed",
};

// Tailwind text classes for accent text where a hex isn't ideal.
export const SEVERITY_TEXT_CLASS: Record<DamageClass, string> = {
  "No Damage": "text-green-500",
  Minor: "text-yellow-500",
  Major: "text-orange-500",
  Destroyed: "text-red-500",
};

// Tailwind border class for left-accent cards etc.
export const SEVERITY_BORDER_CLASS: Record<DamageClass, string> = {
  "No Damage": "border-green-500",
  Minor: "border-yellow-500",
  Major: "border-orange-500",
  Destroyed: "border-red-500",
};

export function colorFor(cls: string): string {
  return SEVERITY_COLOR[cls as DamageClass] ?? "#71717a"; // zinc-500 fallback
}

// Pick the worst (highest severity) class with a non-zero value.
// Tie-breaking goes to the more severe class.
export function worstClass(damagePct: Record<string, number>): DamageClass {
  let worst: DamageClass = "No Damage";
  let worstIdx = -1;
  const order: DamageClass[] = DAMAGE_CLASSES;
  for (let i = 0; i < order.length; i++) {
    const cls = order[i];
    const pct = damagePct[cls] ?? 0;
    if (pct > 0 && i >= worstIdx) {
      worst = cls;
      worstIdx = i;
    }
  }
  return worst;
}

// The dominant (largest %) class. Useful for map markers.
export function dominantClass(damagePct: Record<string, number>): DamageClass {
  let best: DamageClass = "No Damage";
  let bestPct = -1;
  for (const cls of DAMAGE_CLASSES) {
    const pct = damagePct[cls] ?? 0;
    if (pct > bestPct) {
      bestPct = pct;
      best = cls;
    }
  }
  return best;
}

// Format a relative time like "3m ago" / "2h ago" / "5d ago".
export function relativeTime(iso?: string): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const diff = Math.max(0, Date.now() - t);
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 30) return `${d}d ago`;
  const mo = Math.floor(d / 30);
  if (mo < 12) return `${mo}mo ago`;
  const y = Math.floor(mo / 12);
  return `${y}y ago`;
}
