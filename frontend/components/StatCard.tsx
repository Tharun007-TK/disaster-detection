import type { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  /** Optional accent color (hex) used for the value. */
  color?: string;
}

export default function StatCard({ label, value, hint, color }: StatCardProps) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <p className="text-xs text-zinc-400 uppercase tracking-wide">{label}</p>
      <p
        className="mt-2 text-2xl font-semibold tabular-nums text-zinc-100"
        style={color ? { color } : undefined}
      >
        {value}
      </p>
      {hint !== undefined && hint !== null && (
        <p className="mt-1 text-xs text-zinc-500">{hint}</p>
      )}
    </div>
  );
}
