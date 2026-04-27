import { colorFor } from "@/lib/severity";

interface Props {
  cls: string;
  pct: number;
}

export default function DamageBar({ cls, pct }: Props) {
  const color = colorFor(cls);
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-zinc-300 w-24 shrink-0">{cls}</span>
      <div className="flex-1 h-1.5 rounded-full bg-zinc-800 overflow-hidden">
        <div
          className="h-full rounded-full transition-[width] duration-300"
          style={{ width: `${clamped}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs text-zinc-400 w-12 text-right tabular-nums">
        {pct.toFixed(1)}%
      </span>
    </div>
  );
}
