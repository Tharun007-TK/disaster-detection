import { SEVERITY_COLOR, worstClass } from "@/lib/severity";

interface Props {
  damagePct: Record<string, number>;
}

export default function SeverityBadge({ damagePct }: Props) {
  const cls = worstClass(damagePct);
  const color = SEVERITY_COLOR[cls];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium tabular-nums"
      style={{
        color,
        backgroundColor: `${color}1f`, // ~12% alpha tint
        boxShadow: `inset 0 0 0 1px ${color}40`,
      }}
    >
      <span
        aria-hidden="true"
        className="w-1.5 h-1.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      {cls}
    </span>
  );
}
