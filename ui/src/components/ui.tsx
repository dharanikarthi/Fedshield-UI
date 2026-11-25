import { ReactNode } from "react";

const TIER: Record<string, { c: string; icon: string; label: string }> = {
  "BLOCK-REVIEW": { c: "#d03b3b", icon: "⛔", label: "Block · review" },
  "HOLD-REVIEW": { c: "#ec835a", icon: "▲", label: "Hold · review" },
  MONITOR: { c: "#fab219", icon: "●", label: "Monitor" },
  CLEAR: { c: "#0ca30c", icon: "✓", label: "Clear" },
  BLOCK: { c: "#d03b3b", icon: "⛔", label: "Block" },
  "STEP-UP": { c: "#ec835a", icon: "▲", label: "Step-up auth" },
  ALLOW: { c: "#0ca30c", icon: "✓", label: "Allow" },
};

export function Tier({ tier }: { tier: string }) {
  const t = TIER[tier] ?? { c: "#898781", icon: "•", label: tier };
  return (
    <span className="inline-flex items-center gap-1.5 text-xs whitespace-nowrap text-ink-2">
      <span style={{ color: t.c }} aria-hidden>{t.icon}</span>{t.label}
    </span>
  );
}

export function ScoreBar({ v, thr }: { v: number; thr?: number }) {
  const c = thr != null && v >= thr ? "#d03b3b" : thr != null && v >= thr / 2 ? "#fab219" : "#3987e5";
  return (
    <div className="flex items-center gap-2 min-w-[120px]">
      <div className="relative h-1.5 flex-1 rounded-full bg-raised">
        <div className="absolute left-0 top-0 h-1.5 rounded-full" style={{ width: `${Math.max(2, v * 100)}%`, background: c }} />
        {thr != null && <div className="absolute -top-1 h-3.5 w-px bg-ink-muted" style={{ left: `${thr * 100}%` }} title="decision threshold" />}
      </div>
      <span className="text-xs w-10 text-right">{(v * 100).toFixed(0)}%</span>
    </div>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="card px-4 py-3">
      <div className="label">{label}</div>
      <div className="text-2xl font-semibold mt-1">{value}</div>
      {sub && <div className="text-xs text-ink-muted mt-0.5">{sub}</div>}
    </div>
  );
}

export function Section({ title, sub, children, right }: { title: string; sub?: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="card p-4">
      <div className="flex items-start mb-3">
        <div>
          <h3 className="text-sm font-semibold">{title}</h3>
          {sub && <p className="text-xs text-ink-muted mt-0.5">{sub}</p>}
        </div>
        <div className="ml-auto">{right}</div>
      </div>
      {children}
    </section>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dash?: boolean }[] }) {
  return (
    <div className="flex flex-wrap gap-4 text-xs text-ink-2">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3 h-[3px] rounded" style={{ background: i.color, opacity: i.dash ? 0.6 : 1 }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}

export const tooltipStyle = {
  contentStyle: { background: "#232321", border: "1px solid #383835", borderRadius: 8, fontSize: 12, color: "#f0efec" },
  labelStyle: { color: "#c3c2b7" },
  itemStyle: { color: "#f0efec" },
  cursor: { fill: "rgba(255,255,255,0.04)", stroke: "#383835" },
};
export const axis = { stroke: "#383835", tick: { fill: "#898781", fontSize: 11 }, tickLine: false };
