import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BANK_COLOR, get, num, pct } from "../api";
import { axis, Legend, Section, Stat, tooltipStyle } from "../components/ui";

export default function Federation() {
  const [f, setF] = useState<any>(null);
  useEffect(() => { get("/federation").then(setF); }, []);
  const rounds = useMemo(() => (f?.rounds ?? []).filter((r: any) => Object.keys(r).some((k) => k.startsWith("val_auprc_"))), [f]);
  if (!f) return <div className="text-ink-muted">Loading…</div>;
  const banks = Object.keys(f.banks ?? {});
  const g2 = f.gate2 ?? {};
  const cfg = f.config ?? {};
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        <Stat label="Consortium members" value={banks.length} sub="regional banks, simulated from NFPC" />
        <Stat label="Federation rounds" value={(f.rounds ?? []).length} sub="1 round = 1 boosted tree" />
        <Stat label="Raw rows shared" value="0" sub="only masked G/H histograms" />
        <Stat label="Secure aggregation" value={cfg.secure_agg ? "On" : "Off"} sub="pairwise masks cancel in the sum" />
        <Stat label="Byzantine gates" value={cfg.byzantine ? "On" : "Off"} sub={`accepted: ${(cfg.accepted ?? []).join(", ")}`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {banks.map((b) => (
          <div key={b} className="card p-4">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: BANK_COLOR[b] }} />
              <h3 className="font-semibold">{b}</h3>
              <span className="ml-auto text-xs text-ink-2">
                {g2[b]?.rejected ? <><span style={{ color: "#d03b3b" }}>✕</span> rejected</> : <><span style={{ color: "#0ca30c" }}>✓</span> accepted</>}
              </span>
            </div>
            <dl className="grid grid-cols-2 gap-y-1 text-sm mt-3">
              <dt className="text-ink-muted">Accounts</dt><dd>{f.banks[b].accounts.toLocaleString()}</dd>
              <dt className="text-ink-muted">Confirmed mules</dt><dd>{f.banks[b].mules.toLocaleString()}</dd>
              <dt className="text-ink-muted">Peer AUC-PR (Gate 2)</dt><dd>{num(g2[b]?.peer_auprc)}</dd>
              <dt className="text-ink-muted">Contribution weight</dt><dd>{num(f.weights?.[b], 2)}</dd>
            </dl>
            <div className="text-[11px] text-ink-muted mt-2 leading-relaxed">{f.banks[b].states.join(", ")}</div>
          </div>
        ))}
      </div>

      <Section title="Validation AUC-PR by round" sub="Each bank scores the shared model on its own validation accounts and reports only the metric"
        right={<Legend items={banks.map((b) => ({ label: b, color: BANK_COLOR[b] }))} />}>
        <div className="h-60">
          <ResponsiveContainer>
            <LineChart data={rounds} margin={{ left: 0, right: 16, top: 6 }}>
              <CartesianGrid stroke="#2c2c2a" vertical={false} />
              <XAxis dataKey="round" {...axis} />
              <YAxis {...axis} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} width={40} />
              <Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(4)} />
              {banks.map((b) => <Line key={b} dataKey={`val_auprc_${b}`} name={b} stroke={BANK_COLOR[b]} strokeWidth={2} dot={{ r: 4 }} />)}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Section title="Poisoning attack: Bank-C turns malicious" sub="Test AUC-PR of the shared model with the two-stage gates off vs on">
          <table className="data w-full">
            <thead><tr><th>Attack</th><th>Gates</th><th>AUC-PR</th><th>Precision</th><th>Recall</th><th>Who was excluded</th></tr></thead>
            <tbody>
              {Object.entries(f.byzantine ?? {}).map(([k, v]: any) => (
                <tr key={k}>
                  <td>{k.startsWith("label_flip") ? "Label flipping" : "Gradient scaling ×−10"}</td>
                  <td>{k.endsWith("on") ? "on" : "off"}</td>
                  <td className="font-medium">{num(v.overall?.auprc)}</td>
                  <td>{pct(v.overall?.precision)}</td>
                  <td>{pct(v.overall?.recall)}</td>
                  <td className="text-xs text-ink-2">
                    {k.endsWith("on") ? `${Object.entries(v.gate2 ?? {}).filter(([, g]: any) => g.rejected).map(([n]) => n).join(", ") || "none at Gate 2"}; ${v.gate1_rejections} tree-level exclusions` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
        <Section title="Differential privacy cost" sub="Record-level (ε, δ=1e-5) Gaussian mechanism on each bank's histograms, zCDP accounting">
          <table className="data w-full">
            <thead><tr><th>Target ε</th><th>Spent ε</th><th>Noise σ</th><th>AUC-PR</th><th>AUC</th></tr></thead>
            <tbody>
              {(f.dp ?? []).map((r: any) => (
                <tr key={String(r.epsilon_target)}>
                  <td>{r.epsilon_target ?? "∞ (no DP)"}</td>
                  <td>{r.epsilon_target ? num(r.epsilon_spent, 2) : "—"}</td>
                  <td>{r.sigma ? num(r.sigma, 1) : "—"}</td>
                  <td className="font-medium">{num(r.auprc)}</td>
                  <td>{num(r.auc)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-ink-muted mt-2">The DP runs use a smaller model (60 trees, depth 4) to keep the number of noisy queries low.</p>
        </Section>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Section title="Inclusiveness: wrongful-hold rate by group" sub="False-positive rate at the live threshold. RBI's concern is genuine customers held by mistake.">
          {f.fairness ? Object.entries(f.fairness).map(([dim, groups]: any) => (
            <div key={dim} className="mb-3">
              <div className="label mb-1">{dim.replace("_", " ")} · max/min FPR ratio {num(groups.fpr_ratio_max_min, 2)}</div>
              <div className="space-y-1">
                {Object.entries(groups).filter(([k]) => k !== "fpr_ratio_max_min").map(([g, v]: any) => (
                  <div key={g} className="flex items-center gap-3 text-sm">
                    <span className="w-24 text-ink-2">{g}</span>
                    <div className="flex-1 h-1.5 rounded-full bg-raised">
                      <div className="h-1.5 rounded-full bg-s1" style={{ width: `${Math.min(100, v.fpr * 100 * 10)}%` }} />
                    </div>
                    <span className="w-28 text-right text-xs">FPR {pct(v.fpr, 2)} · n={v.n.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>
          )) : <div className="text-ink-muted text-sm">Not computed</div>}
        </Section>
        <Section title="Red-herring shield" sub="Mule labels down-weighted because their evidence is implausible (label confidence < 1)">
          <table className="data w-full">
            <thead><tr><th>Rule</th><th>Mule labels</th></tr></thead>
            <tbody>
              {Object.entries(f.label_shield ?? {}).sort((a: any, b: any) => b[1] - a[1]).map(([k, v]: any) => (
                <tr key={k}><td>{k.replace(/_/g, " ")}</td><td>{v.toLocaleString()}</td></tr>
              ))}
            </tbody>
          </table>
        </Section>
      </div>

      <Section title="Living pattern library" sub="Typologies tracked across banks; state moves emerging → confirmed → dormant → retired as evidence arrives each round">
        <table className="data w-full">
          <thead><tr><th>Pattern</th><th>State</th><th>Banks with cases</th><th>Confirmed cases</th><th>Test recall</th><th>Evidence sources</th><th>Signature</th></tr></thead>
          <tbody>
            {(f.patterns ?? []).map((p: any) => (
              <tr key={p.pattern}>
                <td className="font-medium">{p.pattern}</td>
                <td><span className="text-xs">{p.state}</span></td>
                <td>{p.banks}</td>
                <td>{p.cases}</td>
                <td>{pct(p.recall, 0)}</td>
                <td className="text-xs text-ink-2">{Object.entries(p.sources ?? {}).map(([s, n]) => `${s} ${n}`).join(" · ")}</td>
                <td className="text-xs text-ink-2">{(p.signature ?? []).join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>
    </div>
  );
}
