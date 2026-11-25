import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, LabelList, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { SERIES, get, num, pct } from "../api";
import { Legend, Section, Stat, axis, tooltipStyle } from "../components/ui";

function perBank(block: any) {
  if (!block?.rows) return [];
  const fed = block.rows.find((r: any) => r.model === "FedShield");
  return Object.keys(fed?.per_bank ?? {}).map((b) => {
    const loc = block.rows.find((r: any) => r.model === `${b} local-only`);
    return { bank: b, local: loc?.per_bank?.[b]?.auprc, fed: fed.per_bank[b].auprc };
  });
}

function Comparison({ title, sub, block }: { title: string; sub: string; block: any }) {
  const data = perBank(block);
  const cen = block?.summary?.central_overall_auprc;
  return (
    <Section title={title} sub={sub} right={<Legend items={[{ label: "Local only", color: SERIES[1] }, { label: "FedShield", color: SERIES[0] }, { label: "Centralised ceiling", color: "#898781", dash: true }]} />}>
      <div className="h-64">
        <ResponsiveContainer>
          <BarChart data={data} margin={{ left: 0, right: 12, top: 18 }} barGap={2}>
            <CartesianGrid stroke="#2c2c2a" vertical={false} />
            <XAxis dataKey="bank" {...axis} />
            <YAxis {...axis} width={40} tickFormatter={(v) => v.toFixed(2)} />
            <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(4)} />
            {cen != null && <ReferenceLine y={cen} stroke="#898781" strokeDasharray="4 4" />}
            <Bar dataKey="local" name="Local only" fill={SERIES[1]} radius={[4, 4, 0, 0]} maxBarSize={36}>
              <LabelList dataKey="local" position="top" formatter={(v: number) => v?.toFixed(3)} fill="#c3c2b7" fontSize={11} />
            </Bar>
            <Bar dataKey="fed" name="FedShield" fill={SERIES[0]} radius={[4, 4, 0, 0]} maxBarSize={36}>
              <LabelList dataKey="fed" position="top" formatter={(v: number) => v?.toFixed(3)} fill="#c3c2b7" fontSize={11} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-ink-muted mt-1">AUC-PR on each bank's own held-out accounts. Dashed line: pooled-data model, overall test set.</p>
    </Section>
  );
}

function RowsTable({ rows }: { rows: any[] }) {
  return (
    <table className="data w-full">
      <thead><tr><th>Model</th><th>AUC</th><th>AUC-PR</th><th>Precision</th><th>Recall</th><th>F1</th><th>Flag rate</th></tr></thead>
      <tbody>
        {rows.map((r) => {
          const o = r.model.includes("local-only") ? Object.values(r.per_bank)[0] as any : r.overall;
          return (
            <tr key={r.model}>
              <td>{r.model}{r.model.includes("local-only") ? <span className="text-ink-muted text-xs"> (own test)</span> : ""}</td>
              <td>{num(o.auc)}</td><td className="font-medium">{num(o.auprc)}</td><td>{pct(o.precision)}</td>
              <td>{pct(o.recall)}</td><td>{num(o.f1)}</td><td>{pct(o.flag_rate, 2)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function Results() {
  const [r, setR] = useState<any>(null);
  useEffect(() => { get("/experiments").then(setR); }, []);
  if (!r) return <div className="text-ink-muted">Loading…</div>;
  const nf = r.nfpc_full ?? {};
  const na = r.nfpc_asof ?? {};
  const aml = r["aml_HI-Small"] ?? {};
  const fm = nf.final_model?.test;
  const regGain = (x: any) => x?.summary ? x.summary.fed_overall_auprc : null;
  // AMLworld and as-of results are optional artefacts: only show their panels when present
  const hasAml = !!aml.experiments?.base;
  const hasAsof = !!na.main;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Stat label="NFPC final model AUC-PR" value={num(fm?.auprc)} sub={`AUC ${num(fm?.auc)} · F1 ${num(fm?.f1)} · ${nf.n_features ?? "—"} features`} />
        <Stat label="NFPC precision / recall" value={`${pct(fm?.precision, 0)} / ${pct(fm?.recall, 0)}`} sub={`flag rate ${pct(fm?.flag_rate, 2)}`} />
        {hasAml ? (<>
          <Stat label="AML: federated vs mean local" value={`${num(aml.experiments?.base?.summary?.fed_own_auprc_mean)} vs ${num(aml.experiments?.base?.summary?.local_own_auprc_mean)}`} sub="AUC-PR on each bank's own accounts" />
          <Stat label="AML + consortium registry" value={num(regGain(aml.experiments?.registry_fed))} sub={`local-scope registry: ${num(regGain(aml.experiments?.registry_local))}`} />
        </>) : (<>
          <Stat label="NFPC: federated vs mean local" value={`${num(nf.main?.summary?.fed_own_auprc_mean)} vs ${num(nf.main?.summary?.local_own_auprc_mean)}`} sub="AUC-PR on each bank's own accounts" />
          <Stat label="NFPC + consortium registry" value={num(regGain(nf.registry?.fed))} sub={`centralised ceiling: ${num(nf.registry?.fed?.summary?.central_overall_auprc)}`} />
        </>)}
      </div>

      {hasAml ? (<>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Comparison title="NFPC · regional banks" sub="RBIH NFPC Phase-2, 3 banks by state, 5-year window" block={nf.main} />
          <Comparison title="IBM AMLworld · 5 institutions" sub="HI-Small; 98% of laundering transactions cross banks" block={aml.experiments?.base} />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Comparison title="NFPC + consortium suspect registry" sub="Adds hashed links to mules confirmed at any consortium bank" block={nf.registry?.fed} />
          <Comparison title="AMLworld + consortium suspect registry" sub="Counterparty and same-owner (KYC entity) links across banks" block={aml.experiments?.registry_fed} />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Section title="NFPC: all models" sub="Test set; threshold picked on validation (max F1)">{nf.main && <RowsTable rows={nf.main.rows} />}</Section>
          <Section title="AMLworld: all models" sub="Test set; threshold picked on validation (max F1)"><RowsTable rows={aml.experiments.base.rows} /></Section>
        </div>
      </>) : (<>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Comparison title="NFPC · regional banks" sub="RBIH NFPC Phase-2, 3 banks by state, 5-year window" block={nf.main} />
          <Comparison title="NFPC + consortium suspect registry" sub="Adds hashed links to mules confirmed at any consortium bank" block={nf.registry?.fed} />
        </div>
        <Section title="NFPC: all models" sub="Test set; threshold picked on validation (max F1)">{nf.main && <RowsTable rows={nf.main.rows} />}</Section>
      </>)}

      <div className={`grid grid-cols-1 gap-4 ${hasAsof ? "lg:grid-cols-3" : "lg:grid-cols-2"}`}>
        <Section title="Label shield ablation" sub="FedShield trained on raw vs confidence-weighted labels">
          <table className="data w-full"><tbody>
            <tr><td>Raw labels</td><td className="font-medium">{num(nf.label_shield?.without?.overall?.auprc)}</td></tr>
            <tr><td>Confidence-weighted</td><td className="font-medium">{num(nf.label_shield?.with?.overall?.auprc)}</td></tr>
          </tbody></table>
        </Section>
        {hasAsof && (
          <Section title="Point-in-time evaluation" sub="Features use only transactions before each account's cutoff (no post-flag behaviour)">
            <table className="data w-full"><tbody>
              <tr><td>Full window (benchmark-comparable)</td><td className="font-medium">{num(nf.main?.summary?.fed_overall_auprc)}</td></tr>
              <tr><td>As-of cutoff (deployment-honest)</td><td className="font-medium">{num(na.main?.summary?.fed_overall_auprc)}</td></tr>
            </tbody></table>
          </Section>
        )}
        <Section title="Seed stability" sub="Re-split + retrain, 3 seeds (AUC-PR, own-bank test)">
          <table className="data w-full">
            <thead><tr><th>Seed</th><th>Local mean</th><th>FedShield</th></tr></thead>
            <tbody>{(nf.seeds ?? []).map((s: any) => <tr key={s.seed}><td>{s.seed}</td><td>{num(s.local_own_mean)}</td><td className="font-medium">{num(s.fed_own_mean)}</td></tr>)}</tbody>
          </table>
        </Section>
      </div>

      <Section title="What is and isn't shown here" sub="Scope, stated plainly">
        <ul className="text-sm text-ink-2 list-disc pl-5 space-y-1">
          <li>The banks are partitions of one dataset (NFPC by state; AMLworld by bank-ID hash). Real cross-bank gains depend on how different real institutions are.</li>
          <li>EFRM / AML / I4C evidence feeds are simulated from the labels; the fusion logic runs, but its inputs aren't live bank feeds.</li>
          <li>Registry features are built only from training-split confirmations, and an account's own label is never used for its own features.</li>
          <li>PaySim drives the live stream only. Its near-perfect separability makes accuracy on it meaningless as evidence.</li>
        </ul>
      </Section>
    </div>
  );
}
