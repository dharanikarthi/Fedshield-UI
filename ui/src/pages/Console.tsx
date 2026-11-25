import { useEffect, useMemo, useState } from "react";
import { Area, Bar, ComposedChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { get, inr, pct, post, session } from "../api";
import FlowGraph from "../components/FlowGraph";
import { axis, Legend, ScoreBar, Section, Tier, tooltipStyle } from "../components/ui";

type Alert = { account_id: string; score: number; threshold: number; model: string; tier: string; bank: string; state: string; window: [string, string] | null; status?: string; ground_truth: number };

export default function Console() {
  const bank = session()?.bank ?? "";
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [cases, setCases] = useState<any[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const refresh = () => get<Alert[]>(`/alerts?limit=300`).then(setAlerts);
  useEffect(() => {
    get<Alert[]>(`/alerts?limit=300`).then((a) => { setAlerts(a); setSel(a[0]?.account_id ?? null); });
    get(`/demo/cases`).then((d) => setCases(d.cases ?? [])).catch(() => {});
  }, [bank]);
  const thr = alerts[0]?.threshold ?? 0.5;
  const model = alerts[0]?.model;
  const shown = alerts.filter((a) => !q || a.account_id.toLowerCase().includes(q.toLowerCase()));
  const counts = useMemo(() => ({
    review: alerts.filter((a) => a.score >= thr).length,
    monitor: alerts.filter((a) => a.score < thr).length,
    actioned: alerts.filter((a) => a.status).length,
  }), [alerts, thr]);

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 xl:col-span-4 space-y-3">
        <div className="card p-3">
          <div className="flex items-center gap-2">
            <span className="label">Scoring model</span>
            <span className={`ml-auto text-xs px-2 py-0.5 rounded border ${model === "federated" ? "border-good" : "border-axis text-ink-2"}`}>
              {model === "federated" ? "Federated global model" : `${bank} local model`}
            </span>
          </div>
          <div className="text-xs text-ink-muted mt-1.5">
            {model === "federated" ? "Trained across the consortium: patterns from every bank, none of their data."
              : "Trained on this bank's accounts only. It has never seen fraud patterns from other banks."}
          </div>
          {cases.length > 0 && (
            <div className="mt-3">
              <div className="label mb-1.5">Watch list · test mules this bank's local model scores below its threshold</div>
              <div className="flex flex-wrap gap-1.5">
                {cases.slice(0, 6).map((c) => (
                  <button key={c.account_id} onClick={() => setSel(c.account_id)}
                    className={`btn text-xs py-1 ${sel === c.account_id ? "bg-raised border-ink-muted" : ""}`}>{c.account_id}</button>
                ))}
              </div>
            </div>
          )}
        </div>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="card py-2"><div className="label">To review</div><div className="text-xl font-semibold">{counts.review}</div></div>
          <div className="card py-2"><div className="label">Monitor</div><div className="text-xl font-semibold">{counts.monitor}</div></div>
          <div className="card py-2"><div className="label">Actioned</div><div className="text-xl font-semibold">{counts.actioned}</div></div>
        </div>
        <div className="card overflow-hidden">
          <div className="p-3 border-b border-line flex items-center gap-2">
            <h3 className="text-sm font-semibold">Alert queue</h3>
            <span className="text-xs text-ink-muted">held-out test accounts</span>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search account…"
              className="ml-auto bg-page border border-line rounded-md px-2 py-1 text-sm w-36 outline-none focus:border-ink-muted" />
          </div>
          <div className="max-h-[calc(100vh-290px)] overflow-y-auto">
            <table className="data w-full">
              <thead className="sticky top-0 bg-surface"><tr><th>Account</th><th>Risk</th><th>Tier</th></tr></thead>
              <tbody>
                {shown.map((a) => (
                  <tr key={a.account_id} onClick={() => setSel(a.account_id)}
                    className={`cursor-pointer ${sel === a.account_id ? "bg-raised" : "hover:bg-raised/60"}`}>
                    <td>
                      <div className="font-medium">{a.account_id}</div>
                      <div className="text-[11px] text-ink-muted">{a.state}{a.status ? ` · ${a.status}` : ""}</div>
                    </td>
                    <td><ScoreBar v={a.score} thr={thr} /></td>
                    <td><Tier tier={a.tier} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div className="col-span-12 xl:col-span-8">
        {sel ? <AccountView id={sel} onAction={refresh} /> : <div className="card p-10 text-center text-ink-muted">No alerts</div>}
      </div>
    </div>
  );
}

function AccountView({ id, onAction }: { id: string; onAction: () => void }) {
  const [d, setD] = useState<any>(null);
  const [g, setG] = useState<any>(null);
  const [note, setNote] = useState("");
  const load = () => get(`/accounts/${id}`).then(setD);
  useEffect(() => { setD(null); setG(null); load(); get(`/accounts/${id}/graph`).then(setG).catch(() => setG({ nodes: [], edges: [] })); }, [id]);
  if (!d) return <div className="card p-10 text-center text-ink-muted">Loading…</div>;
  const act = async (a: string) => { await post(`/accounts/${id}/action`, { action: a, note }); setNote(""); load(); onAction(); };
  const win = d.window?.[0] ? [d.window[0].slice(0, 7), d.window[1].slice(0, 7)] : null;
  const reg = d.registry ?? {};
  return (
    <div className="space-y-4">
      {d.model === "federated" && d.local_score != null && (
        <div className={`card p-4 flex flex-wrap items-center gap-6 ${d.local_score < d.local_threshold && d.score >= d.threshold ? "border-critical" : ""}`}>
          <div className="label w-full -mb-3">Federation effect on this account</div>
          <div><div className="text-xs text-ink-muted">{d.bank} local model</div>
            <div className="text-2xl font-semibold text-ink-2">{(d.local_score * 100).toFixed(1)}%</div>
            <div className="text-[11px] text-ink-muted">{d.local_score >= d.local_threshold ? "flagged" : "not flagged"} · threshold {(d.local_threshold * 100).toFixed(0)}%</div></div>
          <div className="text-2xl text-ink-muted" aria-hidden>→</div>
          <div><div className="text-xs text-ink-muted">Federated global model</div>
            <div className="text-2xl font-semibold">{(d.score * 100).toFixed(1)}%</div>
            <div className="text-[11px] text-ink-muted">{d.score >= d.threshold ? "flagged" : "not flagged"} · threshold {(d.threshold * 100).toFixed(0)}%</div></div>
          {d.local_score < d.local_threshold && d.score >= d.threshold && (
            <div className="text-sm text-ink-2 max-w-md">The federated model learned this pattern from other banks' confirmed cases. None of their data left their bank.</div>
          )}
        </div>
      )}
      <div className="card p-4 flex flex-wrap items-center gap-6">
        <div>
          <div className="label">Account</div>
          <div className="text-xl font-semibold">{d.account_id}</div>
          <div className="text-xs text-ink-muted">{d.bank} · {d.state}</div>
        </div>
        <div><div className="label">Mule risk · {d.model === "federated" ? "federated" : "local"} model</div><div className="text-3xl font-semibold">{(d.score * 100).toFixed(1)}%</div></div>
        <div><div className="label">Decision</div><div className="mt-1"><Tier tier={d.tier} /></div></div>
        <div>
          <div className="label">Suspicious window</div>
          <div className="text-sm mt-1">{d.window?.[0] ? `${d.window[0]} → ${d.window[1]}` : "—"}</div>
        </div>
        <div className="ml-auto text-right">
          <div className="label">Consortium registry</div>
          <div className="text-sm mt-1">{reg.reg_cp_flagged ? `${reg.reg_cp_flagged} linked counterparties` : "no hits"}</div>
          {reg.reg_ip_shared_mule > 0 && <div className="text-xs text-ink-2"><span style={{ color: "#ec835a" }}>▲</span> device shared with a confirmed mule</div>}
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12 lg:col-span-6">
          <Section title="Why this alert" sub={`Top drivers of this score (tree-path feature attribution, ${d.model === "federated" ? "federated" : "local"} model)`}>
            <ol className="space-y-2.5">
              {d.reasons.map((r: any, i: number) => (
                <li key={r.feature} className="flex gap-3">
                  <span className="text-ink-muted text-xs w-4 pt-0.5">{i + 1}</span>
                  <div className="flex-1">
                    <div className="text-sm">{r.text}</div>
                    <div className="mt-1 h-1 rounded-full bg-raised">
                      <div className="h-1 rounded-full bg-critical" style={{ width: `${Math.min(100, (r.impact / d.reasons[0].impact) * 100)}%` }} />
                    </div>
                  </div>
                </li>
              ))}
              {!d.reasons.length && <li className="text-sm text-ink-muted">No risk-raising factors.</li>}
              {d.mitigating && <li className="text-sm text-ink-2 pl-7">↓ {d.mitigating.text}</li>}
            </ol>
          </Section>
        </div>
        <div className="col-span-12 lg:col-span-6">
          <Section title="Money flow" sub="Top counterparties by value, plus other accounts sharing rare counterparties">
            <FlowGraph data={g} />
          </Section>
        </div>
      </div>

      <Section title="Monthly activity" sub={win ? "Shaded band: predicted suspicious window" : undefined}
        right={<Legend items={[{ label: "Credits", color: "#3987e5" }, { label: "Debits", color: "#d95926" }]} />}>
        <div className="h-52">
          <ResponsiveContainer>
            <ComposedChart data={d.timeline} margin={{ left: 8, right: 8, top: 4 }}>
              <XAxis dataKey="month" {...axis} minTickGap={30} />
              <YAxis {...axis} tickFormatter={(v) => inr(v)} width={70} />
              <Tooltip {...tooltipStyle} formatter={(v: number) => inr(v)} />
              {win && <ReferenceArea x1={win[0]} x2={win[1]} fill="#d03b3b" fillOpacity={0.08} stroke="#d03b3b" strokeOpacity={0.3} />}
              <Bar dataKey="credit" name="Credits" fill="#3987e5" radius={[4, 4, 0, 0]} maxBarSize={10} />
              <Area dataKey="debit" name="Debits" stroke="#d95926" strokeWidth={2} fill="#d95926" fillOpacity={0.08} type="monotone" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12 lg:col-span-7">
          <Section title="Graduated response" sub="Proportionate action instead of an immediate freeze. Every action goes to the audit log and back into training as a label.">
            <div className="flex flex-wrap gap-2">
              {Object.entries(d.actions).map(([k, v]: any) => (
                <button key={k} className="btn" onClick={() => act(k)}>{v}</button>
              ))}
            </div>
            <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Case note (optional)"
              className="mt-3 w-full bg-page border border-line rounded-md px-2 py-1.5 text-sm outline-none focus:border-ink-muted" />
            {d.history.length > 0 && (
              <ul className="mt-3 text-xs text-ink-2 space-y-1">
                {d.history.map((h: any) => <li key={h.ts}>{h.ts.slice(0, 19).replace("T", " ")} · {h.officer} · {h.label}{h.note ? ` — ${h.note}` : ""}</li>)}
              </ul>
            )}
          </Section>
        </div>
        <div className="col-span-12 lg:col-span-5">
          <Section title="Evaluation ground truth" sub="Prototype only; a real officer would not see this">
            <dl className="grid grid-cols-2 gap-y-1.5 text-sm">
              <dt className="text-ink-muted">Labelled mule</dt><dd>{d.label.is_mule ? "yes" : "no"}</dd>
              <dt className="text-ink-muted">Flag date</dt><dd>{d.label.flag_date ?? "—"}</dd>
              <dt className="text-ink-muted">Bank's alert reason</dt><dd>{d.label.alert_reason ?? "—"}</dd>
              <dt className="text-ink-muted">Label confidence</dt><dd>{pct(d.label.label_confidence, 0)}</dd>
              <dt className="text-ink-muted">Red-herring flags</dt><dd className="text-xs">{d.label.red_herring_flags ?? "none"}</dd>
            </dl>
          </Section>
        </div>
      </div>
    </div>
  );
}
