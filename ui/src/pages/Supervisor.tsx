import cytoscape from "cytoscape";
import { useEffect, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BANK_COLOR, get, pct } from "../api";
import { Legend, Section, Stat, axis, tooltipStyle } from "../components/ui";

const SRC_COLOR: Record<string, string> = { EFRM: "#3987e5", AML: "#d95926", I4C: "#199e70" };
const STATE_STYLE: Record<string, string> = { confirmed: "border-good text-ink", emerging: "border-warn text-ink", dormant: "border-axis text-ink-2", retired: "border-line text-ink-muted" };

export default function Supervisor() {
  const [d, setD] = useState<any>(null);
  const [audit, setAudit] = useState<any[]>([]);
  const [sel, setSel] = useState(0);
  const [err, setErr] = useState<string>();
  useEffect(() => {
    get("/rbi/overview").then(setD).catch((e) => setErr(e.message));
    get("/audit").then(setAudit).catch(() => {});
  }, []);
  if (err) return <div className="card p-6 text-critical">⚠ {err}</div>;
  if (!d) return <div className="card p-10 text-center text-ink-muted">Building cross-bank view (first load takes a few seconds)…</div>;

  const cl = d.clusters ?? [];
  const c = cl[sel];
  const banks = Object.keys(d.per_bank);
  const fe = d.federation_effect ?? {};
  const multi = cl.filter((x: any) => x.banks.length >= 3).length;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        <Stat label="Cross-bank clusters" value={cl.length} sub={`${multi} span all ${banks.length} banks`} />
        <Stat label="Open alerts" value={banks.reduce((s, b) => s + d.per_bank[b].alerts, 0).toLocaleString()} sub="federated model, held-out accounts" />
        <Stat label="Extra mules caught by federation" value={fe.caught_by_fed_missed_by_local != null ? `+${fe.caught_by_fed_missed_by_local - fe.caught_by_local_missed_by_fed}` : "—"}
          sub={fe.caught_by_fed_missed_by_local != null ? `${fe.caught_by_fed_missed_by_local} gained, ${fe.caught_by_local_missed_by_fed} lost, of ${fe.test_mules} test mules` : ""} />
        <Stat label="Model precision / recall" value={d.model ? `${pct(d.model.precision, 0)} / ${pct(d.model.recall, 0)}` : "—"} sub={d.model ? `AUPRC ${d.model.auprc.toFixed(3)}` : ""} />
        <Stat label="Data access" value="Hashed only" sub="no balances, transactions or identities" />
      </div>

      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12 lg:col-span-4">
          <Section title="Cross-bank mule clusters" sub="Flagged and confirmed accounts linked by rare shared counterparties, across 2 or more banks">
            <div className="max-h-[430px] overflow-y-auto -mx-1">
              {cl.map((x: any, i: number) => (
                <button key={x.cluster} onClick={() => setSel(i)}
                  className={`w-full text-left px-2 py-2 rounded-md mb-1 ${i === sel ? "bg-raised" : "hover:bg-raised/60"}`}>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs text-ink-2">#{x.cluster}</span>
                    <span className="ml-auto text-xs text-ink-muted">{x.size} accounts</span>
                  </div>
                  <div className="flex items-center gap-1.5 mt-1">
                    {x.banks.map((b: string) => <span key={b} className="text-[10px] px-1.5 rounded border" style={{ borderColor: BANK_COLOR[b] }}>{b}</span>)}
                    <span className="ml-auto text-[11px] text-ink-2">{x.confirmed} confirmed</span>
                  </div>
                </button>
              ))}
              {!cl.length && <div className="text-sm text-ink-muted p-2">No cross-bank clusters.</div>}
            </div>
          </Section>
        </div>
        <div className="col-span-12 lg:col-span-8">
          <Section title={c ? `Cluster #${c.cluster}` : "Cluster"} sub="Nodes are keyed-hash pseudonyms. The supervisor sees the links, never the identities."
            right={<Legend items={banks.map((b) => ({ label: b, color: BANK_COLOR[b] }))} />}>
            {c && <ClusterGraph c={c} />}
            {c && (
              <div className="grid grid-cols-3 gap-3 mt-3 text-sm">
                <div><div className="label">Shared counterparties</div>{c.shared_counterparties}</div>
                <div><div className="label">Highest risk score</div>{(c.max_score * 100).toFixed(0)}%</div>
                <div><div className="label">Evidence sources (simulated)</div>
                  <div className="flex gap-3">{Object.entries(c.evidence).map(([k, v]: any) => <span key={k}><span style={{ color: SRC_COLOR[k] }}>■</span> {k} {v}</span>)}</div></div>
              </div>
            )}
          </Section>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12 lg:col-span-5">
          <Section title="Alerts by bank" sub="Held-out accounts flagged by the federated model">
            <div className="h-48">
              <ResponsiveContainer>
                <BarChart data={banks.map((b) => ({ bank: b, alerts: d.per_bank[b].alerts }))} margin={{ top: 8, right: 8, left: -12 }}>
                  <CartesianGrid stroke="#2c2c2a" vertical={false} />
                  <XAxis dataKey="bank" {...axis} />
                  <YAxis {...axis} />
                  <Tooltip {...tooltipStyle} />
                  <Bar dataKey="alerts" radius={[4, 4, 0, 0]} maxBarSize={36} fill="#3987e5"
                    shape={(p: any) => <rect x={p.x} y={p.y} width={p.width} height={p.height} rx={4} fill={BANK_COLOR[p.payload.bank]} />} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Section>
        </div>
        <div className="col-span-12 lg:col-span-7">
          <Section title="Pattern library" sub="Typology states across the consortium: emerging, confirmed, dormant or retired">
            <div className="flex flex-wrap gap-1.5">
              {(d.patterns ?? []).map((p: any) => (
                <span key={p.pattern} className={`text-xs px-2 py-1 rounded-md border ${STATE_STYLE[p.state] ?? "border-axis"}`}
                  title={`${p.cases} cases · ${p.banks} banks · recall ${pct(p.recall, 0)} · signature: ${(p.signature ?? []).join(", ")}`}>
                  {p.state === "confirmed" ? "✓" : p.state === "emerging" ? "▲" : "○"} {p.pattern} <span className="text-ink-muted">{p.cases}</span>
                </span>
              ))}
            </div>
          </Section>
        </div>
      </div>

      <Section title="Audit trail" sub="Every officer action across member banks. Account IDs are pseudonymised and case notes withheld.">
        {audit.length ? (
          <table className="data w-full">
            <thead><tr><th>Time</th><th>Bank</th><th>Account</th><th>Action</th><th>Officer</th></tr></thead>
            <tbody>{audit.slice(0, 50).map((a, i) => (
              <tr key={i}><td className="text-ink-2">{a.ts.slice(0, 19).replace("T", " ")}</td><td>{a.bank ?? "—"}</td>
                <td className="font-mono text-xs">{a.account_id}</td><td>{a.label}</td><td className="text-ink-2">{a.officer}</td></tr>))}
            </tbody>
          </table>
        ) : <div className="text-sm text-ink-muted">No officer actions yet. Actions taken in a bank console appear here.</div>}
      </Section>
    </div>
  );
}

function ClusterGraph({ c }: { c: any }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const ids = new Set(c.members.map((m: any) => m.id));
    const cy = cytoscape({
      container: ref.current,
      elements: [
        ...c.members.map((m: any) => ({ data: { ...m, label: "", color: BANK_COLOR[m.bank] ?? "#898781" } })),
        ...c.links.filter((l: any) => ids.has(l.a) && ids.has(l.b)).map((l: any, i: number) => ({ data: { id: `l${i}`, source: l.a, target: l.b } })),
      ],
      style: [
        { selector: "node", style: { "background-color": "data(color)", width: 14, height: 14, "border-width": 2, "border-color": "#1a1a19", color: "#c3c2b7", "font-size": 9, "text-valign": "bottom", "text-margin-y": 3 } },
        { selector: 'node[status="confirmed"]', style: { shape: "triangle", width: 18, height: 18 } },
        { selector: "edge", style: { width: 1.2, "line-color": "#383835", "curve-style": "haystack" } },
      ] as any,
      layout: { name: "cose", animate: false, nodeRepulsion: () => 6000, idealEdgeLength: () => 50 } as any,
    });
    cy.on("mouseover", "node", (e) => {
      const m = e.target.data();
      e.target.style("label", `${m.id} · ${m.bank} · ${(m.score * 100).toFixed(0)}% · ${m.status}${m.sources.length ? ` · ${m.sources.join("+")}` : ""}`);
    });
    cy.on("mouseout", "node", (e) => e.target.style("label", ""));
    return () => cy.destroy();
  }, [c]);
  return (
    <div>
      <div ref={ref} className="h-80 rounded-lg bg-page border border-line" />
      <div className="text-[11px] text-ink-2 mt-2">▲ confirmed mule · ● flagged by the federated model · colour = bank · hover a node for details</div>
    </div>
  );
}
