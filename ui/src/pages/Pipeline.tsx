import { useEffect, useMemo, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { get, num, openSocket, post } from "../api";
import { Legend, Section, Stat, axis, tooltipStyle } from "../components/ui";

// categorical slots, dark, validated (5 slots pass all six checks on #1a1a19)
const PAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"];
const STAGES = [
  ["load", "Load adapter"],
  ["setup", "Sketch + Gate 2"],
  ["gate1", "Gate 1 consensus"],
  ["histograms", "Masked histograms"],
  ["sum", "Secure sum"],
  ["tree", "Tree added"],
] as const;

type Ev = Record<string, any>;
type Packet = { id: number; bank: string; masked: number[] };

const W = 980, H = 420, AGG = { x: 560, y: 210 }, MODEL = { x: 880, y: 210 };

export default function Pipeline() {
  const [ds, setDs] = useState<{ registered: string[]; available: Record<string, boolean> }>();
  const [cfg, setCfg] = useState({ dataset: "nfpc", attack: "", dp: "", byzantine: true, n_trees: 40, sample: 30000 });
  const [events, setEvents] = useState<Ev[]>([]);
  const [packets, setPackets] = useState<Packet[]>([]);
  const [stage, setStage] = useState<string>("");
  const [pulse, setPulse] = useState(0);
  const [err, setErr] = useState<string>();
  const pid = useRef(0);

  useEffect(() => { get("/datasets").then(setDs).catch(() => {}); }, []);

  useEffect(() => {
    const ws = openSocket(`/api/pipeline`);
    ws.onmessage = (m) => {
      const e: Ev = JSON.parse(m.data);
      if (e.stage === "load" && e.status === "loading") { setEvents([e]); setPackets([]); setErr(undefined); }
      else setEvents((p) => [...p, e]);
      setStage(e.stage);
      if (e.stage === "histograms") {
        const fresh = Object.keys(e.sent).map((b) => ({ id: pid.current++, bank: b, masked: e.masked_preview?.[b] ?? [] }));
        setPackets((p) => [...p.slice(-30), ...fresh]);
        setTimeout(() => setStage("sum"), 350);
      }
      if (e.stage === "tree") setPulse((x) => x + 1);
      if (e.stage === "error") setErr(e.message);
    };
    return () => ws.close();
  }, []);

  // ------- derived state
  const load = events.find((e) => e.stage === "load" && e.status === "ready");
  const setup = events.find((e) => e.stage === "setup");
  const trees = events.filter((e) => e.stage === "tree");
  const lastG1 = [...events].reverse().find((e) => e.stage === "gate1");
  const done = events.find((e) => e.stage === "done");
  const banks: string[] = load?.banks ?? [];
  const color = (b: string) => PAL[banks.indexOf(b) % PAL.length];
  const pos = (i: number) => ({ x: 130, y: banks.length <= 1 ? H / 2 : 50 + (i * (H - 100)) / (banks.length - 1) });
  const rejectedG2 = (b: string) => !!setup?.gate2?.[b]?.rejected;
  const g1Rejected = useMemo(() => {
    const c: Record<string, number> = {};
    trees.forEach((t) => (t.gate1_rejected || []).forEach((b: string) => (c[b] = (c[b] || 0) + 1)));
    return c;
  }, [trees.length]);
  const lastEval = [...trees].reverse().find((t) => t.eval_auprc != null);
  const curve = trees.filter((t) => t.eval_auprc != null).map((t) => {
    const row: Ev = { round: t.round, global: t.eval_auprc };
    banks.forEach((b) => { if (t[`val_auprc_${b}`] != null) row[b] = t[`val_auprc_${b}`]; });
    return row;
  });
  const eps = lastEval?.epsilon ? Math.max(...(Object.values(lastEval.epsilon) as number[])) : 0;
  const running = events.length > 0 && !done && !err;
  const deployedEv = events.find((e) => e.stage === "deployed");
  const ledger = useMemo(() => {
    const L: Record<string, { bytes: number; msgs: number }> = {};
    events.filter((e) => e.stage === "histograms").forEach((e) =>
      Object.entries(e.sent as Record<string, any>).forEach(([b, v]) => {
        L[b] = L[b] || { bytes: 0, msgs: 0 }; L[b].bytes += v.bytes; L[b].msgs += 1;
      }));
    return L;
  }, [events.length]);
  const [resetMsg, setResetMsg] = useState<string>();
  async function resetDemo() { await post("/demo/reset"); setResetMsg("Banks are back on their local-only models."); setTimeout(() => setResetMsg(undefined), 4000); }

  async function run() {
    setErr(undefined);
    try {
      await post("/pipeline/run", {
        dataset: cfg.dataset, sample: cfg.sample, n_trees: cfg.n_trees, byzantine: cfg.byzantine,
        attack: cfg.attack || null, dp_epsilon: cfg.dp ? Number(cfg.dp) : null, secure_agg: true,
      });
    } catch (e: any) { setErr(String(e.message)); }
  }

  const stageIdx = STAGES.findIndex(([k]) => k === stage);

  return (
    <div className="space-y-4">
      {/* filters: one row above the charts */}
      <div className="card p-3 flex flex-wrap items-end gap-4">
        <Field label="Dataset adapter">
          <select className="sel" value={cfg.dataset} onChange={(e) => setCfg({ ...cfg, dataset: e.target.value })}>
            {(ds?.registered ?? ["nfpc"]).map((d) => (
              <option key={d} value={d} disabled={ds && !ds.available[d]}>{d.toUpperCase()}{ds && !ds.available[d] ? " (not ingested)" : ""}</option>
            ))}
          </select>
        </Field>
        <Field label="Malicious bank">
          <select className="sel" value={cfg.attack} onChange={(e) => setCfg({ ...cfg, attack: e.target.value })}>
            <option value="">None</option><option value="label_flip">Label flipping</option><option value="gradient_scale">Gradient scaling</option>
          </select>
        </Field>
        <Field label="Differential privacy">
          <select className="sel" value={cfg.dp} onChange={(e) => setCfg({ ...cfg, dp: e.target.value })}>
            <option value="">Off</option><option value="64">ε = 64</option><option value="16">ε = 16</option><option value="8">ε = 8</option>
          </select>
        </Field>
        <Field label="Trees">
          <select className="sel" value={cfg.n_trees} onChange={(e) => setCfg({ ...cfg, n_trees: Number(e.target.value) })}>
            {[20, 40, 80].map((n) => <option key={n}>{n}</option>)}
          </select>
        </Field>
        <label className="flex items-center gap-2 text-sm text-ink-2 pb-1.5">
          <input type="checkbox" checked={cfg.byzantine} onChange={(e) => setCfg({ ...cfg, byzantine: e.target.checked })} />
          Byzantine gates
        </label>
        <button className="btn ml-auto bg-s1/15 border-s1 text-ink disabled:opacity-40" disabled={running} onClick={run}>
          {running ? "Training…" : "▶ Run federation"}
        </button>
      </div>

      {err && <div className="card p-3 text-sm border-critical"><span className="text-critical font-medium">⚠ Error</span> <span className="text-ink-2">{err}</span></div>}

      {/* stage strip */}
      <div className="grid grid-cols-6 gap-2">
        {STAGES.map(([k, l], i) => (
          <div key={k} className={`rounded-lg border px-3 py-2 text-xs transition-all duration-300 ${
            i === stageIdx && running ? "border-s1 bg-s1/15 text-ink shadow-[0_0_18px_-4px_#3987e5]" :
            i < stageIdx || done ? "border-line text-ink-2" : "border-line text-ink-muted"}`}>
            <div className="label mb-0.5">Step {i + 1}</div>{l}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-12 gap-4">
        {/* live topology */}
        <div className="card col-span-12 xl:col-span-8 p-2 relative overflow-hidden">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Live federation topology">
            <defs>
              <radialGradient id="glow"><stop offset="0%" stopColor="#3987e5" stopOpacity="0.45" /><stop offset="100%" stopColor="#3987e5" stopOpacity="0" /></radialGradient>
              <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M28 0H0V28" fill="none" stroke="#232321" strokeWidth="1" /></pattern>
            </defs>
            <rect width={W} height={H} fill="url(#grid)" />
            {/* links */}
            {banks.map((b, i) => {
              const p = pos(i);
              const off = rejectedG2(b) || (lastG1?.rejected || []).includes(b);
              return <path key={b} id={`link-${b}`} d={`M${p.x + 70},${p.y} C${p.x + 230},${p.y} ${AGG.x - 200},${AGG.y} ${AGG.x - 60},${AGG.y}`}
                fill="none" stroke={off ? "#d03b3b" : "#383835"} strokeWidth={2} strokeDasharray={off ? "5 6" : undefined} />;
            })}
            <path id="link-model" d={`M${AGG.x + 60},${AGG.y} L${MODEL.x - 64},${MODEL.y}`} stroke="#383835" strokeWidth={2} />

            {/* packets in flight: masked histogram vectors */}
            {packets.map((k) => (
              <g key={k.id}>
                <circle r={6} fill={color(k.bank)} stroke="#1a1a19" strokeWidth={2}>
                  <animateMotion dur="0.55s" fill="freeze" rotate="auto"><mpath href={`#link-${k.bank}`} /></animateMotion>
                  <animate attributeName="opacity" values="1;1;0" keyTimes="0;0.85;1" dur="0.6s" fill="freeze" />
                </circle>
              </g>
            ))}
            {pulse > 0 && (
              <circle key={`m${pulse}`} r={7} fill="#f0efec">
                <animateMotion dur="0.35s" fill="freeze"><mpath href="#link-model" /></animateMotion>
                <animate attributeName="opacity" values="1;0" dur="0.4s" fill="freeze" />
              </circle>
            )}

            {/* bank nodes */}
            {banks.map((b, i) => {
              const p = pos(i), info = setup?.banks?.[b], g2 = setup?.gate2?.[b];
              const cos = lastG1?.cos?.[b];
              const rej2 = rejectedG2(b), rej1 = (lastG1?.rejected || []).includes(b);
              return (
                <g key={b} transform={`translate(${p.x - 70},${p.y - 30})`}>
                  <rect width={140} height={60} rx={10} fill="#232321" stroke={rej2 || rej1 ? "#d03b3b" : color(b)} strokeWidth={2} />
                  <rect width={6} height={60} rx={3} fill={color(b)} />
                  <text x={16} y={20} fill="#f0efec" fontSize={13} fontWeight={600}>{b}</text>
                  <text x={16} y={36} fill="#898781" fontSize={10.5}>{info ? `${info.rows.toLocaleString()} rows · ${info.positives} pos` : "…"}</text>
                  <text x={16} y={51} fill={rej2 || rej1 ? "#ec835a" : "#c3c2b7"} fontSize={10.5}>
                    {rej2 ? "✕ excluded · Gate 2" : rej1 ? "✕ dropped this tree · Gate 1" :
                      g2 ? `peer AUPRC ${num(g2.peer_auprc, 2)}${cos != null ? ` · cos ${num(cos, 2)}` : ""}` : "awaiting probe"}
                  </text>
                  {info?.attack && <text x={136} y={14} textAnchor="end" fontSize={10} fill="#ec835a">⚠ {info.attack.replace("_", " ")}</text>}
                </g>
              );
            })}

            {/* aggregator */}
            <circle cx={AGG.x} cy={AGG.y} r={95} fill="url(#glow)" opacity={stage === "sum" || stage === "histograms" ? 1 : 0.25} style={{ transition: "opacity .3s" }} />
            <g transform={`translate(${AGG.x - 60},${AGG.y - 48})`}>
              <rect width={120} height={96} rx={14} fill="#1a1a19" stroke="#3987e5" strokeWidth={2} />
              <text x={60} y={24} textAnchor="middle" fill="#f0efec" fontSize={12.5} fontWeight={600}>Aggregator</text>
              <text x={60} y={40} textAnchor="middle" fill="#898781" fontSize={10}>sums masks · sees no rows</text>
              {(packets.slice(-3).map((k, j) => (
                <text key={k.id} x={60} y={58 + j * 12} textAnchor="middle" fill="#c3c2b7" fontSize={9.5} fontFamily="ui-monospace,monospace">
                  {k.masked.slice(0, 4).map((v) => v.toString(16).padStart(5, "0")).join(" ")}
                </text>
              )))}
            </g>

            {/* global model */}
            <g transform={`translate(${MODEL.x - 64},${MODEL.y - 70})`}>
              <rect width={128} height={140} rx={14} fill="#1a1a19" stroke="#f0efec" strokeOpacity={0.5} strokeWidth={1.5} />
              <text x={64} y={22} textAnchor="middle" fill="#f0efec" fontSize={12.5} fontWeight={600}>Global model</text>
              <text x={64} y={52} textAnchor="middle" fill="#f0efec" fontSize={26} fontWeight={600}>{trees.length}</text>
              <text x={64} y={67} textAnchor="middle" fill="#898781" fontSize={10}>of {setup?.n_trees ?? "—"} trees</text>
              {/* forest glyph: one tick per tree */}
              {trees.slice(-40).map((t, j) => (
                <rect key={t.round} x={10 + (j % 20) * 5.4} y={84 + Math.floor(j / 20) * 16} width={3.4} height={12} rx={1.5}
                  fill={(t.gate1_rejected || []).length ? "#ec835a" : "#3987e5"} />
              ))}
              <text x={64} y={130} textAnchor="middle" fill="#c3c2b7" fontSize={10}>{trees[trees.length - 1]?.top_feature ?? ""}</text>
            </g>
          </svg>
          <div className="absolute left-3 bottom-2 text-[11px] text-ink-muted">
            Dots = secure-aggregation-masked gradient histograms. The aggregator only ever sees their sum.
          </div>
        </div>

        {/* headline numbers */}
        <div className="col-span-12 xl:col-span-4 grid grid-cols-2 gap-3 content-start">
          <Stat label="Dataset" value={(load?.dataset ?? cfg.dataset).toUpperCase()} sub={load ? `${load.rows.toLocaleString()} ${load.unit}s · ${load.features} features` : "—"} />
          <Stat label="Global AUPRC" value={num(lastEval?.eval_auprc, 3)} sub={lastEval ? `held-out test, round ${lastEval.round}` : "evaluated every 5 trees"} />
          <Stat label="Privacy spent" value={setup?.dp_epsilon ? `ε ${eps.toFixed(1)}` : "no DP"} sub={setup?.dp_epsilon ? `budget ε ${setup.dp_epsilon}` : "secure aggregation only"} />
          <Stat label="Time" value={done ? `${done.seconds.toFixed(1)}s` : running ? "running" : "—"} sub={`${events.length} pipeline events`} />
          <div className="card col-span-2 p-3">
            <div className="label mb-2">Privacy ledger: what crossed the wire</div>
            <table className="w-full text-sm">
              <thead><tr className="text-ink-muted text-[11px] uppercase tracking-wider"><th className="text-left font-medium">Bank</th><th className="text-right font-medium">Masked histograms</th><th className="text-right font-medium">Rows shared</th></tr></thead>
              <tbody>
                {banks.map((b) => (
                  <tr key={b}><td className="py-0.5"><span className="inline-block w-2.5 h-2.5 rounded-sm mr-2" style={{ background: color(b) }} />{b}</td>
                    <td className="text-right text-ink-2">{ledger[b] ? `${ledger[b].msgs} msgs · ${(ledger[b].bytes / 1048576).toFixed(1)} MB` : "—"}</td>
                    <td className="text-right font-semibold">0</td></tr>
                ))}
              </tbody>
            </table>
            {banks.length === 0 && <div className="text-sm text-ink-muted">Run a federation to fill the ledger.</div>}
            <div className="text-[11px] text-ink-muted mt-2">Labels, account IDs and transactions never leave a bank. The aggregator only ever unmasks the sum across banks.</div>
          </div>
          <div className="card col-span-2 p-3 flex items-center gap-3">
            <div className="text-sm flex-1">
              {deployedEv ? <><span className="text-good">✓</span> Global model deployed to member banks. Their consoles now score with it.</> : "When an NFPC run finishes, the global model is deployed to the bank consoles."}
              {resetMsg && <div className="text-xs text-ink-muted mt-1">{resetMsg}</div>}
            </div>
            <button className="btn text-xs" onClick={resetDemo}>Reset banks to local</button>
          </div>
          <div className="card col-span-2 p-3">
            <div className="label mb-2">Byzantine gate decisions</div>
            {banks.length === 0 && <div className="text-sm text-ink-muted">Run a federation to see each bank's checks.</div>}
            {banks.map((b) => (
              <div key={b} className="flex items-center gap-2 py-1 text-sm">
                <span className="w-2.5 h-2.5 rounded-sm" style={{ background: color(b) }} />
                <span className="w-16">{b}</span>
                <span className={rejectedG2(b) ? "text-critical" : "text-ink-2"}>
                  {rejectedG2(b) ? "✕ Gate 2: excluded at start" : "✓ Gate 2"}
                </span>
                <span className="ml-auto text-ink-muted text-xs">{g1Rejected[b] ? `Gate 1 dropped ${g1Rejected[b]}×` : ""}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="col-span-12 lg:col-span-7">
          <Section title="Learning curve" sub="AUPRC as trees are added: global test set and each bank's validation set">
            {banks.length > 1 && <Legend items={[{ label: "Global (test)", color: "#f0efec" }, ...banks.map((b) => ({ label: b, color: color(b), dash: true }))]} />}
            <div className="h-56">
              <ResponsiveContainer>
                <LineChart data={curve} margin={{ top: 8, right: 16, bottom: 0, left: -8 }}>
                  <CartesianGrid stroke="#2c2c2a" vertical={false} />
                  <XAxis dataKey="round" {...axis} />
                  <YAxis domain={[0, 1]} {...axis} />
                  <Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(3)} labelFormatter={(r) => `Tree ${r}`} />
                  {banks.map((b) => <Line key={b} dataKey={b} stroke={color(b)} strokeWidth={2} strokeDasharray="4 4" dot={false} isAnimationActive={false} />)}
                  <Line dataKey="global" stroke="#f0efec" strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Section>
        </div>

        <div className="col-span-12 lg:col-span-5">
          <Section title="Event log" sub="what crossed the wire, newest first">
            <div className="h-56 overflow-auto font-mono text-[11.5px] leading-5 pr-1">
              {[...events].reverse().slice(0, 120).map((e) => <LogLine key={e.seq} e={e} />)}
            </div>
          </Section>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="flex flex-col gap-1"><span className="label">{label}</span>{children}</label>;
}

function LogLine({ e }: { e: Ev }) {
  const msg = (() => {
    switch (e.stage) {
      case "load": return e.status === "loading" ? `loading adapter "${e.dataset}"…` : `adapter ready: ${e.rows.toLocaleString()} rows, ${e.features} features, banks ${e.banks.join(", ")}`;
      case "setup": return `quantile sketch merged · Gate 2 accepted ${e.accepted.join(", ")}`;
      case "gate1": return `tree ${e.round}: consensus check${e.rejected.length ? ` → dropped ${e.rejected.join(", ")}` : " → all agree"}`;
      case "histograms": {
        const kb = Object.values(e.sent as Record<string, any>).reduce((s: number, v: any) => s + v.bytes, 0) / 1024;
        return `tree ${e.round}: ${Object.keys(e.sent).length} masked histograms (${kb.toFixed(0)} KB) → secure sum`;
      }
      case "tree": return `tree ${e.round} added · ${e.leaves} leaves · top split ${e.top_feature ?? "—"}${e.eval_auprc != null ? ` · AUPRC ${e.eval_auprc.toFixed(3)}` : ""}`;
      case "done": return `federation complete in ${e.seconds.toFixed(1)}s`;
      case "deployed": return `global model deployed to member banks`;
      case "error": return `error: ${e.message}`;
      default: return e.stage;
    }
  })();
  const c = e.stage === "error" ? "text-critical" : e.stage === "tree" ? "text-ink" : e.stage === "gate1" && e.rejected?.length ? "text-serious" : "text-ink-2";
  return <div className={c}><span className="text-ink-muted">{String(e.seq).padStart(4, "0")}</span> {msg}</div>;
}
