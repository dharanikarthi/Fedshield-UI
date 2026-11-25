import { useEffect, useRef, useState } from "react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Section, Stat, Tier, axis, tooltipStyle } from "../components/ui";
import { openSocket } from "../api";

type Ev = { step: number; type: string; amount: number; sender: string; receiver: string; bank: string; score: number; decision: string; reasons: string[]; is_fraud: number; latency_ms: number; receiver_fanin_24h: number };

export default function Stream() {
  const [events, setEvents] = useState<Ev[]>([]);
  const [running, setRunning] = useState(true);
  const [rate, setRate] = useState(10);
  const lat = useRef<number[]>([]);
  const tot = useRef({ n: 0, block: 0, step: 0, caught: 0, fraud: 0 });
  useEffect(() => {
    if (!running) return;
    const ws = openSocket(`/api/stream?rate=${rate}`);
    ws.onmessage = (m) => {
      const e: Ev = JSON.parse(m.data);
      lat.current.push(e.latency_ms);
      if (lat.current.length > 2000) lat.current.shift();
      const t = tot.current;
      t.n++; if (e.decision === "BLOCK") t.block++; if (e.decision === "STEP-UP") t.step++;
      if (e.is_fraud) { t.fraud++; if (e.decision !== "ALLOW") t.caught++; }
      setEvents((p) => [e, ...p].slice(0, 60));
    };
    return () => ws.close();
  }, [running, rate]);
  const sorted = [...lat.current].sort((a, b) => a - b);
  const p = (q: number) => (sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] : 0);
  const bins = Array.from({ length: 12 }, (_, i) => ({ ms: `${i * 2}-${i * 2 + 2}`, n: 0 }));
  sorted.forEach((v) => { bins[Math.min(11, Math.floor(v / 2))].n++; });
  const t = tot.current;
  return (
    <div className="space-y-4">
      <div className="card p-3 flex flex-wrap items-center gap-3 text-sm">
        <button className="btn" onClick={() => setRunning((r) => !r)}>{running ? "Pause" : "Resume"}</button>
        <span className="label">Events/sec</span>
        {[2, 10, 30].map((r) => <button key={r} className={`btn ${rate === r ? "bg-raised" : ""}`} onClick={() => setRate(r)}>{r}</button>)}
        <span className="text-xs text-ink-muted ml-auto max-w-xl">
          Replays a PaySim-derived payment stream (enriched to about 1 fraud per 25 events). Each event is scored on the bank's own system using the federated transaction model and an online feature store. PaySim accuracy is not reported as a FedShield result.
        </span>
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        <Stat label="Events scored" value={t.n.toLocaleString()} />
        <Stat label="Blocked" value={t.block} sub="score ≥ 0.9" />
        <Stat label="Step-up auth" value={t.step} sub="score ≥ threshold" />
        <Stat label="Fraud intercepted" value={t.fraud ? `${Math.round((t.caught / t.fraud) * 100)}%` : "—"} sub={`${t.caught}/${t.fraud} in this session`} />
        <Stat label="Latency p50 / p99" value={`${p(0.5).toFixed(1)} / ${p(0.99).toFixed(1)} ms`} sub="feature lookup + scoring" />
      </div>
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12 lg:col-span-8">
          <Section title="Live decisions">
            <div className="max-h-[560px] overflow-y-auto">
              <table className="data w-full">
                <thead className="sticky top-0 bg-surface"><tr><th>Hour</th><th>Bank</th><th>Payment</th><th className="text-right">Amount</th><th>Receiver fan-in 24h</th><th>Score</th><th>Decision</th><th>Top signals</th></tr></thead>
                <tbody>
                  {events.map((e, i) => (
                    <tr key={i} className={e.decision !== "ALLOW" ? "bg-critical/5" : ""}>
                      <td className="text-ink-muted">{e.step}</td>
                      <td className="whitespace-nowrap">{e.bank}</td>
                      <td>{e.type}<div className="text-[11px] text-ink-muted">…{e.sender} → …{e.receiver}</div></td>
                      <td className="text-right">{e.amount.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                      <td>{e.receiver_fanin_24h}</td>
                      <td>{(e.score * 100).toFixed(1)}%</td>
                      <td><Tier tier={e.decision} />{e.is_fraud ? <span className="ml-2 text-[10px] text-ink-muted">(label: fraud)</span> : null}</td>
                      <td className="text-xs text-ink-2">{e.reasons.join(", ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
        <div className="col-span-12 lg:col-span-4">
          <Section title="Scoring latency" sub="Last 2,000 events, ms per event">
            <div className="h-56">
              <ResponsiveContainer>
                <BarChart data={bins} margin={{ left: 0, right: 8 }}>
                  <XAxis dataKey="ms" {...axis} interval={1} />
                  <YAxis {...axis} width={36} />
                  <Tooltip {...tooltipStyle} />
                  <Bar dataKey="n" name="events" fill="#3987e5" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-ink-muted mt-2">UPI's end-to-end budget is a few hundred ms; risk scoring has to fit in a small slice of it.</p>
          </Section>
        </div>
      </div>
    </div>
  );
}
