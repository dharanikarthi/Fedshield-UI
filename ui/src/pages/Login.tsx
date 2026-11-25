import { useEffect, useState } from "react";
import { BANK_COLOR, get, post, setSession } from "../api";

type Persona = { user: string; role: string; bank: string | null; name: string; scope: { can: string[]; cannot: string[] } };

const accent = (p: Persona) => (p.role === "bank" ? BANK_COLOR[p.bank!] : p.role === "admin" ? "#f0efec" : "#d55181");
const ICON: Record<string, string> = { bank: "🏦", admin: "🖥️", rbi: "🏛️" };

export default function Login() {
  const [ps, setPs] = useState<Persona[]>([]);
  const [sel, setSel] = useState<Persona | null>(null);
  const [pw, setPw] = useState("fedshield");
  const [err, setErr] = useState<string>();
  useEffect(() => { get<Persona[]>("/personas").then(setPs).catch((e) => setErr(String(e.message))); }, []);

  async function go(p: Persona) {
    setErr(undefined);
    try { setSession(await post("/login", { username: p.user, password: pw })); }
    catch (e: any) { setErr(e.message); }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-10 relative overflow-hidden">
      <div className="absolute inset-0 opacity-40 pointer-events-none"
        style={{ backgroundImage: "radial-gradient(circle at 20% 20%, #3987e522, transparent 40%), radial-gradient(circle at 80% 70%, #d5518118, transparent 40%)" }} />
      <div className="relative text-center mb-8">
        <div className="flex items-center justify-center gap-2 mb-2">
          <svg width="30" height="30" viewBox="0 0 24 24" aria-hidden><path d="M12 2l8 3v6c0 5-3.4 9.4-8 11-4.6-1.6-8-6-8-11V5l8-3z" fill="none" stroke="#3987e5" strokeWidth="2" /><path d="M8.5 12l2.5 2.5 4.5-5" stroke="#f0efec" strokeWidth="2" fill="none" /></svg>
          <span className="text-2xl font-semibold tracking-tight">FedShield</span>
        </div>
        <div className="text-ink-2 text-sm">Federated mule-account intelligence. Banks learn together, and no customer data leaves any bank.</div>
      </div>

      <div className="relative grid gap-3 w-full max-w-5xl" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))" }}>
        {ps.map((p) => (
          <button key={p.user} onClick={() => setSel(p)}
            className={`card text-left p-4 transition-all hover:-translate-y-0.5 ${sel?.user === p.user ? "ring-2" : ""}`}
            style={{ borderTop: `3px solid ${accent(p)}`, ...(sel?.user === p.user ? { boxShadow: `0 0 0 2px ${accent(p)}` } : {}) }}>
            <div className="text-2xl mb-2" aria-hidden>{ICON[p.role]}</div>
            <div className="font-semibold text-sm">{p.name}</div>
            <div className="text-xs text-ink-muted mt-0.5 font-mono">{p.user}</div>
          </button>
        ))}
      </div>

      {sel && (
        <div className="relative card mt-5 p-5 w-full max-w-5xl grid md:grid-cols-3 gap-5">
          <div>
            <div className="label mb-2">Sign in as</div>
            <div className="font-semibold">{sel.name}</div>
            <input type="password" value={pw} onChange={(e) => setPw(e.target.value)} onKeyDown={(e) => e.key === "Enter" && go(sel)}
              className="mt-3 w-full bg-page border border-line rounded-md px-2 py-1.5 text-sm outline-none focus:border-ink-muted" aria-label="Password" />
            <button className="btn mt-3 w-full" style={{ borderColor: accent(sel) }} onClick={() => go(sel)}>Enter →</button>
            <div className="text-[11px] text-ink-muted mt-2">Demo password: fedshield</div>
            {err && <div className="text-xs text-critical mt-2">⚠ {err}</div>}
          </div>
          <div>
            <div className="label mb-2">✓ Can see</div>
            <ul className="text-sm text-ink-2 space-y-1">{sel.scope.can.map((c) => <li key={c}>{c}</li>)}</ul>
          </div>
          <div>
            <div className="label mb-2">✕ Cannot see. The server enforces this.</div>
            <ul className="text-sm text-ink-2 space-y-1">{sel.scope.cannot.map((c) => <li key={c}>{c}</li>)}</ul>
          </div>
        </div>
      )}
    </div>
  );
}
