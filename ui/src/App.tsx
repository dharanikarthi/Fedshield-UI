import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { ROLE_ACCENT, Session, get, session, setSession } from "./api";
import DemoBar from "./components/DemoBar";
import Console from "./pages/Console";
import Federation from "./pages/Federation";
import Login from "./pages/Login";
import Pipeline from "./pages/Pipeline";
import Results from "./pages/Results";
import Stream from "./pages/Stream";
import Supervisor from "./pages/Supervisor";

const NAV: Record<string, { to: string; label: string; el: JSX.Element }[]> = {
  bank: [
    { to: "/", label: "Officer console", el: <Console /> },
    { to: "/stream", label: "Live stream", el: <Stream /> },
    { to: "/results", label: "Evidence", el: <Results /> },
  ],
  admin: [
    { to: "/", label: "Live pipeline", el: <Pipeline /> },
    { to: "/federation", label: "Federation", el: <Federation /> },
    { to: "/results", label: "Evidence", el: <Results /> },
  ],
  rbi: [
    { to: "/", label: "Supervisory dashboard", el: <Supervisor /> },
    { to: "/results", label: "Evidence", el: <Results /> },
  ],
};

export default function App() {
  const [s, setS] = useState<Session | null>(session());
  const [me, setMe] = useState<any>(null);
  const [probe, setProbe] = useState<string>();
  useEffect(() => {
    const h = () => setS(session());
    window.addEventListener("fedshield-session", h);
    return () => window.removeEventListener("fedshield-session", h);
  }, []);
  useEffect(() => { if (s) get("/me").then(setMe).catch(() => {}); else setMe(null); }, [s?.token]);
  if (!s) return <><DemoBar /><Login /></>;

  const accent = ROLE_ACCENT(s);
  const nav = NAV[s.role];

  async function runProbe() {
    const t = me?.probe_target;
    if (!t) return;
    try { await get(`/accounts/${t.account_id}`); setProbe(`⚠ Unexpected: ${t.bank} account was returned`); }
    catch (e: any) { setProbe(`${e.status} Forbidden: "${e.message}". ${s!.bank} asked for ${t.account_id} (${t.bank}) and the server refused.`); }
    setTimeout(() => setProbe(undefined), 7000);
  }

  return (
    <div className="min-h-screen flex flex-col">
      <DemoBar />
      <div className="h-[3px]" style={{ background: accent }} />
      <header className="border-b border-line bg-surface/80 backdrop-blur sticky top-0 z-20">
        <div className="max-w-[1600px] mx-auto px-5 h-14 flex items-center gap-6">
          <div className="flex items-center gap-2">
            <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden><path d="M12 2l8 3v6c0 5-3.4 9.4-8 11-4.6-1.6-8-6-8-11V5l8-3z" fill="none" stroke="#3987e5" strokeWidth="2" /><path d="M8.5 12l2.5 2.5 4.5-5" stroke="#f0efec" strokeWidth="2" fill="none" /></svg>
            <span className="font-semibold tracking-tight">FedShield</span>
          </div>
          <nav className="flex gap-1">
            {nav.map((n) => (
              <NavLink key={n.to} to={n.to} end
                className={({ isActive }) => `px-3 py-1.5 rounded-md text-sm ${isActive ? "bg-raised text-ink" : "text-ink-2 hover:text-ink"}`}>
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-3">
            {s.role === "bank" && (
              <span className={`text-xs px-2 py-1 rounded-md border ${me?.deployed ? "border-good text-ink" : "border-axis text-ink-2"}`}>
                {me?.deployed ? "● Federated global model" : "○ Local model only"}
              </span>
            )}
            {s.role === "bank" && <button className="btn text-xs" onClick={runProbe} title="Ask the server for another bank's account">Privacy probe</button>}
            {s.role === "rbi" && <span className="text-xs px-2 py-1 rounded-md border border-axis text-ink-2">Read-only</span>}
            <div className="text-right leading-tight">
              <div className="text-sm font-medium" style={{ color: accent }}>{s.name}</div>
              <div className="text-[11px] text-ink-muted font-mono">{s.user}</div>
            </div>
            <button className="btn text-xs" onClick={() => setSession(null)}>Switch role</button>
          </div>
        </div>
      </header>
      {probe && (
        <div className="max-w-[1600px] w-full mx-auto px-5 pt-3">
          <div className="card p-3 text-sm border-good"><span className="text-good font-medium">✓ Scope enforced.</span> <span className="text-ink-2">{probe}</span></div>
        </div>
      )}
      <main className="flex-1 max-w-[1600px] w-full mx-auto px-5 py-5">
        <Routes>
          {nav.map((n) => <Route key={n.to} path={n.to} element={n.el} />)}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="max-w-[1600px] w-full mx-auto px-5 pb-4 text-[11px] text-ink-muted">
        Signed in as {s.name}. Can see: {s.scope.can.join(" · ")}. Cannot see: {s.scope.cannot.join(" · ")}.
      </footer>
    </div>
  );
}
