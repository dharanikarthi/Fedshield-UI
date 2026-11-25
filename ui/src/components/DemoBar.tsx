import { useEffect, useRef, useState } from "react";
import { setSession } from "../api";
import { DEMO, resetDemo } from "../demo/mock";

const TOUR = [
  { who: "Bank A · Fraud officer", what: "Starts on its local-only model. Open an account from the watch list: these are real test mules the local model scores below its threshold." },
  { who: "FedShield aggregation server", what: "Run federation. Watch masked histograms flow from each bank, the Byzantine gates vote, and trees join the global model. Try a malicious bank or differential privacy." },
  { who: "Bank A · Fraud officer", what: "The federated model is now deployed. The same watch-list accounts are flagged, with the reasons why. Take a graduated action on one." },
  { who: "RBI supervisory desk", what: "See cross-bank mule clusters through hashed identifiers only, and your action in the audit trail." },
];

/** Slim banner shown only in demo builds (VITE_DEMO=1). */
export default function DemoBar() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);
  if (!DEMO) return null;

  return (
    <div ref={ref} className="relative z-30 border-b border-line bg-[#131312] text-[12px]">
      <div className="max-w-[1600px] mx-auto px-5 h-9 flex items-center gap-3 text-ink-2">
        <span className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-s1/40 bg-s1/10 px-2 py-0.5 text-[10.5px] font-medium uppercase tracking-wider text-ink">
          <span className="h-1.5 w-1.5 rounded-full bg-s1 animate-pulse" aria-hidden />
          Interactive demo
        </span>
        <span className="hidden md:inline truncate">
          Every screen replays outputs recorded from the FedShield prototype on the NFPC benchmark. No backend is connected, and your actions stay in this browser.
        </span>
        <span className="md:hidden truncate text-ink-muted">Best viewed on a laptop or desktop</span>
        <div className="ml-auto flex shrink-0 items-center gap-3">
          <button className="hover:text-ink transition-colors" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            Guided tour <span aria-hidden>{open ? "▴" : "▾"}</span>
          </button>
          <span className="h-3.5 w-px bg-axis" aria-hidden />
          <button className="hover:text-ink transition-colors" onClick={() => { resetDemo(); setSession(null); setOpen(false); }}
            title="Back to local-only models, clear officer actions, sign out">
            Reset demo
          </button>
        </div>
      </div>
      {open && (
        <div className="absolute right-5 top-10 w-[min(440px,calc(100vw-40px))] card p-4 shadow-2xl shadow-black/60">
          <div className="label mb-3">Suggested walkthrough · about 3 minutes</div>
          <ol className="space-y-3">
            {TOUR.map((t, i) => (
              <li key={i} className="flex gap-3">
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-axis text-[11px] text-ink">{i + 1}</span>
                <div>
                  <div className="text-sm font-medium text-ink">{t.who}</div>
                  <div className="text-xs text-ink-2 mt-0.5 leading-relaxed">{t.what}</div>
                </div>
              </li>
            ))}
          </ol>
          <div className="mt-4 border-t border-line pt-3 text-[11px] text-ink-muted">
            Use <span className="text-ink-2">Switch role</span> in the top-right to move between personas. The demo password is <span className="font-mono text-ink-2">fedshield</span>.
          </div>
        </div>
      )}
    </div>
  );
}
