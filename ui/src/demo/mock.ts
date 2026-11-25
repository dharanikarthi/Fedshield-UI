// Demo mode: the UI runs with no backend. Every API call and WebSocket is answered from
// fixtures recorded from the real FedShield API (ui/public/demo, built by
// scripts/record_demo_fixtures.py). Officer actions and the "deployed" switch are kept in
// this browser only, so each visitor gets their own walkthrough.
import { ApiError, session } from "../api";

export const DEMO = import.meta.env.VITE_DEMO === "1";

const BASE = `${import.meta.env.BASE_URL}demo`;
const cache = new Map<string, Promise<any>>();
function fx<T = any>(file: string): Promise<T> {
  if (!cache.has(file)) {
    cache.set(file, fetch(`${BASE}/${file}`).then((r) => {
      if (!r.ok) throw new ApiError(404, "Not Found");
      return r.json();
    }).catch((e) => { cache.delete(file); throw e; }));
  }
  return cache.get(file)!;
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ per-browser demo state
type Audit = { account_id: string; bank: string; action: string; label: string; officer: string; note: string; ts: string; feedback_label: number | null };
type State = { deployed: boolean; audit: Audit[] };
const KEY = "fedshield.demo.v1";
function read(): State {
  try { return { deployed: false, audit: [], ...JSON.parse(localStorage.getItem(KEY) || "{}") }; }
  catch { return { deployed: false, audit: [] }; }
}
let st: State = read();
function save() {
  try { localStorage.setItem(KEY, JSON.stringify(st)); } catch { /* storage unavailable: state stays in memory */ }
}
const view = () => (st.deployed ? "federated" : "local");

export function resetDemo() {
  st = { deployed: false, audit: [] };
  save();
  LIVE.events = []; LIVE.cfg = null; LIVE.running = false; LIVE.gen++;
}

// ------------------------------------------------------------------ live federation replay
const LIVE = { events: [] as any[], running: false, cfg: null as any, gen: 0, subs: new Set<(e: any) => void>() };
function emit(e: any) {
  e.seq = LIVE.events.length;
  e.t = Date.now() / 1000;
  LIVE.events.push(e);
  LIVE.subs.forEach((f) => f(e));
}
const runKey = (c: any) => `${c.attack || "none"}_${c.dp_epsilon ?? "off"}_${c.byzantine ? "byz" : "nobyz"}_${c.n_trees}`;

async function replay(cfg: any) {
  const gen = ++LIVE.gen;
  LIVE.events = []; LIVE.cfg = cfg; LIVE.running = true;
  try {
    let rec: any[];
    try { rec = await fx(`pipeline/${runKey(cfg)}.json`); }
    catch { rec = await fx("pipeline/none_off_byz_40.json"); }
    let paced = 0;
    for (const r of rec) {
      if (gen !== LIVE.gen) return;
      const e = { ...r };
      // the server sleeps after each histogram and tree so the browser can show each hop;
      // its timers include that pacing, so ours do too
      if (e.stage === "tree" && e.elapsed != null) e.elapsed += paced;
      if (e.stage === "done" && e.seconds != null) e.seconds += paced;
      if (e.stage === "deployed") { st.deployed = true; save(); }
      emit(e);
      const hop = e.stage === "histograms" || e.stage === "tree";
      if (hop) paced += 0.15;
      await sleep(e.stage === "load" && e.status === "loading" ? 1200 : e.stage === "load" ? 500 : e.stage === "setup" ? 450 : hop ? 150 : 25);
    }
  } finally {
    if (gen === LIVE.gen) LIVE.running = false;
  }
}

// ------------------------------------------------------------------ REST
function need(...roles: string[]) {
  const s = session();
  if (!s) throw new ApiError(401, "login required");
  if (!roles.includes(s.role)) throw new ApiError(403, `role '${s.role}' cannot access this resource`);
  return s;
}
async function own(id: string) {
  const owners = await fx<Record<string, string>>("owners.json");
  const bank = owners[id];
  if (!bank) throw new ApiError(404, "unknown account");
  const s = session();
  if (!s) throw new ApiError(401, "login required");
  if (s.role !== "bank") throw new ApiError(403, "only bank officers can open account-level data");
  if (s.bank !== bank) throw new ApiError(403, `${s.bank} cannot access ${bank} accounts`);
  return s;
}

export async function mockCall(path: string, init: RequestInit = {}): Promise<any> {
  await sleep(80 + Math.random() * 140);
  const method = (init.method || "GET").toUpperCase();
  const body = init.body ? JSON.parse(String(init.body)) : {};
  const [p, qs] = path.split("?");
  const q = new URLSearchParams(qs);
  let m: RegExpMatchArray | null;

  if (method === "POST") {
    if (p === "/login") {
      const ps = await fx<any[]>("personas.json");
      const u = ps.find((x) => x.user === body.username);
      if (!u || body.password !== "fedshield") throw new ApiError(401, "invalid username or password");
      return { token: `demo.${u.user}`, ...u };
    }
    if (p === "/demo/reset") { need("admin"); st.deployed = false; save(); return { deployed: false }; }
    if (p === "/demo/deploy") { need("admin"); st.deployed = true; save(); return { deployed: true }; }
    if (p === "/pipeline/run") {
      need("admin");
      if (LIVE.running) throw new ApiError(409, "a run is already in progress");
      const cfg = { dataset: body.dataset ?? "nfpc", sample: body.sample ?? 30000, n_trees: body.n_trees ?? 40, attack: body.attack ?? null,
        dp_epsilon: body.dp_epsilon ?? null, byzantine: body.byzantine ?? true, secure_agg: body.secure_agg ?? true };
      replay(cfg);
      return { started: true, config: cfg };
    }
    if ((m = p.match(/^\/accounts\/([^/]+)\/action$/))) {
      const s = await own(m[1]);
      const d = await fx(`account/${view()}/${m[1]}.json`);
      if (!(body.action in d.actions)) throw new ApiError(400, `action must be one of ${Object.keys(d.actions)}`);
      const rec: Audit = { account_id: m[1], bank: s.bank!, action: body.action, label: d.actions[body.action], officer: s.user,
        note: body.note ?? "", ts: new Date().toISOString(),
        feedback_label: body.action === "dismiss" ? 0 : ["hold", "freeze"].includes(body.action) ? 1 : null };
      st.audit.push(rec); save();
      return rec;
    }
    throw new ApiError(404, "Not Found");
  }

  if (p === "/health") return { ok: true, mode: "demo", time: new Date().toISOString() };
  if (p === "/personas") return fx("personas.json");
  if (p === "/datasets") return fx("datasets.json");
  if (p === "/federation") return fx("federation.json");
  if (p === "/experiments") return fx("experiments.json");
  if (p === "/me") {
    const s = need("bank", "admin", "rbi");
    return { ...(await fx(`me/${s.user}.json`)), deployed: st.deployed };
  }
  if (p === "/alerts") {
    const s = need("bank");
    const rows = await fx<any[]>(`alerts/${s.bank}_${view()}.json`);
    const acted: Record<string, Audit> = {};
    st.audit.forEach((a) => (acted[a.account_id] = a));
    return rows.slice(0, Number(q.get("limit") ?? 200)).map((r) => ({ ...r, status: acted[r.account_id]?.action ?? null }));
  }
  if (p === "/demo/cases") {
    const s = need("bank", "admin", "rbi");
    if (s.role === "bank") return fx(`cases/${s.bank}_${view()}.json`);
    const d = await fx(`cases/rbi_${view()}.json`);
    return s.role === "rbi" ? d : { ...d, cases: [] };
  }
  if ((m = p.match(/^\/accounts\/([^/]+)\/graph$/))) { await own(m[1]); return fx(`graph/${m[1]}.json`); }
  if ((m = p.match(/^\/accounts\/([^/]+)$/))) {
    await own(m[1]);
    const d = await fx(`account/${view()}/${m[1]}.json`);
    return { ...d, history: st.audit.filter((a) => a.account_id === m![1]) };
  }
  if (p === "/audit") {
    const s = need("bank", "admin", "rbi");
    const rows = [...st.audit].reverse().slice(0, 500);
    if (s.role === "bank") return rows.filter((r) => r.bank === s.bank);
    if (s.role === "rbi") {
      const masks = await fx<Record<string, string>>("masks.json");
      return rows.map((r) => ({ ...r, account_id: masks[r.account_id] ?? "H-unknown", note: "" }));
    }
    throw new ApiError(403, "the aggregator holds no account-level audit data");
  }
  if (p === "/rbi/overview") {
    need("rbi", "admin");
    return { ...(await fx(`rbi_overview_${view()}.json`)), deployed: st.deployed };
  }
  if (p === "/pipeline/state") return { running: LIVE.running, config: LIVE.cfg, events: LIVE.events.length };
  throw new ApiError(404, "Not Found");
}

// ------------------------------------------------------------------ WebSockets
type Msg = { data: string };
export class MockSocket {
  onmessage: ((m: Msg) => void) | null = null;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readyState = 0;
  private stop: () => void = () => {};

  constructor(path: string) {
    setTimeout(() => this.open(path), 0);
  }

  private send1(e: any) {
    if (this.readyState === 1) this.onmessage?.({ data: JSON.stringify(e) });
  }

  private open(path: string) {
    if (this.readyState === 3) return;
    this.readyState = 1;
    this.onopen?.();
    const [p, qs] = path.split("?");
    if (p === "/api/pipeline") {
      LIVE.events.forEach((e) => this.send1(e));          // replay from the start, then follow
      const f = (e: any) => this.send1(e);
      LIVE.subs.add(f);
      this.stop = () => LIVE.subs.delete(f);
    } else if (p === "/api/stream") {
      const rate = Math.max(0.5, Number(new URLSearchParams(qs).get("rate") ?? 20));
      let i = 0, alive = true, timer: number | undefined;
      fx<any[]>("stream.json").then((evs) => {
        const tick = () => {
          if (!alive) return;
          this.send1(evs[i++ % evs.length]);
          timer = window.setTimeout(tick, 1000 / rate);
        };
        tick();
      });
      this.stop = () => { alive = false; clearTimeout(timer); };
    }
  }

  send() { /* the demo sockets are receive-only, like the real ones */ }
  close() {
    this.readyState = 3;
    this.stop();
    this.onclose?.();
  }
}
