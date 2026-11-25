import cytoscape from "cytoscape";
import { useEffect, useRef } from "react";

type G = { nodes: any[]; edges: any[] };

export default function FlowGraph({ data }: { data: G | null }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current || !data || !data.nodes.length) return;
    const ids = new Set(data.nodes.map((n) => n.id));
    const cy = cytoscape({
      container: ref.current,
      elements: [
        ...data.nodes.map((n) => ({ data: { ...n, label: n.type === "counterparty" ? "" : n.id } })),
        // drop edges to nodes that are not in the payload: cytoscape throws on them
        ...data.edges.filter((e) => ids.has(e.source) && ids.has(e.target)).map((e, i) => ({ data: { id: `e${i}`, ...e } })),
      ],
      style: [
        { selector: "node", style: { "background-color": "#383835", width: 10, height: 10, label: "data(label)", color: "#c3c2b7", "font-size": 8, "text-valign": "bottom", "text-margin-y": 3 } },
        { selector: 'node[type="subject"]', style: { "background-color": "#3987e5", width: 26, height: 26, "border-width": 2, "border-color": "#f0efec", "font-size": 10, color: "#f0efec" } },
        { selector: 'node[type="counterparty"]', style: { "background-color": "#898781", shape: "diamond", width: 9, height: 9 } },
        { selector: 'node[type="confirmed_mule"]', style: { "background-color": "#d03b3b", width: 16, height: 16, shape: "triangle" } },
        { selector: 'node[type="account"]', style: { "background-color": "#c3c2b7", width: 11, height: 11 } },
        { selector: "edge", style: { width: 1, "line-color": "#383835", "target-arrow-color": "#383835", "target-arrow-shape": "triangle", "arrow-scale": 0.6, "curve-style": "bezier" } },
        { selector: "edge[?shared]", style: { "line-style": "dashed", "line-color": "#d03b3b", "target-arrow-shape": "none", opacity: 0.6 } },
      ] as any,
      layout: { name: "cose", animate: false, nodeRepulsion: () => 9000, idealEdgeLength: () => 60 } as any,
    });
    cy.on("mouseover", "node", (e) => {
      const d = e.target.data();
      const s = d.score != null ? ` · ${(d.score * 100).toFixed(0)}%` : "";
      e.target.style("label", d.type === "counterparty" ? `${d.id} · ${d.degree} accounts` : `${d.id}${s}${d.bank ? ` · ${d.bank}` : ""}`);
    });
    cy.on("mouseout", "node", (e) => { e.target.style("label", e.target.data("label")); });
    return () => cy.destroy();
  }, [data]);
  if (!data) return <div className="h-72 grid place-items-center text-ink-muted text-sm">Loading graph…</div>;
  if (!data.nodes.length) return <div className="h-72 grid place-items-center text-ink-muted text-sm">No counterparties</div>;
  return (
    <div>
      <div ref={ref} className="h-72 rounded-lg bg-page border border-line" />
      <div className="flex flex-wrap gap-4 text-[11px] text-ink-2 mt-2">
        <span><span style={{ color: "#3987e5" }}>●</span> this account</span>
        <span><span style={{ color: "#898781" }}>◆</span> counterparty</span>
        <span><span style={{ color: "#d03b3b" }}>▲</span> confirmed mule (consortium registry)</span>
        <span><span style={{ color: "#c3c2b7" }}>●</span> other account sharing a rare counterparty</span>
      </div>
    </div>
  );
}
