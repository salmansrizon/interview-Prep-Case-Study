"""A single server-rendered HTML page over /v1/stats + the newest log rows. No JS, no build step.

HIGHLIGHTS: Production-এ এটা Grafana (Prometheus থেকে) + Phoenix/Langfuse (trace থেকে)। এখানে একটা
ছোট page যাতে Docker ছাড়াই monitoring "দেখা" যায়। প্রতিটা মান html.escape করা — প্রশ্নের text
user-এর দেওয়া, তাই escape না করলে dashboard-এ XSS।
"""
from __future__ import annotations

from html import escape

CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--bad:#b91c1c;--ok:#15803d;--warn:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c;
--bad:#f87171;--ok:#4ade80;--warn:#fbbf24}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:20px 0 8px}.muted{color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
.tile b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.tile span{color:var(--muted);font-size:12px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;background:var(--card)}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap;font-variant-numeric:tabular-nums}
td.q{white-space:normal;min-width:220px}.bad{color:var(--bad);font-weight:600}.ok{color:var(--ok)}.warn{color:var(--warn)}
"""


def _fmt(v, suffix="", pct=False):
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if pct else f"{v}{suffix}"


def render(summary: dict, recent: list[dict]) -> str:
    s, q, slo = summary, summary["quality"], summary["slo"]
    alerts = slo["availability"]["alerts"] + slo["latency"]["alerts"]
    tiles = [
        ("requests", s["requests"]), ("error rate", _fmt(s["rates"]["error"], pct=True)),
        ("latency p50 / p95", f"{_fmt(s['latency_s']['p50'], 's')} / {_fmt(s['latency_s']['p95'], 's')}"),
        ("TTFT p50 (stream)", _fmt(s["ttft_ms"]["p50"], " ms")), ("decode tok/s p50", _fmt(s["decode_tokens_per_s_p50"])),
        ("not-found rate", _fmt(s["rates"]["not_found"], pct=True)), ("cache hit", _fmt(s["rates"]["cache_hit"], pct=True)),
        ("degraded", s["outcomes"]["degraded"]), ("citations valid", _fmt(q["citation_valid_rate"], pct=True)),
        ("numbers supported", _fmt(q["numbers_supported_rate"], pct=True)),
        ("judge supported", f"{_fmt(q['judge_supported_rate'], pct=True)} (n={q['judged']})"),
        ("feedback 👍 / 👎", f"{s['feedback']['up']} / {s['feedback']['down']}"),
        ("GPU seconds", s["gpu_seconds"]), ("tokens in / out", f"{s['tokens']['prompt']} / {s['tokens']['completion']}"),
    ]
    burn = "".join(
        f"<tr><td>{name}</td><td>{escape(str(v['target']))}</td>"
        + "".join(f"<td class='{'bad' if (b or 0) > 6 else 'warn' if (b or 0) > 1 else 'ok'}'>{_fmt(b, '×')}</td>"
                  for b in v["burn_rate"].values())
        + f"<td>{escape(', '.join(v['alerts']) or 'none')}</td></tr>"
        for name, v in (("availability", slo["availability"]), ("latency", slo["latency"])))
    rows = "".join(
        "<tr>"
        f"<td>{escape(r['request_id'] or '')}</td><td>{escape(r['op'])}</td>"
        f"<td class='{'bad' if r['outcome'] in ('error', 'degraded') else ''}'>{escape(r['outcome'] or '')}</td>"
        f"<td>{_fmt(r['latency_s'], 's')}</td><td>{_fmt(r['ttft_ms'], ' ms')}</td>"
        f"<td>{_fmt(r['prompt_tokens'])}/{_fmt(r['completion_tokens'])}</td>"
        f"<td>{_flags(r)}</td><td>{_judge(r)}</td>"
        f"<td>{ {1: '👍', -1: '👎'}.get(r['feedback'], '') }</td>"
        f"<td class='q'>{escape((r['question'] or '')[:140])}</td></tr>"
        for r in recent)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="10">
<link rel="icon" href="data:,">
<title>RAG API Monitor</title><style>{CSS}</style></head><body>
<h1>RAG API Monitor</h1><div class="muted">last {int(s['window_s'] // 60)} min · refreshes every 10 s ·
alerts: <span class="{'bad' if alerts else 'ok'}">{escape(', '.join(alerts) or 'none')}</span></div>
<div class="tiles" style="margin-top:12px">{''.join(f"<div class='tile'><b>{escape(str(v))}</b><span>{escape(k)}</span></div>" for k, v in tiles)}</div>
<h2>SLO burn rate (1.0 = budget lasts exactly the period)</h2>
<div class="scroll"><table><tr><th>SLI</th><th>target</th><th>5m</th><th>30m</th><th>1h</th><th>6h</th><th>alerts</th></tr>{burn}</table></div>
<h2>Recent requests</h2>
<div class="scroll"><table><tr><th>request</th><th>op</th><th>outcome</th><th>latency</th><th>TTFT</th><th>tokens</th>
<th>flags</th><th>judge</th><th>fb</th><th>question</th></tr>{rows}</table></div>
</body></html>"""


def _flags(r: dict) -> str:
    out = []
    if r["citation_valid"] == 0:
        out.append("bad citation")
    if r["numbers_supported"] == 0:
        out.append(f"number? {r['unsupported_numbers']}")
    if r["truncated"]:
        out.append("truncated")
    return f"<span class='bad'>{escape(', '.join(out))}</span>" if out else ""


def _judge(r: dict) -> str:
    if r["judge_supported"] is None:
        return ""
    return "<span class='ok'>✔</span>" if r["judge_supported"] else f"<span class='bad' title='{escape(r['judge_reason'] or '')}'>✘</span>"
