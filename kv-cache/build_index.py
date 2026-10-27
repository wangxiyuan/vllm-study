#!/usr/bin/env python3
"""Build the KV Cache single-page web app from the markdown chapters."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "index.html"

CHAPTERS = [
    dict(id="start", num="00", title="KV Cache 全景", file="README.md",
         desc="四个模型、四种 spec、七个环节"),
    dict(id="models", num="01", title="四模型结构", file="01-models.md",
         desc="参数量、每层 spec、KV 字节手算"),
    dict(id="lifecycle", num="02", title="全链路", file="02-lifecycle.md",
         desc="spec → group → tensor → bind → slot"),
    dict(id="groups", num="03", title="分组与后端", file="03-groups-backends.md",
         desc="为什么 group 跨层、backend 怎么选"),
    dict(id="trace", num="04", title="一次请求全流程", file="07-trace.md",
         desc="“你好，你是谁”的 6 个 token 从 prefill 到 decode"),
    dict(id="labnotes", num="05", title="验证与调试", file="04-lab.md",
         desc="打印 spec/group、对拍、练习"),
    dict(id="quiz", num="06", title="自测 20 题", file="05-self-test.md",
         desc="闭卷检验；答案默认折叠"),
    dict(id="appendix", num="07", title="KV 速查表", file="08-cheatsheet.md",
         desc="四模型、四种 spec、五个公式、易错点"),
]

LABS = [
    dict(id="request", title="请求演示", file="request-demo.html",
         desc="按 4 步走完 prefill → sample → decode", back="chapter/trace"),
    dict(id="visualizer", title="互动实验室", file="visualizer.html",
         desc="模型地图、分页、显存、Ring、自测", back="chapter/trace"),
    dict(id="grouping", title="分组实验", file="grouping.html",
         desc="从模型层到 spec 条目，再到 group", back="chapter/lifecycle"),
]

CONFIG_MODELS = [
    dict(slug="ds", name="DeepSeek-V4.1-Flash", file="deepseek_v41_flash.json",
         params="552B 总参 / 8B 输入、16B 输出", model_id="deepseek-ai/DeepSeek-V4.1-Flash-1017"),
    dict(slug="kimi", name="Kimi-K3", file="kimi_k3.json",
         params="2.8T 总参 / 104B 激活", model_id="moonshotai/Kimi-K3"),
    dict(slug="qwen", name="Qwen3.8-Flash-Next", file="qwen3_8_flash_next.json",
         params="约 180B / 6B 主干", model_id="Qwen/Qwen3.8-Flash-Next"),
    dict(slug="glm", name="GLM-5.3-Flash", file="glm5_3_flash.json",
         params="320B 总参 / 18B 激活", model_id="ZhipuAI/GLM-5.3-Flash"),
]

MD_LINK_MAP = {
    "README.md": "#/chapter/start",
    "01-models.md": "#/chapter/models",
    "02-lifecycle.md": "#/chapter/lifecycle",
    "03-groups-backends.md": "#/chapter/groups",
    "07-trace.md": "#/chapter/trace",
    "04-lab.md": "#/chapter/labnotes",
    "05-self-test.md": "#/chapter/quiz",
    "08-cheatsheet.md": "#/chapter/appendix",
    "request-demo.html": "#/lab/request",
    "visualizer.html": "#/lab/visualizer",
}

md = MarkdownIt("commonmark").enable("table").enable("strikethrough")


def strip_first_h1(text: str) -> str:
    m = re.search(r"^#\s+(.+?)\s*$", text, flags=re.M)
    if not m:
        return text
    return text[:m.start()] + text[m.end():]


def rewrite_links(s: str) -> str:
    for old, new in MD_LINK_MAP.items():
        s = s.replace(f'href="{old}"', f'href="{new}"')
    return s


def add_heading_ids(body: str, cid: str) -> tuple[str, list[tuple[str, str]]]:
    toc: list[tuple[str, str]] = []
    counter = 0

    def repl(m: re.Match) -> str:
        nonlocal counter
        counter += 1
        hid = f"ch-{cid}-s{counter}"
        inner = m.group(1)
        text = re.sub(r"<[^>]+>", "", inner)
        text = html.unescape(text).strip()
        toc.append((hid, text))
        return f'<h2 id="{hid}">{inner}</h2>'

    body = re.sub(r"<h2>(.*?)</h2>", repl, body, flags=re.S)
    return body, toc


def render_md(text: str, cid: str) -> tuple[str, list[tuple[str, str]]]:
    body = md.render(strip_first_h1(text))
    body = rewrite_links(body)
    return add_heading_ids(body, cid)


def _short(v):
    if isinstance(v, list):
        if len(v) > 8:
            return f"[{', '.join(map(str, v[:6]))}, …]（{len(v)} 项）"
        return str(v)
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_short(val)}" for k, val in list(v.items())[:5]) + ("…" if len(v) > 5 else "") + "}"
    return str(v)


def configs_section() -> str:
    cards, scripts = [], []
    for m in CONFIG_MODELS:
        c = json.loads((ROOT / "configs" / m["file"]).read_text())
        t = c.get("text_config", c)
        layer_types = t.get("layer_types")
        counts = ""
        if layer_types:
            from collections import Counter
            counts = ", ".join(f"{k}={v}" for k, v in Counter(layer_types).items())
        rows = [
            ("架构", c.get("architectures", ["?"])[0]),
            ("参数", m["params"]),
            ("层数", t.get("num_hidden_layers")),
            ("hidden", t.get("hidden_size")),
            ("attention heads", t.get("num_attention_heads")),
            ("KV heads", t.get("num_key_value_heads")),
            ("head_dim", t.get("head_dim")),
            ("qk_nope / rope", f'{t.get("qk_nope_head_dim")} / {t.get("qk_rope_head_dim")}'),
            ("kv_lora_rank", t.get("kv_lora_rank")),
            ("max_position", t.get("max_position_embeddings")),
            ("full / linear", counts or "—"),
            ("compress_ratios", _short(t.get("compress_ratios")) if t.get("compress_ratios") else "—"),
        ]
        table = "".join(f"<tr><th>{k}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows if v not in (None, "None"))
        raw = json.dumps(c, ensure_ascii=False, indent=2)
        cards.append(f'''<div class="cfg-card">
  <div class="cfg-head"><b>{html.escape(m["name"])}</b><span>{html.escape(m["model_id"])}</span></div>
  <table>{table}</table>
  <button class="btn small" data-config="cfg-{m["slug"]}" data-title="{html.escape(m["name"])}">查看完整 config.json</button>
</div>''')
        scripts.append(f'<script type="application/json" id="cfg-{m["slug"]}">{raw}</script>')
    return ('''<h2 id="ch-models-config">原始 config</h2>
<div class="lead"><p>四个模型的原始配置：</p></div>
<div class="cfg-grid">''' + "".join(cards) + "</div>" + "".join(scripts))


def lab_view(lab: dict) -> str:
    src = (ROOT / lab["file"]).read_text()
    escaped = html.escape(src, quote=True)
    return f'''<section class="view labview" id="view-lab-{lab["id"]}" data-view="lab/{lab["id"]}">
  <div class="labhead">
    <a class="btn" href="#/home">⌂ 首页</a>
    <a class="btn" href="#/{lab.get("back", "chapter/trace")}">← 返回章节</a>
    <div><h1>{html.escape(lab["title"])}</h1><p>{html.escape(lab["desc"])}</p></div>
    <a class="btn ghost" href="{lab["file"]}" target="_blank" rel="noopener">新窗口 ↗</a>
  </div>
  <iframe class="labframe" title="{html.escape(lab["title"])}" srcdoc="{escaped}"></iframe>
</section>'''


CSS = r'''
:root{
  --bg:#080d19; --panel:#111a2e; --panel2:#16223b; --line:#26324f;
  --text:#e9effc; --muted:#93a2c4; --accent:#6ea8fe; --accent2:#7ee7c7;
  --warn:#ffcc66; --code:#0a1124; --maxw:960px;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:radial-gradient(1100px 560px at 12% -8%,#16264a 0%,var(--bg) 58%);
     color:var(--text);font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB",sans-serif}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
code,pre,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.shell{display:grid;grid-template-columns:286px minmax(0,1fr);min-height:100vh}
aside{position:sticky;top:0;height:100vh;overflow:auto;border-right:1px solid var(--line);
      background:linear-gradient(180deg,#0c1426,#0a111f);padding:18px 16px 26px;z-index:20}
.brand{display:flex;gap:10px;align-items:center;margin-bottom:14px;color:var(--text)}
.logo{width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#2d6bd0,#7ee7c7);
      display:grid;place-items:center;font-weight:800;color:#07101f}
.brand:hover{text-decoration:none}
.brand b{display:block;font-size:15px} .brand span{color:var(--muted);font-size:11px}
.progressbox{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin:8px 0 14px}
.progressbox .ptop{display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}
.pbar{height:7px;background:#0a1326;border-radius:999px;overflow:hidden;margin-top:7px}
.pbar i{display:block;height:100%;width:0;background:linear-gradient(90deg,#3d7bd6,#7ee7c7);transition:width .25s}
.search{width:100%;background:#0a1326;border:1px solid var(--line);border-radius:9px;color:var(--text);
        padding:8px 10px;font:inherit;outline:none;margin-bottom:10px}
.search:focus{border-color:var(--accent)}
nav{display:flex;flex-direction:column;gap:3px}
.navgroup{color:#617196;font-size:10px;letter-spacing:1.2px;margin:12px 8px 4px}
.navchapter{border-radius:9px;overflow:hidden}
.navlink{display:flex;gap:8px;align-items:center;padding:7px 8px;border-radius:8px;color:#cfdbf5;font-size:13px}
.navlink:hover{background:#162440;text-decoration:none}
.navlink.active{background:#172b4c;color:#fff}
.navnum{width:23px;color:var(--muted);font-size:10px;flex:none}
.navcheck{margin-left:auto;color:#5f7097;font-size:12px}
.navchapter.done .navcheck{color:var(--accent2)}
.navsub{display:block;padding:3px 8px 3px 39px;color:#7f90b4;font-size:11.5px;border-left:1px solid #1d2a46;margin-left:23px}
.navsub:hover{color:var(--text);text-decoration:none}
.navchapter.hidden,.navsub.hidden{display:none}
.sidefoot{margin-top:16px;border-top:1px solid var(--line);padding-top:10px;color:#68789b;font-size:10.5px;line-height:1.6}
main{min-width:0}
.content{max-width:var(--maxw);margin:0 auto;padding:30px 38px 70px}
.view{display:none}
.view.active{display:block}
.hero{padding:26px 0 24px}
.eyebrow{color:var(--accent2);font-size:11px;letter-spacing:1.6px;text-transform:uppercase}
.hero h1{font-size:42px;line-height:1.15;margin:8px 0 10px;letter-spacing:-1px}
.hero h1 span{background:linear-gradient(90deg,#8db9ff,#7ee7c7);-webkit-background-clip:text;background-clip:text;color:transparent}
.herosub{color:#b6c4e2;font-size:16px;max-width:720px;margin:0 0 18px}
.hero-actions{display:flex;gap:9px;flex-wrap:wrap;margin:18px 0 22px}
.btn{display:inline-block;border:1px solid var(--line);background:var(--panel2);color:var(--text);
      border-radius:9px;padding:8px 13px;font:inherit;font-size:13px;cursor:pointer}
.btn:hover{border-color:#47679e;text-decoration:none}
.btn.primary{background:#2457a6;border-color:#3b73c9}
.btn.ghost{background:transparent}
.btn.small{padding:5px 9px;font-size:11px;margin-top:8px}
.modelgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:9px}
.mcard{background:linear-gradient(180deg,#121c33,#0f1729);border:1px solid var(--line);border-radius:12px;padding:11px 13px}
.mname{color:#c9d7f2;font-size:12px} .mnum{font-size:22px;font-weight:800;margin:2px 0 3px} .mline{color:#8fa0c2;font-size:11px}
.mtag{color:var(--accent2);font-size:10.5px;margin-top:6px}
.homegrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:18px}
.homecard{display:block;background:#0e1729;border:1px solid var(--line);border-radius:12px;padding:12px 14px;color:var(--text)}
.homecard:hover{border-color:#3c65a8;text-decoration:none}
.homecard b{display:block;margin-bottom:3px} .homecard span{color:var(--muted);font-size:11.5px}
.chapter-head{position:relative;margin-bottom:8px}
.kicker{color:var(--accent2);font-size:11px;letter-spacing:1.4px}
.chapter-head h1{font-size:30px;line-height:1.25;margin:4px 0}
.chapter-desc{color:var(--muted);margin:0 0 6px;font-size:13px}
.mark{position:absolute;right:0;top:10px;color:#91a2c6;font-size:11px;cursor:pointer}
.mark input{vertical-align:-2px;accent-color:#7ee7c7}
.chapter h2{font-size:20px;margin:26px 0 8px;scroll-margin-top:18px}
.chapter h3{font-size:16px;margin:20px 0 6px}
.chapter p{margin:8px 0} .chapter ul,.chapter ol{margin:8px 0 8px 22px;padding:0}
.chapter li{margin:3px 0}
.chapter blockquote{margin:12px 0;padding:8px 14px;border-left:3px solid #3d6fb5;background:#0d1729;color:#b9c7e4;border-radius:0 8px 8px 0}
.chapter table{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0;display:block;overflow-x:auto}
.chapter th,.chapter td{border-bottom:1px solid #22304c;padding:7px 8px;text-align:left;vertical-align:top;min-width:90px}
.chapter th{color:#9fb0d2;font-size:12px;font-weight:600}
.chapter pre{background:var(--code);border:1px solid #1e2c49;border-radius:10px;padding:12px 14px;overflow:auto;font-size:12.5px;line-height:1.55;margin:10px 0;position:relative}
.chapter code{background:#0c162a;border:1px solid #1d2a45;border-radius:4px;padding:1px 4px;font-size:.92em}
.chapter pre code{background:none;border:0;padding:0;font-size:inherit}
.chapter hr{border:0;border-top:1px solid #1b2842;margin:22px 0}
.chapter a{border-bottom:1px dotted #34507f}
.chapter-foot{display:flex;justify-content:space-between;margin-top:22px;font-size:12px;color:#7f90b4}
.lead{background:#0f1a2e;border:1px solid var(--line);border-radius:12px;padding:12px 15px;margin:12px 0;color:#c3d1ea}
.muted{color:var(--muted)}
.callout{background:#0e1e1b;border:1px solid #245b4d;border-radius:12px;padding:12px 15px;margin:16px 0;display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.callout b{color:var(--accent2)}
.cfg-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px;margin:10px 0}
.cfg-card{border:1px solid #26324f;border-radius:11px;background:#0d1628;padding:10px 12px;min-width:0}
.cfg-head{display:flex;flex-direction:column;gap:2px;margin-bottom:6px}
.cfg-head b{font-size:13px} .cfg-head span{color:#8596ba;font-size:10px}
.cfg-card table{font-size:11.5px;margin:0}
.cfg-card th{width:38%;color:#8fa1c4;font-weight:500;min-width:0} .cfg-card td{min-width:0}
details.answers{margin:16px 0;border:1px solid #26324f;border-radius:10px;background:#0c1527;padding:0 14px 10px}
details.answers summary{cursor:pointer;color:#7ee7c7;padding:10px 0;font-size:13px}
details.answers[open] summary{margin-bottom:4px}
.copybtn{position:absolute;right:8px;top:8px;border:1px solid #2b3b5d;background:#111d35;color:#9fb0d2;
         border-radius:6px;font-size:10px;padding:3px 7px;cursor:pointer;opacity:.75}
.copybtn:hover{opacity:1}
.labview{padding:0}
.labhead{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:12px 0 10px}
.labhead h1{font-size:20px;margin:0}
.labhead p{color:var(--muted);font-size:12px;margin:2px 0 0}
.labhead .btn{margin:0}
.labframe{display:block;width:100%;height:calc(100vh - 138px);min-height:620px;border:1px solid var(--line);border-radius:12px;background:#0a1120}
.modal{display:none;position:fixed;inset:0;background:#02060ecc;z-index:100;padding:18px;backdrop-filter:blur(4px)}
.modal.open{display:flex;align-items:center;justify-content:center}
.modalbox{background:#0d1729;border:1px solid #2b3b5d;border-radius:14px;max-width:960px;width:100%;max-height:88vh;display:flex;flex-direction:column;padding:14px 16px}
.modalbox h3{margin:0 0 8px;font-size:16px}
.modalbox pre{flex:1;overflow:auto;background:#080f1d;border:1px solid #1d2a45;border-radius:9px;padding:12px;font-size:12px;line-height:1.5;margin:0}
.modalbar{display:flex;gap:8px;justify-content:flex-end;margin-top:10px}
.topbar{display:none}
@media(max-width:980px){
  .shell{grid-template-columns:1fr} aside{display:none} .content{padding:18px 16px 60px}
  .hero h1{font-size:31px} .chapter-head h1{font-size:25px} .mark{position:static;display:block;margin:4px 0 0}
  .topbar{display:flex;position:sticky;top:0;z-index:30;align-items:center;gap:10px;padding:9px 14px;
           background:#0b1323ee;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}
  .topbar .toplink{font-size:13px;font-weight:700;color:var(--text)} .topbar .spacer{flex:1}
  .topbar button{background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:8px;padding:6px 9px}
  aside.open{display:block;position:fixed;inset:0;height:100vh;width:88%;max-width:340px;box-shadow:20px 0 60px #0008}
  .labframe{height:calc(100vh - 190px);min-height:560px}
}
@media print{
  aside,.topbar,.hero-actions,.mark,.labhead,.modal{display:none!important}
  body{background:#fff;color:#111} .shell{display:block} .content{max-width:none;padding:0}
  .view{display:block!important} .hero,.chapter{border-color:#ddd}
  .chapter h1,.hero h1{color:#111} .herosub,.chapter-desc,.muted{color:#444}
  .chapter pre{background:#f6f8fb;border-color:#ddd;color:#111} .chapter code{background:#f2f4f8;color:#111}
  .chapter a{color:#1450a3} .labview{display:none!important}
}
'''

JS = r'''
const CHAPTERS = __CHAPTERS__;
const KEY = 'vllm-kv-course-v2';
let done = {};
try { done = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch(e) { done = {}; }

function chapterNav(){ return CHAPTERS.map((c,i)=>({c,i})); }
function updateProgress(){
  const n = CHAPTERS.filter(id => done[id]).length;
  const pct = Math.round(n / CHAPTERS.length * 100);
  const pbar=document.getElementById('pbar'); if(pbar) pbar.style.width = pct + '%';
  const pt=document.getElementById('pct'); if(pt) pt.textContent = pct + '%';
  document.querySelectorAll('[data-check]').forEach(el=>{
    const id=el.getAttribute('data-check');
    el.textContent = done[id] ? '●' : '○';
    const ch=el.closest('.navchapter'); if(ch) ch.classList.toggle('done', !!done[id]);
  });
  document.querySelectorAll('[data-done]').forEach(cb=>{ cb.checked = !!done[cb.getAttribute('data-done')]; });
}
document.querySelectorAll('[data-done]').forEach(cb=>{
  cb.addEventListener('change',()=>{
    done[cb.getAttribute('data-done')] = cb.checked;
    try { localStorage.setItem(KEY, JSON.stringify(done)); } catch(e) {}
    updateProgress();
  });
});

function setActive(route){
  document.querySelectorAll('[data-route]').forEach(a=>a.classList.toggle('active', a.getAttribute('data-route')===route));
}
function show(id){
  document.querySelectorAll('.view').forEach(v=>v.classList.toggle('active', v.id===id));
}
function parseHash(){
  const raw = location.hash.replace(/^#\/?/, '');
  if(!raw || raw === 'home') return {kind:'home'};
  const parts = raw.split('/');
  return {kind:parts[0], id:parts[1], anchor:parts[2]};
}
function route(){
  const r = parseHash();
  if(r.kind === 'home'){
    show('view-home'); setActive('home'); window.scrollTo(0,0); return;
  }
  if(r.kind === 'chapter'){
    const id = r.id || CHAPTERS[0].id;
    show('view-chapter-' + id); setActive('chapter/' + id);
    if(r.anchor){ setTimeout(()=>document.getElementById('ch-'+id+'-'+r.anchor)?.scrollIntoView({block:'start'}), 20); }
    else window.scrollTo(0,0);
    return;
  }
  if(r.kind === 'lab'){
    show('view-lab-' + r.id); setActive('lab/' + r.id); window.scrollTo(0,0); return;
  }
  show('view-home'); setActive('home');
}
window.addEventListener('hashchange', route);
window.addEventListener('load', route);

function filterNav(){
  const q=(document.getElementById('search').value||'').trim().toLowerCase();
  document.querySelectorAll('.navchapter').forEach(ch=>{
    const subs = [...ch.querySelectorAll('.navsub')];
    const chMatch = !q || (ch.getAttribute('data-search')||'').includes(q);
    const subMatch = subs.some(sub => (sub.getAttribute('data-search')||'').includes(q));
    ch.classList.toggle('hidden', !(!q || chMatch || subMatch));
    subs.forEach(sub=>{
      const sm = !q || chMatch || (sub.getAttribute('data-search')||'').includes(q);
      sub.classList.toggle('hidden', !sm);
    });
  });
  document.querySelectorAll('.navlab').forEach(el=>{
    const ok = !q || (el.getAttribute('data-search')||'').includes(q);
    el.classList.toggle('hidden', !ok);
  });
}
document.getElementById('search').addEventListener('input',filterNav);
document.addEventListener('keydown',(e)=>{
  if(e.key==='/' && document.activeElement.tagName!=='INPUT'){ e.preventDefault(); document.getElementById('search').focus(); }
  if(e.key==='Escape'){ document.getElementById('sidebar').classList.remove('open'); closeModal(); }
});
document.getElementById('menuBtn').addEventListener('click',()=>document.getElementById('sidebar').classList.toggle('open'));
document.getElementById('printBtn').addEventListener('click',()=>window.print());

// config modal
const modal = document.getElementById('modal');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');
function openModal(title, text){
  modalTitle.textContent = title;
  try { modalBody.textContent = JSON.stringify(JSON.parse(text), null, 2); }
  catch(e) { modalBody.textContent = text; }
  modal.classList.add('open');
}
function closeModal(){ modal.classList.remove('open'); }
document.querySelectorAll('[data-config]').forEach(btn=>{
  btn.addEventListener('click',()=>{
    const id=btn.getAttribute('data-config');
    const el=document.getElementById(id);
    if(el) openModal(btn.getAttribute('data-title') || id, el.textContent);
  });
});
document.getElementById('modalClose').addEventListener('click', closeModal);
modal.addEventListener('click',(e)=>{ if(e.target===modal) closeModal(); });
document.getElementById('modalCopy').addEventListener('click',()=>{
  if(navigator.clipboard) navigator.clipboard.writeText(modalBody.textContent).then(()=>{
    const b=document.getElementById('modalCopy'); b.textContent='已复制'; setTimeout(()=>b.textContent='复制 JSON',1200);
  });
});

// copy code blocks
document.querySelectorAll('.chapter pre').forEach(pre=>{
  const b=document.createElement('button'); b.className='copybtn'; b.textContent='复制';
  b.addEventListener('click',()=>{
    const code=pre.querySelector('code'); const txt=code?code.innerText:pre.innerText;
    if(navigator.clipboard) navigator.clipboard.writeText(txt).then(()=>{b.textContent='已复制';setTimeout(()=>b.textContent='复制',1200);});
  });
  pre.appendChild(b);
});
updateProgress();
'''


def build() -> None:
    views, nav = [], []

    home = '''<section class="view active" id="view-home" data-view="home">
  <div class="hero">
    <div class="eyebrow">vLLM @ 4e964b2623 · 四模型 KV Cache</div>
    <h1>KV Cache，<span>从结构到 slot</span></h1>
    <p class="herosub">DeepSeek V4.1 / Kimi K3 / Qwen3.8 / GLM-5.3 的 KV 结构、分组、tensor 布局与运行时寻址。</p>
    <div class="hero-actions">
      <a class="btn primary" href="#/chapter/start">开始阅读</a>
      <a class="btn" href="#/lab/request">打开请求演示</a>
      <a class="btn" href="#/lab/grouping">打开分组实验</a>
      <a class="btn" href="#/lab/visualizer">打开互动实验室</a>
      <button class="btn ghost" onclick="window.print()">打印 / PDF</button>
    </div>
    <div class="modelgrid">
      <div class="mcard"><div class="mname">DeepSeek V4.1 Flash</div><div class="mnum">552B</div><div class="mline">8B 输入 / 16B 输出 · 40 层 sparse MLA</div><div class="mtag">跨层共享 compressed KV</div></div>
      <div class="mcard"><div class="mname">Kimi K3</div><div class="mnum">2.8T</div><div class="mline">104B 激活 · 24 MLA + 69 KDA</div><div class="mtag">线性层固定 state</div></div>
      <div class="mcard"><div class="mname">Qwen3.8 Flash Next</div><div class="mnum">≈180B</div><div class="mline">6B 主干 · 12 GQA + 36 GDN</div><div class="mtag">QSA 压缩 key + ring</div></div>
      <div class="mcard"><div class="mname">GLM-5.3 Flash</div><div class="mnum">320B</div><div class="mline">18B 激活 · 11 sparse MLA + 34 KDA</div><div class="mtag">indexer kpool + tail ring</div></div>
    </div>
    <div class="homegrid">
      <a class="homecard" href="#/chapter/models"><b>01 四模型结构</b><span>每层到底要什么 cache，字节怎么算</span></a>
      <a class="homecard" href="#/chapter/lifecycle"><b>02 全链路</b><span>spec → group → tensor → bind → slot</span></a>
      <a class="homecard" href="#/lab/grouping"><b>分组实验</b><span>模型层 → spec 条目 → group</span></a>
      <a class="homecard" href="#/lab/request"><b>请求演示</b><span>6 个 token 从 prefill 到 decode</span></a>
    </div>
  </div>
</section>'''
    views.append(home)

    for idx, ch in enumerate(CHAPTERS):
        cid = ch["id"]
        body, toc = render_md((ROOT / ch["file"]).read_text(), cid)
        if cid == "models":
            body += configs_section()
            toc.append(("ch-models-config", "原始 config"))
        if cid == "quiz":
            m = re.search(r"<h2[^>]*>参考答案</h2>", body)
            if m:
                body = (body[:m.start()]
                        + '<details class="answers"><summary>展开参考答案</summary>'
                        + body[m.start():] + '</details>')
        if cid == "trace":
            body += '''<div class="callout"><div><b>可交互</b>　把这个请求按 4 步走一遍：Prefill → 采样 → Decode。</div>
              <a class="btn primary" href="#/lab/request">打开请求演示</a></div>'''
        if cid == "lifecycle":
            body += '''<div class="callout"><div><b>可交互</b>　看 40/93/48/45 层如何收缩成 6/2/5/4 个 group。</div>
              <a class="btn primary" href="#/lab/grouping">打开分组实验</a></div>'''
        prev_id = CHAPTERS[idx-1]["id"] if idx else "home"
        prev_href = "#/home" if prev_id == "home" else f"#/chapter/{prev_id}"
        prev_label = "← 返回首页" if idx == 0 else "← 上一章"
        if idx + 1 < len(CHAPTERS):
            next_href = f"#/chapter/{CHAPTERS[idx+1]['id']}"
            next_label = "下一章 →"
        else:
            next_href = "#/home"
            next_label = "回到首页 →"
        views.append(f'''<section class="view chapter" id="view-chapter-{cid}" data-view="chapter/{cid}">
  <div class="chapter-head">
    <div class="kicker">Chapter {ch["num"]}</div>
    <h1>{ch["title"]}</h1>
    <p class="chapter-desc">{ch["desc"]}</p>
    <label class="mark"><input type="checkbox" data-done="{cid}"> 标记完成</label>
  </div>
  {body}
  <div class="chapter-foot"><a href="{prev_href}">{prev_label}</a><a href="{next_href}">{next_label}</a></div>
</section>''')

        subs = "".join(f'<a class="navsub" href="#/chapter/{cid}/{hid.split("-")[-1]}" data-search="{html.escape(text.lower())}">{html.escape(text)}</a>' for hid, text in toc)
        search_blob = (ch["title"] + " " + ch["desc"] + " " + " ".join(t for _, t in toc)).lower()
        nav.append(f'''<div class="navchapter" data-search="{html.escape(search_blob)}">
  <a class="navlink" href="#/chapter/{cid}" data-route="chapter/{cid}"><span class="navnum">{ch["num"]}</span><span>{html.escape(ch["title"])}</span><span class="navcheck" data-check="{cid}">○</span></a>
  {subs}
</div>''')

    for lab in LABS:
        views.append(lab_view(lab))
        nav.append(f'''<a class="navlink navlab" href="#/lab/{lab["id"]}" data-route="lab/{lab["id"]}" data-search="{html.escape((lab["title"]+' '+lab["desc"]).lower())}">
  <span class="navnum">↗</span><span>{html.escape(lab["title"])}</span></a>''')

    sidebar_nav = '<a class="navlink navhome" href="#/home" data-route="home"><span class="navnum">⌂</span><span>首页</span></a><div class="navgroup">章节</div>' + "".join(nav[:len(CHAPTERS)]) + '<div class="navgroup">实验室</div>' + "".join(nav[len(CHAPTERS):])
    page = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vLLM KV Cache · 四模型与全链路</title>
<style>''' + CSS + '''</style>
</head>
<body>
<div class="topbar"><button id="menuBtn">☰</button><a class="toplink" href="#/home">⌂ 首页</a><span class="spacer"></span><button id="printBtn">打印</button></div>
<div class="shell">
  <aside id="sidebar">
    <a class="brand" href="#/home"><div class="logo">KV</div><div><b>KV Cache</b><span>vLLM @ 4e964b2623</span></div></a>
    <div class="progressbox"><div class="ptop"><span>阅读进度</span><span id="pct">0%</span></div><div class="pbar"><i id="pbar"></i></div></div>
    <input class="search" id="search" placeholder="搜索章节 / 小节  (/ 聚焦)">
    <nav>''' + sidebar_nav + '''</nav>
    <div class="sidefoot">基线：<span class="mono">4e964b2623</span></div>
  </aside>
  <main><div class="content">''' + "".join(views) + '''</div></main>
</div>
<div class="modal" id="modal"><div class="modalbox">
  <div style="display:flex;justify-content:space-between;gap:12px;align-items:center">
    <h3 id="modalTitle">config.json</h3><button class="btn small" id="modalClose">关闭 ×</button>
  </div>
  <pre id="modalBody"></pre>
  <div class="modalbar"><button class="btn small" id="modalCopy">复制 JSON</button></div>
</div></div>
<script>''' + JS.replace("__CHAPTERS__", json.dumps([c["id"] for c in CHAPTERS])) + '''</script>
</body>
</html>'''
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page)/1024:.1f} KiB)")


if __name__ == "__main__":
    build()
