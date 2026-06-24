#!/usr/bin/env python3
"""
LyricsGPT Studio - a simple local web UI for generating lyrics.

Put this file in the SAME FOLDER as the files produced by the notebook:
    lyrics_gpt.pt        (model weights + saved config)
    tokenizer.json       (byte-level BPE tokenizer)

Then run:
    python app.py
and open http://127.0.0.1:8000 in your browser.

Only depends on `torch` and `tokenizers` (already installed if you trained the
model), plus Python's built-in HTTP server - no Flask or other extras.
"""
import json
import math
import os
import sys
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

CKPT_PATH = "lyrics_gpt.pt"
CONFIG_PATH = "lyrics_gpt_config.json"
TOKENIZER_PATH = "tokenizer.json"
EOT = "<|endoftext|>"
HOST = "127.0.0.1"
PORT = 8000


# --------------------------------------------------------------------------- #
#  Model definition (must match the notebook exactly so the weights load)     #
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LyricsGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 repetition_penalty=1.0, no_repeat_ngram_size=0,
                 penalty_window=0, min_new_tokens=0, eos_id=None):
        self.eval()
        start_len = idx.shape[1]
        for _ in range(max_new_tokens):
            cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            generated = idx.shape[1] - start_len
            if eos_id is not None and generated < min_new_tokens:
                logits[:, eos_id] = float('-inf')

            for b in range(idx.size(0)):
                full = idx[b].tolist()
                recent = full[-penalty_window:] if (penalty_window and penalty_window > 0) else full

                if no_repeat_ngram_size >= 2:
                    n = no_repeat_ngram_size
                    L = len(recent)
                    if L >= n:
                        head = tuple(recent[L - n + 1:])
                        banned = set()
                        for i in range(0, L - n + 1):
                            if tuple(recent[i:i + n - 1]) == head:
                                banned.add(recent[i + n - 1])
                        for tok in banned:
                            logits[b, tok] = float('-inf')

                if repetition_penalty != 1.0:
                    score = logits[b]
                    for tok in set(recent):
                        if score[tok] > 0:
                            score[tok] = score[tok] / repetition_penalty
                        else:
                            score[tok] = score[tok] * repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            if top_p is not None and top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                mask = (cum - sp) > top_p
                sp[mask] = 0.0
                sp = sp / sp.sum(dim=-1, keepdim=True)
                nxt = torch.multinomial(sp, num_samples=1)
                next_id = si.gather(1, nxt)
            else:
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
        return idx


# --------------------------------------------------------------------------- #
#  Load model, tokenizer, config                                              #
# --------------------------------------------------------------------------- #
def _to_cfg(d):
    return SimpleNamespace(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


def load_everything():
    missing = [p for p in (CKPT_PATH, TOKENIZER_PATH) if not os.path.exists(p)]
    if missing:
        print("\n[ERROR] Missing files in this folder:", ", ".join(missing))
        print("Put 'lyrics_gpt.pt' and 'tokenizer.json' (from the notebook's download)")
        print("next to app.py, then run again.\n")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    cfg = None
    if isinstance(ckpt, dict) and ckpt.get("cfg"):
        cfg = _to_cfg(ckpt["cfg"])
    elif os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = _to_cfg(json.load(f))
    if cfg is None:
        print("[ERROR] Could not find model config in the checkpoint or", CONFIG_PATH)
        sys.exit(1)

    for k in ("n_layer", "n_embd", "n_head", "block_size", "vocab_size"):
        if not hasattr(cfg, k):
            setattr(cfg, k, {"n_layer": 6, "n_embd": 384, "n_head": 6,
                             "block_size": 256, "vocab_size": 8000}[k])
    if not hasattr(cfg, "dropout"):
        cfg.dropout = 0.0

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eot_id = tokenizer.token_to_id(EOT)

    model = LyricsGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()

    val_loss = ckpt.get("val_loss") if isinstance(ckpt, dict) else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {n_params/1e6:.1f}M params | device: {device} | "
          f"val_loss: {val_loss if val_loss is not None else 'n/a'}")
    return model, tokenizer, cfg, eot_id, device, n_params, val_loss


MODEL, TOKENIZER, CFG, EOT_ID, DEVICE, N_PARAMS, VAL_LOSS = load_everything()

USE_AMP = (DEVICE == "cuda")
AMP_DTYPE = torch.float16
if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16


# --------------------------------------------------------------------------- #
#  Generation helper                                                          #
# --------------------------------------------------------------------------- #
def _trim_cut_off(text):
    work = text.rstrip()
    if not work:
        return text
    last_break = work.rfind("\n\n")
    if last_break != -1:
        head = work[:last_break].rstrip()
        if len(head) >= 0.5 * len(work):
            return head
    last_nl = work.rfind("\n")
    if last_nl != -1:
        head = work[:last_nl].rstrip()
        if len(head) >= 0.5 * len(work):
            return head
    return work


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def generate_lyrics(params):
    prompt = str(params.get("prompt", "")).strip() or "Late at night I think of you"
    temperature = _clamp(float(params.get("temperature", 0.9)), 0.1, 2.0)
    top_k = _clamp(int(params.get("top_k", 50)), 0, 1000)
    top_p = _clamp(float(params.get("top_p", 0.95)), 0.1, 1.0)
    repetition_penalty = _clamp(float(params.get("repetition_penalty", 1.15)), 0.5, 5.0)
    no_repeat_ngram_size = _clamp(int(params.get("no_repeat_ngram_size", 4)), 0, 10)
    penalty_window = _clamp(int(params.get("penalty_window", 64)), 0, int(CFG.block_size))
    max_new_tokens = _clamp(int(params.get("max_new_tokens", 600)), 16, 1024)
    min_new_tokens = _clamp(int(params.get("min_new_tokens", 200)), 0, max_new_tokens)
    clean_ending = bool(params.get("clean_ending", True))
    seed = params.get("seed", None)

    if seed in ("", None):
        seed = None
    else:
        seed = int(seed)
    if seed is not None:
        torch.manual_seed(seed)

    ids = TOKENIZER.encode(prompt).ids or [TOKENIZER.token_to_id("<|unk|>")]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.amp.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
        out_ids = MODEL.generate(
            x, max_new_tokens, temperature,
            top_k if top_k > 0 else None,
            top_p if top_p < 1.0 else None,
            repetition_penalty, no_repeat_ngram_size,
            penalty_window=penalty_window, min_new_tokens=min_new_tokens, eos_id=EOT_ID,
        )
    ids_list = out_ids[0].tolist()
    natural_end = (len(ids_list) > 0 and ids_list[-1] == EOT_ID)
    text = TOKENIZER.decode(ids_list).replace(EOT, "").strip()
    if clean_ending and not natural_end:
        text = _trim_cut_off(text)
    return text


# --------------------------------------------------------------------------- #
#  Web UI (HTML/CSS/JS)                                                       #
# --------------------------------------------------------------------------- #
HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LyricsGPT Studio</title>
<style>
  :root{
    --bg0:#000000; --bg1:#000000; --card:rgba(255,255,255,.045);
    --border:rgba(255,255,255,.09); --txt:#ececf4; --muted:#9a9ab2;
    --a1:#ec4899; --a2:#ec4899; --ok:#34d399; --err:#fb7185;
    --radius:18px;
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0}
  body{
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    background:var(--bg0); color:var(--txt); min-height:100vh; line-height:1.5;
  }
  .bg{position:fixed; inset:0; z-index:-1; background:#000000;}
  .container{max-width:880px; margin:0 auto; padding:40px 20px 80px}
  header{text-align:center; margin-bottom:28px}
  h1{font-size:38px; margin:0 0 6px; letter-spacing:-.5px; color:var(--a2)}
  h1 .grad{color:inherit}
  .subtitle{color:var(--muted); margin:0 0 12px; font-size:15px}
  .badge{display:inline-block; font-size:12.5px; color:var(--muted);
    background:var(--card); border:1px solid var(--border); padding:6px 14px; border-radius:999px}
  .badge b{color:var(--txt)}
  .card{background:var(--card); border:1px solid var(--border); border-radius:var(--radius);
    padding:24px; backdrop-filter:blur(10px); box-shadow:0 10px 40px rgba(0,0,0,.35); margin-bottom:20px}
  label.field-label{display:block; font-size:13px; color:var(--muted); margin-bottom:8px; font-weight:600; letter-spacing:.3px}
  textarea{width:100%; min-height:64px; resize:vertical; padding:14px 16px; border-radius:12px;
    border:1px solid var(--border); background:rgba(0,0,0,.25); color:var(--txt); font-size:16px; font-family:inherit}
  textarea:focus{outline:none; border-color:var(--a1); box-shadow:0 0 0 3px rgba(236,72,153,.18)}
  .controls{display:grid; grid-template-columns:1fr 1fr; gap:18px 26px; margin-top:18px}
  .control-head{display:flex; justify-content:space-between; align-items:center; margin-bottom:8px}
  .control-head .name{font-size:12.5px; color:var(--muted); font-weight:600}
  .control-head .val{font-size:12.5px; color:var(--txt); font-variant-numeric:tabular-nums; background:rgba(255,255,255,.06); padding:1px 8px; border-radius:6px}
  input[type=range]{ -webkit-appearance:none; appearance:none; width:100%; height:6px; border-radius:999px;
    background:rgba(255,255,255,.13); outline:none}
  input[type=range]::-webkit-slider-thumb{ -webkit-appearance:none; width:18px; height:18px; border-radius:50%;
    background:var(--a2); cursor:pointer; box-shadow:0 0 0 4px rgba(236,72,153,.15)}
  input[type=range]::-moz-range-thumb{ width:18px; height:18px; border:none; border-radius:50%;
    background:var(--a2); cursor:pointer}
  details.advanced{margin-top:18px; border-top:1px solid var(--border); padding-top:14px}
  details.advanced summary{cursor:pointer; color:var(--muted); font-size:13.5px; font-weight:600; list-style:none; user-select:none}
  details.advanced summary::-webkit-details-marker{display:none}
  details.advanced summary::before{content:"▸ "; color:var(--a2)}
  details.advanced[open] summary::before{content:"▾ "}
  .seed-row{display:flex; align-items:center; gap:12px}
  .seed-row input[type=number]{flex:1; padding:9px 12px; border-radius:10px; border:1px solid var(--border);
    background:rgba(0,0,0,.25); color:var(--txt); font-size:14px}
  .seed-row input[type=number]:focus{outline:none; border-color:var(--a1)}
  .check-row{display:flex; align-items:center; gap:10px; font-size:13px; color:var(--muted); margin-top:6px}
  .check-row input{width:16px; height:16px; accent-color:var(--a1)}
  .btn{border:none; cursor:pointer; border-radius:12px; font-size:15px; font-weight:600; font-family:inherit; transition:.15s}
  .btn-primary{width:100%; margin-top:22px; padding:15px; color:#fff;
    background:var(--a2); box-shadow:0 8px 24px rgba(236,72,153,.35)}
  .btn-primary:hover{filter:brightness(1.08); transform:translateY(-1px)}
  .btn-primary:disabled{opacity:.6; cursor:default; transform:none}
  .btn-ghost{padding:9px 14px; color:var(--muted); background:rgba(255,255,255,.05); border:1px solid var(--border); font-size:13px}
  .btn-ghost:hover{color:var(--txt); background:rgba(255,255,255,.1)}
  .spinner{display:inline-block; width:15px; height:15px; border:2px solid rgba(255,255,255,.35);
    border-top-color:#fff; border-radius:50%; animation:spin .7s linear infinite; vertical-align:-2px; margin-right:8px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .output-head{display:flex; justify-content:space-between; align-items:center; margin-bottom:14px}
  .output-title{font-size:13px; color:var(--muted); font-weight:600; letter-spacing:.3px}
  .output-actions{display:flex; gap:8px}
  .output{white-space:pre-wrap; word-wrap:break-word; font-family:Georgia,"Times New Roman",serif;
    font-size:18px; line-height:1.75; color:#f3f3f9}
  .error{background:rgba(251,113,133,.1); border:1px solid rgba(251,113,133,.4); color:#ffd5de;
    padding:14px 18px; border-radius:12px; font-size:14px; margin-bottom:20px}
  footer{text-align:center; color:var(--muted); font-size:12px; margin-top:30px; opacity:.7}
  @media (max-width:560px){ .controls{grid-template-columns:1fr} h1{font-size:30px} }
</style>
</head>
<body>
<div class="bg"></div>
<main class="container">
  <header>
    <h1>LyricsGPT <span class="grad">Studio</span></h1>
    <p class="subtitle">Give it a line. Get a whole song.</p>
    <div id="info" class="badge">loading model…</div>
  </header>

  <section class="card">
    <label class="field-label" for="prompt">STARTING LINE / PROMPT</label>
    <textarea id="prompt" placeholder="e.g. Late at night I think of you">Late at night I think of you</textarea>

    <div class="controls">
      <div class="control">
        <div class="control-head"><span class="name">Creativity (temperature)</span><span class="val" id="v-temperature">0.90</span></div>
        <input type="range" id="temperature" min="0.5" max="1.5" step="0.05" value="0.9">
      </div>
      <div class="control">
        <div class="control-head"><span class="name">Max length</span><span class="val" id="v-max">600</span></div>
        <input type="range" id="max_new_tokens" min="200" max="800" step="50" value="600">
      </div>
      <div class="control">
        <div class="control-head"><span class="name">Min length</span><span class="val" id="v-min">200</span></div>
        <input type="range" id="min_new_tokens" min="0" max="400" step="20" value="200">
      </div>
      <div class="control">
        <div class="control-head"><span class="name">Repetition penalty</span><span class="val" id="v-rep">1.15</span></div>
        <input type="range" id="repetition_penalty" min="1" max="2" step="0.05" value="1.15">
      </div>
    </div>

    <details class="advanced">
      <summary>Advanced settings</summary>
      <div class="controls">
        <div class="control">
          <div class="control-head"><span class="name">Top-k</span><span class="val" id="v-topk">50</span></div>
          <input type="range" id="top_k" min="0" max="100" step="5" value="50">
        </div>
        <div class="control">
          <div class="control-head"><span class="name">Top-p (nucleus)</span><span class="val" id="v-topp">0.95</span></div>
          <input type="range" id="top_p" min="0.5" max="1" step="0.05" value="0.95">
        </div>
        <div class="control">
          <div class="control-head"><span class="name">Penalty window</span><span class="val" id="v-win">64</span></div>
          <input type="range" id="penalty_window" min="0" max="256" step="16" value="64">
        </div>
        <div class="control">
          <div class="control-head"><span class="name">No-repeat n-gram</span><span class="val" id="v-ngram">4</span></div>
          <input type="range" id="no_repeat_ngram_size" min="0" max="6" step="1" value="4">
        </div>
      </div>
      <div class="controls" style="margin-top:14px">
        <div class="seed-row">
          <span class="name" style="font-size:12.5px;color:var(--muted);font-weight:600;white-space:nowrap">Seed</span>
          <input type="number" id="seed" placeholder="random">
        </div>
        <div class="check-row">
          <input type="checkbox" id="clean_ending" checked>
          <label for="clean_ending">Trim cut-off endings to a clean verse boundary</label>
        </div>
      </div>
    </details>

    <button id="generate" class="btn btn-primary">Generate lyrics</button>
  </section>

  <div id="error" class="error" hidden></div>

  <section class="card" id="output-card" hidden>
    <div class="output-head">
      <span class="output-title">YOUR SONG</span>
      <div class="output-actions">
        <button id="regenerate" class="btn btn-ghost">Regenerate</button>
        <button id="copy" class="btn btn-ghost">Copy</button>
      </div>
    </div>
    <div id="output" class="output"></div>
  </section>

  <footer>Runs entirely on your machine. Generated by a model trained from scratch with LyricsGPT.</footer>
</main>

<script>
const $ = (id) => document.getElementById(id);
const floaters = ["temperature","max_new_tokens","min_new_tokens","repetition_penalty","top_k","top_p","penalty_window","no_repeat_ngram_size"];

function bindLabels(){
  floaters.forEach(id=>{
    const el = $(id), lbl = $("v-"+id);
    const fmt = (v)=> el.step && el.step.includes(".") ? Number(v).toFixed(2) : v;
    el.addEventListener("input", ()=>{ lbl.textContent = fmt(el.value); });
  });
}
bindLabels();

function params(useSeed){
  const seedRaw = $("seed").value.trim();
  return {
    prompt: $("prompt").value,
    temperature: parseFloat($("temperature").value),
    max_new_tokens: parseInt($("max_new_tokens").value),
    min_new_tokens: parseInt($("min_new_tokens").value),
    repetition_penalty: parseFloat($("repetition_penalty").value),
    top_k: parseInt($("top_k").value),
    top_p: parseFloat($("top_p").value),
    penalty_window: parseInt($("penalty_window").value),
    no_repeat_ngram_size: parseInt($("no_repeat_ngram_size").value),
    clean_ending: $("clean_ending").checked,
    seed: useSeed ? seedRaw : ""
  };
}

function showOutput(text){
  $("output").textContent = text;
  $("output-card").hidden = false;
}

async function generate(useSeed){
  const btn = $("generate");
  $("error").hidden = true;
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>Generating…';
  try{
    const res = await fetch("/generate", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify(params(useSeed))
    });
    const data = await res.json();
    if(!res.ok || data.error){ throw new Error(data.error || ("HTTP "+res.status)); }
    showOutput(data.lyrics || "(empty)");
  }catch(e){
    const box = $("error"); box.hidden = false; box.textContent = "Error: " + e.message;
  }finally{
    btn.disabled = false; btn.innerHTML = "Generate lyrics";
  }
}

$("generate").addEventListener("click", ()=>generate(false));
$("regenerate").addEventListener("click", ()=>generate(false));
$("copy").addEventListener("click", async ()=>{
  const t = $("output").textContent;
  try{ await navigator.clipboard.writeText(t); }
  catch(e){ const ta=document.createElement("textarea"); ta.value=t; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove(); }
});

(async ()=>{
  try{
    const r = await fetch("/info");
    $("info").textContent = r.ok ? "Connected" : "not connected";
  }catch(e){ $("info").textContent = "not connected"; }
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
#  HTTP server                                                                #
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the console clean

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code, obj):
        self._send(code, json.dumps(obj), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML_PAGE, "text/html; charset=utf-8")
        elif self.path == "/info":
            self._send_json(200, {
                "params": round(N_PARAMS / 1e6, 1),
                "device": ("GPU: " + torch.cuda.get_device_name(0)) if DEVICE == "cuda" else "CPU",
                "val_loss": VAL_LOSS,
                "context": CFG.block_size,
            })
        elif self.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, '{"error":"not found"}', "application/json")

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8")
            body = json.loads(raw)
        except Exception:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        try:
            text = generate_lyrics(body)
            self._send_json(200, {"lyrics": text})
        except Exception as e:
            self._send_json(500, {"error": str(e)})


def main():
    host = HOST
    port = PORT
    if "--port" in sys.argv:
        try:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        except Exception:
            pass
    url = f"http://{host}:{port}"
    print("\n" + "=" * 54)
    print("  LyricsGPT Studio is running")
    print("  Open this URL in your browser:")
    print("    " + url)
    print("=" * 54 + "\n")
    try:
        ThreadingHTTPServer((host, port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
