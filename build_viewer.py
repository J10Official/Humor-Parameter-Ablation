"""
build_viewer.py — generates viewer.html with all joke data embedded.
Usage: python3 build_viewer.py

── Configuration (edit here) ──────────────────────────────────────────────────
"""
import json
import random
from pathlib import Path

# Which themes to include (1-indexed, inclusive)
START = 11
END   = 50

# Which models to show (order is shuffled per-theme anyway)
VISIBLE_MODELS = [ "32B", "30A3", "235A22"]

ALL_MODELS  = ["8B", "14B", "32B", "30A3", "235A22"]
RESULTS_DIR = Path("results")
LABELS      = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ── Load data ────────────────────────────────────────────────────────────────
per_model: dict[str, dict] = {}
for key in ALL_MODELS:
    path = RESULTS_DIR / f"{key}.json"
    if not path.exists():
        raise FileNotFoundError(f"results/{key}.json not found")
    with open(path) as f:
        per_model[key] = json.load(f)

all_themes = [t for t in per_model[ALL_MODELS[0]] if all(t in per_model[m] for m in ALL_MODELS)]

# Slice to START..END (1-indexed, inclusive)
themes = all_themes[START - 1 : END]
if not themes:
    raise ValueError(f"No themes in range {START}–{END} (total: {len(all_themes)})")

# For each theme: deterministically shuffle VISIBLE_MODELS.
# Single RNG seeded with 42 shuffles themes in order — fully reproducible.
# Labels A, B, C… assigned by shuffle position.
# Mapping is written once and never overwritten (so label assignments are stable).
data: dict[str, list[str]] = {}
mapping: dict[str, list[str]] = {}   # theme -> [model at A, model at B, ...]

rng = random.Random(42)
for t in themes:
    order = VISIBLE_MODELS[:]
    rng.shuffle(order)
    data[t]    = [per_model[m][t] for m in order]
    mapping[t] = order

# Save mapping only once — never overwrite so label assignments stay stable
mapping_path = RESULTS_DIR / "viewer_label_mapping.json"
if not mapping_path.exists():
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    print(f"  Label mapping saved → {mapping_path}")
else:
    print(f"  Label mapping already exists, skipping → {mapping_path}")

data_js   = json.dumps(data,   ensure_ascii=False, separators=(",", ":"))
themes_js = json.dumps(themes, ensure_ascii=False)
n_jokes   = len(VISIBLE_MODELS)
labels_js = json.dumps(LABELS[:n_jokes], ensure_ascii=False)

# ── HTML ─────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Humor Viewer</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  font-family: 'Georgia', serif;
  min-height: 100vh;
  background: #f7f5f0;
  color: #1a1a1a;
  display: flex;
  flex-direction: column;
}}

/* ── Top bar ── */
#topbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1.6rem;
  border-bottom: 1px solid #ddd9d0;
  gap: 1rem;
  position: sticky;
  top: 0;
  background: #f7f5f0;
  z-index: 10;
}}

#theme-label {{
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  text-align: center;
  flex: 1;
  color: #1a1a1a;
}}

#counter {{
  font-size: 1.1rem;
  color: #666;
  text-align: center;
  margin-top: 0.2rem;
  font-weight: 600;
}}

/* in mode 2 hide the single theme label (each half shows its own) */
body.mode2 #theme-label {{ display: none; }}

.nav-btn {{
  background: none;
  border: 1px solid #ccc;
  color: #555;
  border-radius: 5px;
  padding: 0.4rem 1rem;
  font-size: 0.82rem;
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s;
  white-space: nowrap;
  font-family: inherit;
}}
.nav-btn:hover {{ border-color: #888; color: #1a1a1a; }}
.nav-btn:disabled {{ opacity: 0.2; cursor: default; }}

#mode-btn, #play-btn {{
  background: none;
  border: 1px solid #ccc;
  color: #888;
  border-radius: 5px;
  padding: 0.35rem 0.75rem;
  font-size: 0.65rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s;
  font-family: 'Courier New', monospace;
  white-space: nowrap;
}}
#mode-btn:hover, #play-btn:hover {{ border-color: #888; color: #1a1a1a; }}

#timer {{
  font-size: 1.1rem;
  font-weight: 700;
  font-family: 'Courier New', monospace;
  color: #1a1a1a;
  min-width: 3.5rem;
  text-align: center;
}}
#timer.running {{ color: #c53030; }}

/* ── MODE 1: single column ── */
#stage {{
  flex: 1;
  padding: 2.5rem 5rem;
  display: flex;
  flex-direction: column;
}}

.joke-block {{
  padding: 1.8rem 0;
  border-bottom: 1px solid #e5e1d8;
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}}
.joke-block:last-child {{ border-bottom: none; }}

.joke-label {{
  font-size: 1.1rem;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: #888;
  font-family: 'Courier New', monospace;
  font-weight: 700;
}}

.joke-text {{
  font-size: 1.5rem;
  line-height: 1.75;
  color: #1a1a1a;
  font-weight: 400;
}}

/* ── MODE 2: split screen ── */
body.mode2 #stage {{
  flex-direction: row;
  padding: 0;
  gap: 0;
  align-items: stretch;
}}

.half {{
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}}

.half-left  {{ background: #fef08a; border-right: 2px solid #d4b800; }}
.half-right {{ background: #93c5fd; }}

.half-theme-label {{
  font-size: 1.6rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-align: center;
  color: #1a1a1a;
  padding: 0.75rem 1rem 0.25rem;
  font-family: 'Georgia', serif;
  text-transform: capitalize;
}}

.half-jokes {{
  flex: 1;
  padding: 1.5rem 3.5rem 2rem;
  display: flex;
  flex-direction: column;
  justify-content: center;
}}

body.mode2 .joke-block {{
  border-bottom: 1px solid rgba(0,0,0,0.1);
  padding: 1.4rem 0;
}}
body.mode2 .joke-block:last-child {{ border-bottom: none; }}
body.mode2 .joke-text {{ font-size: 1.7rem; line-height: 1.8; }}
body.mode2 .joke-label {{ color: #333; font-size: 1.2rem; }}

@media (max-width: 760px) {{
  #stage {{ padding: 1.5rem; }}
  .joke-text {{ font-size: 1.1rem; }}
  body.mode2 #stage {{ flex-direction: column; }}
  .half-left {{ border-right: none; border-bottom: 2px solid #d4b800; }}
  .half-jokes {{ padding: 1.5rem; justify-content: flex-start; }}
}}
</style>
</head>
<body>

<div id="topbar">
  <button class="nav-btn" id="prev-btn" onclick="navigate(-1)">&#8592; Prev</button>
  <div style="flex:1;text-align:center;min-width:0">
    <div id="theme-label"></div>
    <div id="counter"></div>
  </div>
  <button class="nav-btn" id="next-btn" onclick="navigate(1)">Next &#8594;</button>
  <span id="timer">1:00</span>
  <button id="play-btn" onclick="togglePlay()">&#9654; Play</button>
  <button id="mode-btn" onclick="toggleMode()">Mode&nbsp;2</button>
</div>

<div id="stage"></div>

<script>
const THEMES = {themes_js};
const DATA   = {data_js};
const LABELS = {labels_js};

// Split themes into two groups for Mode 2
const SPLIT    = Math.floor(THEMES.length / 2);
const THEMES_L = THEMES.slice(0, SPLIT);
const THEMES_R = THEMES.slice(SPLIT);
const SPLIT_LEN = Math.min(THEMES_L.length, THEMES_R.length);

let current   = 0;   // shared index (Mode 1 = all themes, Mode 2 = step within each group)
let mode      = 1;

// ── Timer / auto-advance ───────────────────────────────────────────────────
const INTERVAL_SECS = 60;   // seconds per theme
let timerRunning = false;
let secondsLeft  = INTERVAL_SECS;
let timerHandle  = null;

function fmtTime(s) {{
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return m + ':' + String(sec).padStart(2, '0');
}}

function updateTimerUI() {{
  const el = document.getElementById('timer');
  el.textContent = fmtTime(secondsLeft);
  el.classList.toggle('running', timerRunning);
  document.getElementById('play-btn').innerHTML =
    timerRunning ? '&#9646;&#9646; Pause' : '&#9654; Play';
}}

function timerTick() {{
  secondsLeft--;
  if (secondsLeft <= 0) {{
    // auto-advance
    const maxIdx = mode === 2 ? SPLIT_LEN - 1 : THEMES.length - 1;
    if (current < maxIdx) {{
      current++;
      render();
      window.scrollTo(0, 0);
    }} else {{
      // reached the end — stop
      timerRunning = false;
      clearInterval(timerHandle);
      timerHandle = null;
    }}
    secondsLeft = INTERVAL_SECS;
  }}
  updateTimerUI();
}}

function togglePlay() {{
  timerRunning = !timerRunning;
  if (timerRunning) {{
    secondsLeft = INTERVAL_SECS;
    timerHandle = setInterval(timerTick, 1000);
  }} else {{
    clearInterval(timerHandle);
    timerHandle = null;
    secondsLeft = INTERVAL_SECS;
  }}
  updateTimerUI();
}}

function navigate(dir) {{
  const maxIdx = mode === 2 ? SPLIT_LEN - 1 : THEMES.length - 1;
  current = Math.max(0, Math.min(maxIdx, current + dir));
  if (timerRunning) secondsLeft = INTERVAL_SECS;   // reset countdown on manual nav
  updateTimerUI();
  render();
  window.scrollTo(0, 0);
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft')  navigate(-1);
  if (e.key === 'ArrowRight') navigate(1);
}});

function toggleMode() {{
  mode = mode === 1 ? 2 : 1;
  current = 0;   // reset position when switching modes
  if (timerRunning) secondsLeft = INTERVAL_SECS;   // reset countdown
  updateTimerUI();
  document.getElementById('mode-btn').textContent = mode === 1 ? 'Mode\u00a02' : 'Mode\u00a01';
  document.body.classList.toggle('mode2', mode === 2);
  render();
}}

function esc(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function makeBlock(label, joke) {{
  return `<div class="joke-block">
    <div class="joke-label">Joke ${{esc(label)}}</div>
    <div class="joke-text">${{esc(joke)}}</div>
  </div>`;
}}

// ── Render one Mode-2 half in-place ────────────────────────────────────────
function renderHalf(side, themeList, idx) {{
  const theme = themeList[idx];
  const jokes = DATA[theme];
  const half  = document.getElementById('half-' + side);
  if (!half) return;

  half.querySelector('.half-theme-label').textContent = theme;

  const jokesDiv = half.querySelector('.half-jokes');
  jokesDiv.innerHTML = '';
  jokes.forEach((joke, i) => jokesDiv.insertAdjacentHTML('beforeend', makeBlock(LABELS[i], joke)));
}}

// ── Main render ─────────────────────────────────────────────────────────────
function render() {{
  const stage = document.getElementById('stage');
  stage.innerHTML = '';

  if (mode === 1) {{
    const theme = THEMES[current];
    const jokes = DATA[theme];

    document.getElementById('theme-label').textContent = theme;
    document.getElementById('counter').textContent     = (current + 1) + '\u00a0/\u00a0' + THEMES.length;
    document.getElementById('prev-btn').disabled       = current === 0;
    document.getElementById('next-btn').disabled       = current === THEMES.length - 1;

    jokes.forEach((joke, i) => stage.insertAdjacentHTML('beforeend', makeBlock(LABELS[i], joke)));

  }} else {{
    // Shared counter in topbar
    document.getElementById('counter').textContent     = (current + 1) + '\u00a0/\u00a0' + SPLIT_LEN;
    document.getElementById('prev-btn').disabled = current === 0;
    document.getElementById('next-btn').disabled = current === SPLIT_LEN - 1;

    // Build left half
    const left = document.createElement('div');
    left.className = 'half half-left';
    left.id = 'half-left';
    left.innerHTML = '<div class="half-theme-label"></div><div class="half-jokes"></div>';
    stage.appendChild(left);

    // Build right half
    const right = document.createElement('div');
    right.className = 'half half-right';
    right.id = 'half-right';
    right.innerHTML = '<div class="half-theme-label"></div><div class="half-jokes"></div>';
    stage.appendChild(right);

    renderHalf('left',  THEMES_L, current);
    renderHalf('right', THEMES_R, current);
  }}
}}

render();
</script>
</body>
</html>
"""

out = Path("viewer.html")
out.write_text(html, encoding="utf-8")
print(f"✓  viewer.html  (themes {START}–{END}, {len(themes)} sets, {n_jokes} jokes each, labels A–{LABELS[n_jokes-1]})")
