# Usage Guide

Step-by-step instructions for running `memory_as_dynamics.py`.

---

## 1. Install dependencies

```bash
pip install torch numpy requests beautifulsoup4 matplotlib qtpy PyQt5
```

You can swap `PyQt5` for `PySide2`, `PyQt6`, or `PySide6` — `qtpy` will use
whichever Qt binding is installed.

> If you only want to try the pipeline logic and don't need the desktop UI
> yet, you can skip `qtpy`/`PyQt5` for now and use `--selftest` or
> `--headless` below.

---

## 2. Verify the install (no network, no UI)

```bash
python memory_as_dynamics.py --selftest
```

This runs the full pipeline — scrape (a bundled sample page), build JSON,
encode patterns, store them, recall from a noisy cue, replay the sequence —
and prints a short report. If this finishes with `Self-test OK.`, your
PyTorch/NumPy setup is working correctly.

---

## 3. Try a real URL from the command line (no UI)

```bash
python memory_as_dynamics.py --headless https://en.wikipedia.org/wiki/Hopfield_network
```

Optional flags:

```bash
python memory_as_dynamics.py --headless <URL> --neurons 512
```

This prints the page title, chunk count, a JSON excerpt, a recall demo
result, and capacity estimates for the chosen network size — useful for
scripting or quick checks before opening the GUI.

---

## 4. Launch the desktop UI

```bash
python memory_as_dynamics.py
```

### Step 1 — Build a memory

1. Paste a URL into the **URL** field (defaults to a Wikipedia article on
   Hopfield networks, as an example).
2. Set **Neurons N** — the pattern dimensionality / network size. Larger `N`
   gives more capacity and more distinct patterns, at the cost of a slightly
   slower build. 256–512 is a good starting range for a typical article.
3. Leave **"Store reading-order sequence"** checked if you also want to
   explore sequence replay (tab 3); uncheck it if you only care about
   attractor recall.
4. Click **Scrape & Build Memory**. Progress appears in the log console
   above the tabs. When it finishes, the tabs populate automatically.

If the page has too little extractable text (e.g., it's mostly JavaScript-
rendered or has no headings/paragraphs/list items), you'll see an error —
try a different URL, or an article/blog-style page with real text content.

### Step 2 — Inspect the chunks and JSON

Open the **Chunks / JSON** tab:

- The left list shows every stored chunk in reading order, tagged by its
  source HTML element (`h1`, `p`, `li`, etc.).
- The right panel shows the exact JSON record the memory was built from.
- **Save JSON…** writes that record to a `.json` file.
- **Save Memory (.ptmem)…** saves the trained network (weights + patterns +
  the JSON record) so you can reload it later without re-scraping.
- **Load Memory (.ptmem)…** restores a previously saved memory.

### Step 3 — Run a recall

Open the **Attractor Recall** tab:

1. Pick a chunk from the list — this is the memory you'll try to recover.
2. Set the **cue noise** slider — the percentage of the pattern's bits that
   get randomly flipped before recall starts. Try 10–20% first, then push
   higher to see where recall starts to fail.
3. Set **max steps** if you want a longer or shorter run (60 is a reasonable
   default).
4. Click **Run Recall**.

You'll see:
- A text summary: which chunk was the cue, which chunk was recovered, whether
  it was a match, and when (if at all) the state stopped changing.
- An **overlap plot** — how close the state is to its best-matching stored
  chunk at each step (1.0 = exact match).
- An **energy plot** — the Hopfield-style energy of the state at each step;
  a well-behaved attractor run should trend downward.

If recall lands on the wrong chunk, that's expected sometimes — see
"Interpreting results" below.

### Step 4 — Replay the sequence

Open the **Sequence Replay** tab:

1. Set the number of **steps**.
2. Click **Replay Sequence**.

You'll see:
- The **gate signal** `g(t)` over time — this periodically opens and closes,
  alternately letting the network hold a memory or advance to the next one.
- The **retrieved chunk index** over time — a step plot showing which stored
  chunk the network's state is currently closest to.
- A text log below listing each transition ("t=12 -> #03 ...") so you can
  read off the order the network replayed the page's content in.

### Step 5 — Check diagnostics

Open the **Network Diagnostics** tab to see:

- The network size, number of stored chunks, and key parameters (`β`, `γ`).
- Estimated pattern capacity (`Pmax ≈ 0.14N`) and sequence capacity
  (`Smax ≈ N/(c ln N)`), compared against how many chunks you actually
  stored.
- The spectral radius at one stored pattern (should be below 1 for a stable
  attractor).
- A heatmap of the attractor weight matrix `W_A`.

---

## 5. Interpreting results

- **Recall sometimes lands on a neighboring chunk instead of the exact one.**
  This is expected: real text chunks share vocabulary and are therefore
  correlated, unlike the independent random patterns the paper's capacity
  formulas assume. If it happens a lot, try: a larger `N`, fewer stored
  chunks (a shorter page), or lower cue noise.
- **"Did not fully converge in the given steps."** The network updates all
  neurons in parallel (synchronous updates), which can occasionally cause
  slow oscillation instead of settling. Try more steps, or a lower noise
  level.
- **Sequence replay "gets stuck" on one chunk.** If the gate's `gmax` is too
  low relative to the attractor term, the asymmetric term can't overpower the
  attractor term and the sequence stalls — this matches the paper's own
  discussion in Section 4.5 ("gmax must exceed 1... with margin for
  crosstalk"). This build's defaults are tuned to avoid that, but very large
  `N` or very many chunks can still run into it.

---

## 6. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `ImportError` mentioning Qt on startup | No Qt binding installed. Run `pip install PyQt5` (or another binding). |
| "No readable text chunks... were found" | The page has little plain text in `h1–h4/p/li` tags (e.g., a JS-heavy app). Try a different URL. |
| Scrape fails with an HTTP error | Check the URL is reachable and correctly formed (`https://...`); some sites block non-browser user agents or require login, which this tool can't handle. |
| The UI window doesn't appear at all | Make sure you're not running with `QT_QPA_PLATFORM=offscreen` (that's for headless testing only) and that a display is available. |
| Recall/replay buttons do nothing | You need to click **Scrape & Build Memory** first — those tabs need a built memory to work with. |

---

## 7. Command-line reference

```
python memory_as_dynamics.py [--headless URL] [--selftest] [--neurons N]

  (no flags)        Launch the desktop UI.
  --selftest        Run an offline pipeline check (no network, no UI).
  --headless URL    Run the pipeline against URL and print results (no UI).
  --neurons N       Network size / pattern dimensionality (default: 256).
                     Applies to --selftest and --headless; in the UI, use
                     the "Neurons N" spin box instead.
```
