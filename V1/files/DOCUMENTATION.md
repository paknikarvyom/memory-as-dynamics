# Documentation

Technical reference for `memory_as_dynamics.py`: architecture, module-by-module
design, and how each function maps to the source paper.

---

## 1. Architecture overview

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐     ┌───────────────────────┐
│  scrape_url  │ --> │ build_json_record │ --> │   BipolarEncoder    │ --> │ DynamicMemoryNetwork   │
│ (requests +  │     │  (JSON chunks)     │     │ (text -> ±1 pattern)│     │ (PyTorch attractor +   │
│ BeautifulSoup)│     └──────────────────┘     └────────────────────┘     │ sequence memory)        │
└─────────────┘                                                          └───────────┬────────────┘
                                                                                       │
                                                                          ┌────────────▼────────────┐
                                                                          │   QtPy Desktop UI        │
                                                                          │ (Build / Chunks-JSON /   │
                                                                          │  Recall / Replay /       │
                                                                          │  Diagnostics tabs)       │
                                                                          └──────────────────────────┘
```

The whole pipeline lives in one file, organized into six numbered sections:

1. Scraping → JSON
2. Text → distributed bipolar pattern encoding
3. The recurrent memory network (the paper's core model)
4. Offline self-test / headless runner
5. QtPy desktop UI
6. CLI entry point

---

## 2. Scraping → JSON (`scrape_url`, `build_json_record`)

`scrape_url(url, max_chunks=48, min_chars=20, max_chars=400, timeout=15, html_override=None)`

- Fetches the page with a `requests.get` call (custom User-Agent, 15s timeout).
- Strips non-content tags (`script`, `style`, `nav`, `footer`, `header`, `svg`, `form`).
- Walks the document in order, pulling text out of `h1`–`h4`, `p`, and `li`
  tags — this preserves **reading order**, which the memory later stores as a
  sequence.
- Drops very short fragments (`min_chars`) and truncates very long ones
  (`max_chars`), and caps the total number of chunks (`max_chunks`) so the
  resulting network stays a reasonable size.
- `html_override` lets you bypass the network entirely (used by `--selftest`).

`build_json_record(url, scraped)` wraps the result into the JSON structure
that's stored, displayed, and can be saved from the UI:

```json
{
  "url": "https://example.com/article",
  "scraped_at": "2026-09-23T10:00:00+00:00",
  "title": "Article title",
  "num_chunks": 12,
  "chunks": [
    {"id": 0, "order": 0, "tag": "h1", "text": "..."},
    {"id": 1, "order": 1, "tag": "p",  "text": "..."}
  ]
}
```

---

## 3. Text → pattern encoding (`BipolarEncoder`)

The paper's model operates on abstract patterns `ξ ∈ {-1, +1}^N` (Table 2). To
turn arbitrary scraped text into such patterns without adding a heavyweight
embedding dependency, `BipolarEncoder` uses a **random-projection / feature
hashing** scheme:

- Every distinct word is assigned a fixed pseudo-random ±1 vector of length
  `dim`, derived deterministically from an MD5 hash of the word (so the same
  word always maps to the same vector, across runs and sessions).
- A chunk's pattern is the **element-wise sign of the sum** of its words'
  vectors (ties broken to `+1`).
- `BipolarEncoder.corrupt(pattern, noise_frac, rng)` flips a fraction of bits
  at random — this builds the noisy recall cues used in the *Attractor
  Recall* tab and corresponds to Task A ("pattern recall") in the paper's
  evaluation plan (Table 4).

This gives a distributed code where chunks sharing vocabulary land closer
together in state space (positive overlap), and unrelated chunks are closer
to orthogonal — the property the paper's attractor and sequence machinery
assumes of its patterns.

---

## 4. The recurrent memory network (`DynamicMemoryNetwork`)

This class is a direct, faithful implementation of Sections 3–4 of the paper.

### 4.1 State and parameters

| Attribute | Meaning | Paper reference |
|---|---|---|
| `N` | Number of neurons / pattern dimensionality | Table 2 |
| `beta` | Activation gain in `φ(z) = tanh(βz)` | Table 2, Eq. 1 |
| `gamma` | Homeostatic weight-decay rate | Eq. 7 |
| `gmax`, `period`, `theta_g`, `kappa` | Gate amplitude, period, threshold, sharpness | Eq. 12 |
| `W_A` | Symmetric Hebbian attractor matrix | Eq. 5/8, Sec. 4.5 |
| `W_T` | Asymmetric successor–predecessor matrix | Eq. 6/9 |
| `b` | Bias vector | Eq. 1 |
| `patterns`, `labels` | Stored patterns and their human-readable source text | — |

### 4.2 Storage (learning)

- **`store_patterns(patterns, labels)`** — Eq. 5/8 in closed form (Sec. 4.5):

  ```
  W_A = (1/N) · Ξᵀ Ξ,    Ξ = stacked patterns (P × N)
  ```

  The diagonal is zeroed (standard "no self-connection" Hopfield variant),
  then homeostatic decay `(1 - γ)` is applied (Eq. 7).

- **`store_sequence(order=None, cyclic=False)`** — Eq. 9:

  ```
  W_T = (1/N) · Σ_{i=1}^{k-1} x_{i+1} x_iᵀ
  ```

  By default, the sequence order is the reading order the chunks were
  scraped in. `cyclic=True` adds a closing term `x_1 x_kᵀ` to loop the
  sequence.

### 4.3 Dynamics

- **`gate(t)`** — Eq. 12, a periodic, theta-like signal:

  ```
  g(t) = gmax · σ( κ · (cos(2πt/T) - θg) )
  ```

- **`step(x, gate_val, u=None)`** — Eq. 1 / Eq. 11:

  ```
  x(t+1) = tanh( β · (W_A x(t) + g(t)·W_T x(t) + U u(t) + b) )
  ```

  With `gate_val = 0`, the network is a pure symmetric attractor network
  (holds a memory). With `gate_val > 0`, the asymmetric term pushes the
  state toward the next item in the sequence.

- **`energy(x)`** — a Hopfield-style energy of the symmetric part only
  (`-½xᵀW_A x - bᵀx`). Only meaningful when the gate is closed; the paper
  notes the gated dynamics have no Lyapunov function once `W_T` is active
  (Sec. 3.3 / 8).

- **`overlap(x, pattern)`** — normalized dot product, the `m_μ` of Table 2.

- **`best_match(x)`** — the stored pattern with highest overlap to `x`, used
  throughout the UI to say "which memory is this state closest to right now."

### 4.4 Recall and replay

- **`recall(cue, steps, gate_val=0.0, tol=1e-4)`** — free-runs `step()` from a
  (typically noisy) cue for up to `steps` iterations, closing the gate so the
  network behaves as a pure attractor network. Returns the overlap trace, the
  energy trace, the final best-matching pattern, and the step at which the
  state stopped changing (if it did). This implements **Task A** (pattern
  recall) from the paper's evaluation plan (Table 4).

- **`replay_sequence(steps, x0=None)`** — free-runs the *gated* dynamics
  (Eq. 11–12) starting from the first item in the stored sequence (or a given
  `x0`), recording the gate value and the best-matching chunk at every step.
  This implements **Task B** (sequence recall).

### 4.5 Diagnostics

- **`spectral_radius_at(x)`** — Eq. 3/4: the spectral radius of the Jacobian
  `J = diag(φ'(a))·W_A` at a state `x`. Should be below 1 at a genuinely
  stable stored pattern.

- **`capacity_estimates()`** — Eq. 14/15:

  ```
  Pmax ≈ 0.14 · N                     (Hebbian pattern capacity)
  Smax ≈ N / (c · ln N),  c ∈ [2, 4]  (error-free sequence capacity)
  ```

- **`energy_proxy(traj, e_spike=1.0)`** — Eq. 16, an event-based energy
  estimate using firing probability `s_i = (1 + x_i)/2`.

### 4.6 Persistence

`to_state()` / `from_state()` serialize the network's weights, bias,
patterns, and labels to/from a plain dict, which the UI saves and loads as a
`.ptmem` file via `torch.save` / `torch.load`.

---

## 5. QtPy desktop UI (`_launch_gui`)

Built entirely with `qtpy` (so it works with whichever Qt binding —
PyQt5/6 or PySide2/6 — is installed) plus `matplotlib`'s Qt backend for
charts.

### Layout

- **Top bar**: URL field, neuron-count spin box, "store sequence" checkbox,
  and the **Scrape & Build Memory** button.
- **Log console**: streams progress messages from the background worker.
- **Tabs**:
  1. **Chunks / JSON** — a list of stored chunks on the left; the full JSON
     record on the right, with buttons to save the JSON, and to save/load the
     trained memory as a `.ptmem` file.
  2. **Attractor Recall** — pick a stored chunk, set a noise level, run
     recall, and see the overlap and energy traces plus a text summary of
     whether the correct chunk was recovered.
  3. **Sequence Replay** — replay the gated dynamics and see the gate signal
     and the retrieved chunk index over time, plus a text log of transitions.
  4. **Network Diagnostics** — capacity estimates, spectral radius at a
     stored pattern, and a heatmap of `W_A`.

### Threading

Scraping and building the memory happens on a `QThread` subclass
(`BuildWorker`) so the UI never freezes while waiting on the network request;
it emits `progress`, `finished_ok`, and `failed` signals. Recall and replay
computations are fast (small matrix–vector products) and run synchronously on
the UI thread when you click their buttons.

---

## 6. CLI entry points

`memory_as_dynamics.py` supports three modes (see `USAGE.md` for full
detail):

| Flag | Behavior |
|---|---|
| *(none)* | Launches the QtPy GUI. |
| `--selftest` | Runs the entire pipeline offline against a bundled sample page — no network, no GUI. |
| `--headless URL` | Runs the pipeline against a real URL and prints results to the console — no GUI. |
| `--neurons N` | Sets the network size for `--headless` / `--selftest` (default 256). |

---

## 7. Known limitations (by design, and inherited from the paper)

- **Capacity**: natural-language chunks are correlated, not independent
  random patterns, so realized recall accuracy is well below the
  `Pmax ≈ 0.14N` estimate — the same crosstalk effect the paper analyzes in
  Section 4.7 and Appendix A.2.
- **Synchronous updates**: `step()` updates all neurons in parallel each
  tick. Classic Hopfield-style analysis assumes asynchronous updates for a
  guaranteed monotone energy decrease; synchronous updates can occasionally
  cycle between two states instead of settling, which the UI reports as "did
  not fully converge."
- **No dendritic extension**: Eq. 13 (multi-branch dendritic subunits) is
  described in the paper but not implemented here; the network uses
  point-neuron dynamics (`K = 1` in Eq. 13's notation).
