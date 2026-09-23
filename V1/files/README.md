# Memory as Dynamics — URL Memory Encoder

A single-file Python application that turns a web page into a **recurrent
attractor + sequence memory**, implementing the model from the paper *"Memory
as Dynamics: A Unified Model of Distributed Sequence Encoding in Recurrent
Neural Systems"* (Paknikar, 2026), and lets you explore it through a desktop
UI built with QtPy.

```
URL → scrape → JSON record → distributed patterns → recurrent memory (PyTorch) → QtPy UI
```

## What it does

1. **Scrapes** a URL for its readable text (headings, paragraphs, list items).
2. **Builds a JSON record** of the page: title, URL, timestamp, and an
   ordered list of text "chunks."
3. **Encodes** each chunk as a distributed bipolar (±1) pattern.
4. **Stores** the patterns in a recurrent network as:
   - stable **attractor memories** (Hebbian rule), and
   - a **sequence** of ordered transitions (temporal / successor–predecessor rule).
5. **Visualizes**, in a desktop UI:
   - the scraped chunks and the JSON record,
   - content-addressable recall from a noisy cue (overlap + energy curves),
   - gated sequence replay (which chunk the network "thinks of" over time),
   - capacity and stability diagnostics, plus a heatmap of the memory matrix.

## Files

| File | Purpose |
|---|---|
| `memory_as_dynamics.py` | The entire application — scraper, encoder, memory model, and QtPy UI, in one file. |
| `README.md` | This file — quick overview and setup. |
| `DOCUMENTATION.md` | Technical reference: architecture, classes, and how each part maps to the paper's equations. |
| `USAGE.md` | Step-by-step instructions for running the app in GUI, headless, and self-test modes. |

## Requirements

- Python 3.9+
- `torch`, `numpy`, `requests`, `beautifulsoup4`, `matplotlib`, `qtpy`, and one
  Qt binding (`PyQt5`, `PySide2`, `PyQt6`, or `PySide6`)

## Install

```bash
pip install torch numpy requests beautifulsoup4 matplotlib qtpy PyQt5
```

## Quick start

```bash
# Launch the desktop UI
python memory_as_dynamics.py

# Or verify everything works offline first, with no network and no UI:
python memory_as_dynamics.py --selftest
```

In the UI: paste a URL, choose a memory size (number of neurons `N`), click
**"Scrape & Build Memory"**, then explore the *Attractor Recall*, *Sequence
Replay*, and *Network Diagnostics* tabs.

See `USAGE.md` for a full walkthrough and `DOCUMENTATION.md` for how the
implementation maps onto the paper.

## Honest caveats

- This is a **from-scratch, dependency-light reference implementation** of
  the paper's model, not the paper's own (as-yet-unrun) simulation code. The
  text-to-pattern encoding (random word vectors, summed and thresholded) is a
  stand-in for the paper's abstract ±1 patterns, chosen so related chunks
  land near each other in state space.
- Because real text chunks are **correlated** (unlike the random patterns the
  paper's capacity formulas assume) and updates run **synchronously** rather
  than asynchronously, recall sometimes settles on a neighboring chunk or
  oscillates instead of cleanly converging. This mirrors the crosstalk and
  convergence caveats the paper itself discusses (Sections 4.7 and 8) — it is
  expected behavior, not a bug.
