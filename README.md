# Memory as Dynamics: A Unified Model of Distributed Sequence Encoding in Recurrent Neural Systems

A single-file Python application that turns a web page into a **recurrent
attractor + sequence memory**, implementing the model from the paper _"Memory
as Dynamics: A Unified Model of Distributed Sequence Encoding in Recurrent
Neural Systems"_ (Paknikar, 2026), and lets you explore it through a desktop
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

| File                          | Purpose                                                                                            |
| ----------------------------- | -------------------------------------------------------------------------------------------------- |
| `V1/memory_as_dynamics_v2.py` | **v2 implementation**: Enhanced scraper, projection rule, decorrelation, damping, and benchmarks. |
| `V1/memory_as_dynamics.py`    | **v1 baseline**: Direct paper-faithful implementation of attractor + sequence memory.              |
| `README.md`                   | Project overview, setup, and changelog.                                                            |
| `V1/files/DOCUMENTATION.md`   | Technical reference: architecture, classes, and equation mappings.                                 |
| `V1/files/USAGE.md`           | Step-by-step instructions for running the app in GUI, headless, and self-test modes.               |

## What's New in v2 (Updates over v1)

v2 introduces practical extensions to handle real-world correlated text and enhance numerical stability, while preserving paper-faithful defaults in the library API:

- **Corpus Decorrelation & IDF Weighting**: `BipolarEncoder` adds IDF weighting and corpus-mean centering (`fit` / `encode_corpus`), significantly reducing cross-talk between chunks that share common vocabulary.
- **Projection (Pseudo-Inverse) Learning**: Implements `DynamicMemoryNetwork(rule="projection")` for $W_A$ and $W_T$ as an alternative to Hebbian learning, drastically improving capacity and attractor separation on correlated inputs.
- **Damped Synchronous Dynamics**: Added leaky damping `alpha < 1` to smooth synchronous update oscillations and stabilize limit cycles.
- **Cleaner Web Scraping**: Improved boilerplate/duplicate removal and heading-to-paragraph merging for fewer, more distinct chunks.
- **Diagnostic Metrics & Benchmarking**:
  - Analytical capacity warnings, `stability_report()`, `recommend_n()`, `recall_accuracy()`, and `replay_score()`.
  - Built-in benchmark suite accessible via `--benchmark`.
- **UI Enhancements**:
  - Learning rule selector (Hebbian vs. Projection) and Auto-$N$ sizing.
  - Controls for damping ($\alpha$), gate threshold ($\theta_g$), decorrelation, and text cleaning.
  - Recall verdict that distinguishes between clean convergence, wrong attractor, and blended states.
  - Fixed Matplotlib diagnostics colorbar stacking.
- **Safe Persistence & Unicode**:
  - Saved `.ptmem` checkpoints now store encoder state and learning rule, loading securely with `torch.load(weights_only=True)`.
  - Scraped JSON displays and saves genuine Unicode characters instead of escaped glyphs.

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
# Launch the v2 desktop UI
python V1/memory_as_dynamics_v2.py

# Run benchmark on synthetic correlated corpus
python V1/memory_as_dynamics_v2.py --benchmark

# Run offline selftest pipeline (no UI, no network)
python V1/memory_as_dynamics_v2.py --selftest

# Headless run against a URL
python V1/memory_as_dynamics_v2.py --headless "https://en.wikipedia.org/wiki/Attractor_network"
```

In the UI: paste a URL, choose a memory size (number of neurons `N`), click
**"Scrape & Build Memory"**, then explore the _Attractor Recall_, _Sequence
Replay_, and _Network Diagnostics_ tabs.

See `V1/files/USAGE.md` for a full walkthrough and `V1/files/DOCUMENTATION.md` for how the
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

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or see the LICENSE file in this repository.
