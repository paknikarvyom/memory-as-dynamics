#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory as Dynamics — a web-page memory encoder
================================================

Implements the recurrent attractor + sequence memory model described in
"Memory as Dynamics: A Unified Model of Distributed Sequence Encoding in
Recurrent Neural Systems" (Paknikar, 2026), and wraps it in a small tool that:

  1. scrapes a URL for its readable text,
  2. turns that text into a JSON record of ordered "chunks",
  3. encodes each chunk as a distributed bipolar pattern and stores it in a
     recurrent network as an attractor memory (Hebbian rule) and as a step in
     a sequence memory (temporal / successor-predecessor rule),
  4. visualises recall, sequence replay and basic capacity diagnostics in a
     QtPy desktop UI.

Paper -> code map
------------------
  Eq. (1), (11)   DynamicMemoryNetwork.step         x(t+1)=tanh(beta*(W_A x + g(t) W_T x + b))
  Eq. (3), (4)    DynamicMemoryNetwork.spectral_radius_at   stability check, rho(J) < 1
  Eq. (5), (8)    DynamicMemoryNetwork.store_patterns       Hebbian attractor matrix W_A
  Eq. (6), (9)    DynamicMemoryNetwork.store_sequence       temporal / successor-predecessor matrix W_T
  Eq. (7)         gamma decay applied to W_A / W_T (homeostatic regularization)
  Eq. (12)        DynamicMemoryNetwork.gate                theta-like periodic gate g(t)
  Eq. (14), (15)  DynamicMemoryNetwork.capacity_estimates   Pmax ~ 0.14N, Smax ~ N/(c ln N)
  Eq. (16)        DynamicMemoryNetwork.energy_proxy         event-based energy proxy
  Table 2 (m_mu)  DynamicMemoryNetwork.overlap              pattern/state overlap

Text -> pattern encoding is a distributed random-projection scheme (a stand-in
for the paper's abstract +-1 patterns xi): every word gets a fixed pseudo-random
+-1 vector, and a chunk's pattern is the (thresholded) sum of its word vectors.
This keeps the whole pipeline dependency-light and fully offline once a page
has been scraped, while preserving the property the paper's patterns need:
similar text lands near similar patterns.

Dependencies
------------
  pip install torch numpy requests beautifulsoup4 matplotlib qtpy PyQt5

Usage
-----
  python memory_as_dynamics.py                 # launch the desktop UI
  python memory_as_dynamics.py --headless URL  # console demo, no UI, needs network
  python memory_as_dynamics.py --selftest      # offline pipeline check, no network/UI
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import requests
from bs4 import BeautifulSoup


# ============================================================================
# 1. Scraping -> JSON
# ============================================================================

SAMPLE_HTML_FOR_SELFTEST = """
<html><head><title>Sample Page: Neural Memory</title></head>
<body>
  <h1>Attractor Networks Store Memories as Stable States</h1>
  <p>A Hopfield network is a recurrent neural network that stores patterns
  as fixed points of its dynamics, allowing noisy or partial cues to be
  completed back to the original memory.</p>
  <h2>Sequences Are Trajectories</h2>
  <p>Ordered experiences can be encoded as trajectories that visit a series
  of attractor states one after another, driven by asymmetric connections
  learned through temporally asymmetric plasticity.</p>
  <p>A gating signal can switch the network between holding a single memory
  steady and advancing along a learned sequence of memories.</p>
  <h2>Why It Matters For Hardware</h2>
  <p>Because memory and computation share the same weight matrix, this kind
  of model maps naturally onto in-memory and neuromorphic hardware, where
  moving data between separate memory and compute units is costly.</p>
  <li>Stable attractors give content-addressable recall from partial cues.</li>
  <li>Temporal association rules give ordered, replayable sequences.</li>
</body></html>
"""


def scrape_url(
    url: str,
    max_chunks: int = 48,
    min_chars: int = 20,
    max_chars: int = 400,
    timeout: int = 15,
    html_override: Optional[str] = None,
) -> Dict:
    """Fetch a URL (or use html_override for offline testing) and pull out an
    ordered list of readable text chunks (headings, paragraphs, list items).
    """
    if html_override is not None:
        html = html_override
    else:
        headers = {"User-Agent": "Mozilla/5.0 (MemoryAsDynamics/1.0; research tool)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.encoding or "utf-8"
        html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer", "header"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    chunks: List[Dict] = []
    order = 0
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = " ".join(el.get_text(" ", strip=True).split())
        if len(text) < min_chars:
            continue
        text = text[:max_chars]
        chunks.append({"id": order, "order": order, "tag": el.name, "text": text})
        order += 1
        if len(chunks) >= max_chunks:
            break

    if not chunks:
        raise ValueError("No readable text chunks (headings/paragraphs/list items) were found on this page.")

    return {"title": title, "chunks": chunks}


def build_json_record(url: str, scraped: Dict) -> Dict:
    """Package scraped content into the JSON record the memory is built from."""
    return {
        "url": url,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "title": scraped["title"],
        "num_chunks": len(scraped["chunks"]),
        "chunks": scraped["chunks"],
    }


# ============================================================================
# 2. Text -> distributed bipolar pattern encoding
# ============================================================================

class BipolarEncoder:
    """Maps text to distributed patterns in {-1, +1}^dim.

    Each token is assigned a fixed pseudo-random +-1 vector (seeded from a
    hash of the token, so the mapping is stable across runs). A chunk's
    pattern is the element-wise sign of the sum of its token vectors -- a
    simple, dependency-free stand-in for the paper's abstract patterns xi,
    chosen so that chunks sharing vocabulary land close together in state
    space (nonzero overlap), while unrelated chunks are close to orthogonal.
    """

    def __init__(self, dim: int = 256, seed: int = 1234):
        self.dim = dim
        self.seed = seed
        self._cache: Dict[str, np.ndarray] = {}

    def _token_vector(self, token: str) -> np.ndarray:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        digest = hashlib.md5(f"{self.seed}:{token}".encode("utf-8")).hexdigest()
        rng = np.random.RandomState(int(digest, 16) % (2**32 - 1))
        vec = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=self.dim)
        self._cache[token] = vec
        return vec

    def encode(self, text: str) -> np.ndarray:
        tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
        if not tokens:
            digest = hashlib.md5(f"{self.seed}:{text}".encode("utf-8")).hexdigest()
            rng = np.random.RandomState(int(digest, 16) % (2**32 - 1))
            return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=self.dim)
        acc = np.zeros(self.dim, dtype=np.float32)
        for tok in tokens:
            acc += self._token_vector(tok)
        pattern = np.sign(acc)
        pattern[pattern == 0.0] = 1.0
        return pattern.astype(np.float32)

    @staticmethod
    def corrupt(pattern: np.ndarray, noise_frac: float, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Flip a fraction of bits -- used to build the noisy cues of Task A."""
        rng = rng if rng is not None else np.random.RandomState()
        out = pattern.copy()
        n_flip = int(round(max(0.0, min(1.0, noise_frac)) * len(pattern)))
        if n_flip > 0:
            idx = rng.choice(len(pattern), size=n_flip, replace=False)
            out[idx] = out[idx] * -1.0
        return out


# ============================================================================
# 3. The recurrent memory network (paper's core model)
# ============================================================================

class DynamicMemoryNetwork:
    """Recurrent attractor + sequence memory.

        W = W_A + g(t) * W_T                                   (Eq. 11)

    W_A : symmetric Hebbian matrix, holds items as stable attractors (Eq. 5/8, Sec 4.5)
    W_T : asymmetric successor-predecessor matrix, advances sequences (Eq. 6/9)
    g(t): periodic theta-like gate scaling W_T (Eq. 12)
    """

    def __init__(
        self,
        n_neurons: int,
        beta: float = 2.0,
        gamma: float = 1e-4,
        gmax: float = 1.6,
        period: int = 12,
        theta_g: float = 0.0,
        kappa: float = 8.0,
        device: str = "cpu",
    ):
        self.N = n_neurons
        self.beta = beta
        self.gamma = gamma
        self.gmax = gmax
        self.period = period
        self.theta_g = theta_g
        self.kappa = kappa
        self.device = torch.device(device)

        self.W_A = torch.zeros(self.N, self.N, device=self.device)
        self.W_T = torch.zeros(self.N, self.N, device=self.device)
        self.b = torch.zeros(self.N, device=self.device)

        self.patterns: List[torch.Tensor] = []
        self.labels: List[str] = []
        self._seq_order: List[int] = []

    # ---- storage / learning -------------------------------------------------
    def store_patterns(self, patterns: List[np.ndarray], labels: Optional[List[str]] = None) -> None:
        """Auto-associative storage: closed-form batch Hebbian rule
        W_A = (1/N) * sum_mu xi^mu (xi^mu)^T, with homeostatic decay (Eq. 5, 7, 8; Sec 4.5)."""
        if not patterns:
            raise ValueError("No patterns to store.")
        self.patterns = [torch.tensor(p, dtype=torch.float32, device=self.device) for p in patterns]
        self.labels = labels if labels is not None else [f"pattern_{i}" for i in range(len(patterns))]
        Xi = torch.stack(self.patterns)                 # (P, N)
        W_A = (Xi.t() @ Xi) / float(self.N)              # Eq. (5)/(8) closed form
        W_A.fill_diagonal_(0.0)                          # standard no-self-connection variant
        self.W_A = W_A * (1.0 - self.gamma)              # Eq. (7)

    def store_sequence(self, order: Optional[List[int]] = None, cyclic: bool = False) -> None:
        """Hetero-associative sequence storage, Eq. (9):
        W_T = (1/N) * sum_{i=1}^{k-1} x_{i+1} x_i^T (successor-predecessor)."""
        if len(self.patterns) < 2:
            self.W_T = torch.zeros(self.N, self.N, device=self.device)
            self._seq_order = list(range(len(self.patterns)))
            return
        idx = order if order is not None else list(range(len(self.patterns)))
        seq = [self.patterns[i] for i in idx]
        W_T = torch.zeros(self.N, self.N, device=self.device)
        for i in range(len(seq) - 1):
            W_T += torch.outer(seq[i + 1], seq[i])
        if cyclic:
            W_T += torch.outer(seq[0], seq[-1])
        self.W_T = (W_T / float(self.N)) * (1.0 - self.gamma)
        self._seq_order = idx

    # ---- dynamics -------------------------------------------------------------
    def gate(self, t: float) -> float:
        """Periodic theta-like gate, Eq. (12): g(t) = gmax * sigmoid(kappa*(cos(2*pi*t/T) - theta_g))."""
        z = self.kappa * (math.cos(2.0 * math.pi * t / self.period) - self.theta_g)
        return float(self.gmax / (1.0 + math.exp(-z)))

    def step(self, x: torch.Tensor, gate_val: float = 0.0, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One update of Eq. (11): x(t+1) = tanh(beta * (W_A x + g(t) W_T x + U u + b))."""
        a = self.W_A @ x + gate_val * (self.W_T @ x) + self.b
        if u is not None:
            a = a + u
        return torch.tanh(self.beta * a)

    def energy(self, x: torch.Tensor) -> float:
        """Hopfield-style energy of the symmetric (attractor-only) part, Sec 3.1.
        Not defined once the asymmetric W_T term is active (no Lyapunov function then, Sec 3.3/8)."""
        return float(-0.5 * (x @ (self.W_A @ x)) - (x @ self.b))

    def overlap(self, x: torch.Tensor, pattern: torch.Tensor) -> float:
        """m = xi . x / (||xi|| ||x||), Table 2."""
        denom = (x.norm() * pattern.norm()).clamp_min(1e-8)
        return float((pattern @ x) / denom)

    def best_match(self, x: torch.Tensor) -> Tuple[int, float]:
        if not self.patterns:
            return -1, 0.0
        overlaps = [self.overlap(x, p) for p in self.patterns]
        i = int(np.argmax(overlaps))
        return i, overlaps[i]

    # ---- recall (Task A) --------------------------------------------------
    def recall(self, cue: np.ndarray, steps: int = 60, gate_val: float = 0.0, tol: float = 1e-4) -> Dict:
        """Free-run retrieval from a (possibly noisy) cue, Eq. (1)/(11), Task A."""
        x = torch.tensor(cue, dtype=torch.float32, device=self.device)
        overlap_trace: List[float] = []
        energy_trace: List[float] = []
        converged_at: Optional[int] = None
        for t in range(steps):
            _, best_ov = self.best_match(x)
            overlap_trace.append(best_ov)
            energy_trace.append(self.energy(x))
            x_next = self.step(x, gate_val=gate_val)
            if converged_at is None and torch.norm(x_next - x) < tol:
                converged_at = t
            x = x_next
        final_idx, final_ov = self.best_match(x)
        return {
            "final_state": x.cpu().numpy(),
            "final_best_idx": final_idx,
            "final_overlap": final_ov,
            "overlap_trace": overlap_trace,
            "energy_trace": energy_trace,
            "converged_at": converged_at,
        }

    # ---- sequence replay (Task B) ------------------------------------------
    def replay_sequence(self, steps: int = 60, x0: Optional[torch.Tensor] = None) -> Dict:
        """Free-run the gated dynamics starting from (by default) the first stored
        item, Eq. (11)-(12), Task B."""
        if x0 is None:
            if not self.patterns:
                raise ValueError("No patterns stored.")
            start_idx = self._seq_order[0] if self._seq_order else 0
            x0 = self.patterns[start_idx].clone()
        x = x0.clone()
        gate_trace: List[float] = []
        index_trace: List[int] = []
        overlap_trace: List[float] = []
        for t in range(steps):
            g = self.gate(t)
            gate_trace.append(g)
            idx, ov = self.best_match(x)
            index_trace.append(idx)
            overlap_trace.append(ov)
            x = self.step(x, gate_val=g)
        return {"gate_trace": gate_trace, "index_trace": index_trace, "overlap_trace": overlap_trace}

    # ---- diagnostics (Eqs. 3-4, 14-16) -------------------------------------
    def spectral_radius_at(self, x: torch.Tensor) -> float:
        """rho(J(x*)) for the symmetric part, Eq. (3)-(4). Should be < 1 at a stable memory."""
        a = self.W_A @ x + self.b
        phi_prime = self.beta * (1.0 - torch.tanh(self.beta * a) ** 2)
        J = torch.diag(phi_prime) @ self.W_A
        eigvals = torch.linalg.eigvals(J)
        return float(torch.max(torch.abs(eigvals)).real)

    def capacity_estimates(self) -> Dict[str, float]:
        """Pmax ~ 0.14N (Eq. 14) and Smax ~ N/(c ln N), c in [2,4] (Eq. 15)."""
        n = max(self.N, 3)
        return {
            "Pmax_hebbian": 0.14 * n,
            "Smax_low_c4": n / (4.0 * math.log(n)),
            "Smax_high_c2": n / (2.0 * math.log(n)),
        }

    def energy_proxy(self, traj: List[torch.Tensor], e_spike: float = 1.0) -> float:
        """Event-based energy proxy, Eq. (16), with firing probability s_i = (1+x_i)/2."""
        total = 0.0
        for x in traj:
            s = (1.0 + x) / 2.0
            total += float(s.sum()) * e_spike
        return total

    # ---- persistence --------------------------------------------------------
    def to_state(self) -> Dict:
        return {
            "N": self.N, "beta": self.beta, "gamma": self.gamma, "gmax": self.gmax,
            "period": self.period, "theta_g": self.theta_g, "kappa": self.kappa,
            "W_A": self.W_A.cpu(), "W_T": self.W_T.cpu(), "b": self.b.cpu(),
            "patterns": [p.cpu() for p in self.patterns], "labels": self.labels,
            "seq_order": self._seq_order,
        }

    @classmethod
    def from_state(cls, d: Dict) -> "DynamicMemoryNetwork":
        net = cls(d["N"], beta=d["beta"], gamma=d["gamma"], gmax=d["gmax"],
                   period=d["period"], theta_g=d["theta_g"], kappa=d["kappa"])
        net.W_A = d["W_A"]
        net.W_T = d["W_T"]
        net.b = d["b"]
        net.patterns = d["patterns"]
        net.labels = d["labels"]
        net._seq_order = d.get("seq_order", list(range(len(d["labels"]))))
        return net


# ============================================================================
# 4. Offline self-test (no network, no UI) -- also doubles as a usage example
# ============================================================================

def run_pipeline(url: str, n_neurons: int = 256, html_override: Optional[str] = None) -> Dict:
    """Run scrape -> JSON -> encode -> store -> recall -> replay end to end."""
    scraped = scrape_url(url, html_override=html_override)
    record = build_json_record(url, scraped)

    encoder = BipolarEncoder(dim=n_neurons)
    patterns = [encoder.encode(c["text"]) for c in record["chunks"]]
    labels = [f"[{c['tag']}] {c['text'][:60]}" for c in record["chunks"]]

    net = DynamicMemoryNetwork(n_neurons=n_neurons)
    net.store_patterns(patterns, labels)
    net.store_sequence()

    rng = np.random.RandomState(0)
    cue = BipolarEncoder.corrupt(patterns[0], 0.25, rng=rng)
    recall_result = net.recall(cue, steps=40)
    replay_result = net.replay_sequence(steps=min(30, 3 * len(patterns)))

    return {
        "record": record,
        "network": net,
        "encoder": encoder,
        "recall_result": recall_result,
        "replay_result": replay_result,
    }


def selftest() -> None:
    print("Running offline self-test (embedded sample HTML, no network) ...")
    out = run_pipeline("about:sample", n_neurons=192, html_override=SAMPLE_HTML_FOR_SELFTEST)
    record, net = out["record"], out["network"]
    print(f"Title: {record['title']}")
    print(f"Chunks stored: {record['num_chunks']}")
    print(json.dumps(record["chunks"][:2], indent=2))
    rr = out["recall_result"]
    print(f"\nRecall from a 25%-noise cue of chunk 0:")
    print(f"  converged at step {rr['converged_at']}, final best match = "
          f"chunk {rr['final_best_idx']} (overlap {rr['final_overlap']:.3f})")
    rep = out["replay_result"]
    print(f"\nSequence replay index trace (first 10 steps): {rep['index_trace'][:10]}")
    caps = net.capacity_estimates()
    print(f"\nCapacity estimates for N={net.N}: {caps}")
    print("\nSelf-test OK.")


def run_headless(url: str, n_neurons: int = 256) -> None:
    out = run_pipeline(url, n_neurons=n_neurons)
    record, net = out["record"], out["network"]
    print(f"Title: {record['title']}")
    print(f"Chunks stored: {record['num_chunks']}")
    print(json.dumps(record, indent=2)[:2000])
    rr = out["recall_result"]
    print(f"\nRecall demo: converged_at={rr['converged_at']}, "
          f"final_best_idx={rr['final_best_idx']}, overlap={rr['final_overlap']:.3f}")
    print(f"Capacity estimates: {net.capacity_estimates()}")


# ============================================================================
# 5. QtPy desktop UI
# ============================================================================

def _launch_gui() -> int:
    from qtpy.QtCore import Qt, QThread, Signal
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
        QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QSlider,
        QTabWidget, QListWidget, QPlainTextEdit, QFileDialog, QMessageBox,
        QGroupBox, QFormLayout, QCheckBox, QStatusBar,
    )

    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    # ---- background worker for scraping + building (keeps UI responsive) --
    class BuildWorker(QThread):
        progress = Signal(str)
        finished_ok = Signal(dict)
        failed = Signal(str)

        def __init__(self, url: str, n_neurons: int, use_sequence: bool):
            super().__init__()
            self.url = url
            self.n_neurons = n_neurons
            self.use_sequence = use_sequence

        def run(self):
            try:
                self.progress.emit(f"Fetching {self.url} ...")
                scraped = scrape_url(self.url)
                self.progress.emit(f"Found {len(scraped['chunks'])} text chunks. Building JSON record ...")
                record = build_json_record(self.url, scraped)

                self.progress.emit(f"Encoding chunks into {self.n_neurons}-dim bipolar patterns ...")
                encoder = BipolarEncoder(dim=self.n_neurons)
                patterns = [encoder.encode(c["text"]) for c in record["chunks"]]
                labels = [f"[{c['tag']}] {c['text'][:60]}" for c in record["chunks"]]

                self.progress.emit("Storing attractor memories (Hebbian rule, Eq. 5/8) ...")
                net = DynamicMemoryNetwork(n_neurons=self.n_neurons)
                net.store_patterns(patterns, labels)

                if self.use_sequence:
                    self.progress.emit("Storing reading-order sequence (temporal rule, Eq. 9) ...")
                    net.store_sequence()

                self.progress.emit("Memory built successfully.")
                self.finished_ok.emit({"record": record, "network": net, "encoder": encoder})
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(f"{type(exc).__name__}: {exc}")

    class MplCanvas(FigureCanvas):
        def __init__(self, nrows=1, ncols=1, width=5.5, height=3.6, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
            self.axes = self.fig.subplots(nrows, ncols)
            super().__init__(self.fig)

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Memory as Dynamics — URL Memory Encoder")
            self.resize(1180, 780)

            self.record: Optional[Dict] = None
            self.network: Optional[DynamicMemoryNetwork] = None
            self.encoder: Optional[BipolarEncoder] = None
            self.worker: Optional[BuildWorker] = None

            self._build_ui()

        # -------------------------------------------------------------- UI
        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            root = QVBoxLayout(central)

            # -- top control bar --------------------------------------------------
            top_box = QGroupBox("Build memory from a URL")
            top_layout = QHBoxLayout(top_box)

            top_layout.addWidget(QLabel("URL:"))
            self.url_edit = QLineEdit("https://en.wikipedia.org/wiki/Hopfield_network")
            self.url_edit.setMinimumWidth(360)
            top_layout.addWidget(self.url_edit, 3)

            top_layout.addWidget(QLabel("Neurons N:"))
            self.n_spin = QSpinBox()
            self.n_spin.setRange(32, 2048)
            self.n_spin.setSingleStep(32)
            self.n_spin.setValue(256)
            top_layout.addWidget(self.n_spin)

            self.seq_check = QCheckBox("Store reading-order sequence")
            self.seq_check.setChecked(True)
            top_layout.addWidget(self.seq_check)

            self.build_btn = QPushButton("Scrape && Build Memory")
            self.build_btn.clicked.connect(self.on_build_clicked)
            top_layout.addWidget(self.build_btn)

            root.addWidget(top_box)

            # -- log console -------------------------------------------------------
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(500)
            self.log.setFixedHeight(90)
            mono = QFont("Monospace")
            mono.setStyleHint(QFont.TypeWriter)
            self.log.setFont(mono)
            root.addWidget(self.log)

            # -- tabs ----------------------------------------------------------------
            self.tabs = QTabWidget()
            root.addWidget(self.tabs, 1)

            self.tabs.addTab(self._build_chunks_tab(), "Chunks / JSON")
            self.tabs.addTab(self._build_recall_tab(), "Attractor Recall")
            self.tabs.addTab(self._build_sequence_tab(), "Sequence Replay")
            self.tabs.addTab(self._build_diagnostics_tab(), "Network Diagnostics")

            self.status = QStatusBar()
            self.setStatusBar(self.status)
            self.status.showMessage("Enter a URL and click 'Scrape & Build Memory' to begin.")

        def _build_chunks_tab(self) -> QWidget:
            w = QWidget()
            layout = QHBoxLayout(w)
            splitter = QSplitter(Qt.Horizontal)

            left = QWidget()
            left_layout = QVBoxLayout(left)
            left_layout.addWidget(QLabel("Stored chunks (order = reading order):"))
            self.chunk_list = QListWidget()
            left_layout.addWidget(self.chunk_list)
            splitter.addWidget(left)

            right = QWidget()
            right_layout = QVBoxLayout(right)
            right_layout.addWidget(QLabel("JSON record built from the page:"))
            self.json_view = QPlainTextEdit()
            self.json_view.setReadOnly(True)
            right_layout.addWidget(self.json_view)
            btn_row = QHBoxLayout()
            self.save_json_btn = QPushButton("Save JSON...")
            self.save_json_btn.clicked.connect(self.on_save_json)
            self.save_mem_btn = QPushButton("Save Memory (.ptmem)...")
            self.save_mem_btn.clicked.connect(self.on_save_memory)
            self.load_mem_btn = QPushButton("Load Memory (.ptmem)...")
            self.load_mem_btn.clicked.connect(self.on_load_memory)
            btn_row.addWidget(self.save_json_btn)
            btn_row.addWidget(self.save_mem_btn)
            btn_row.addWidget(self.load_mem_btn)
            right_layout.addLayout(btn_row)
            splitter.addWidget(right)

            splitter.setSizes([380, 700])
            layout.addWidget(splitter)
            return w

        def _build_recall_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            controls = QGroupBox("Content-addressable recall (Task A, Eq. 1 / 11)")
            form = QFormLayout(controls)

            self.recall_combo_list = QListWidget()
            self.recall_combo_list.setMaximumHeight(110)
            form.addRow(QLabel("Pick a stored chunk to recall:"), self.recall_combo_list)

            noise_row = QHBoxLayout()
            self.noise_slider = QSlider(Qt.Horizontal)
            self.noise_slider.setRange(0, 90)
            self.noise_slider.setValue(25)
            self.noise_label = QLabel("25% bits flipped")
            self.noise_slider.valueChanged.connect(
                lambda v: self.noise_label.setText(f"{v}% bits flipped")
            )
            noise_row.addWidget(self.noise_slider)
            noise_row.addWidget(self.noise_label)
            form.addRow(QLabel("Cue noise:"), noise_row)

            self.recall_steps_spin = QSpinBox()
            self.recall_steps_spin.setRange(5, 300)
            self.recall_steps_spin.setValue(60)
            form.addRow(QLabel("Max steps:"), self.recall_steps_spin)

            self.recall_btn = QPushButton("Run Recall")
            self.recall_btn.clicked.connect(self.on_run_recall)
            form.addRow(self.recall_btn)

            layout.addWidget(controls)

            self.recall_result_label = QLabel("Build a memory, then run a recall.")
            self.recall_result_label.setWordWrap(True)
            layout.addWidget(self.recall_result_label)

            self.recall_canvas = MplCanvas(nrows=1, ncols=2, width=9, height=3.4)
            layout.addWidget(self.recall_canvas)
            return w

        def _build_sequence_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            controls = QGroupBox("Gated sequence replay (Task B, Eq. 11 / 12)")
            form = QFormLayout(controls)
            self.replay_steps_spin = QSpinBox()
            self.replay_steps_spin.setRange(5, 400)
            self.replay_steps_spin.setValue(60)
            form.addRow(QLabel("Steps:"), self.replay_steps_spin)
            self.replay_btn = QPushButton("Replay Sequence")
            self.replay_btn.clicked.connect(self.on_run_replay)
            form.addRow(self.replay_btn)
            layout.addWidget(controls)

            self.replay_canvas = MplCanvas(nrows=2, ncols=1, width=9, height=5.2)
            layout.addWidget(self.replay_canvas)

            layout.addWidget(QLabel("Retrieved order (best-matching chunk at each step):"))
            self.replay_text = QPlainTextEdit()
            self.replay_text.setReadOnly(True)
            self.replay_text.setFixedHeight(110)
            layout.addWidget(self.replay_text)
            return w

        def _build_diagnostics_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)
            self.diag_label = QLabel("Build a memory to see capacity and stability diagnostics.")
            self.diag_label.setWordWrap(True)
            layout.addWidget(self.diag_label)
            self.diag_canvas = MplCanvas(nrows=1, ncols=1, width=6, height=5.5)
            layout.addWidget(self.diag_canvas)
            return w

        # ---------------------------------------------------------- logging
        def _log(self, msg: str):
            self.log.appendPlainText(msg)
            self.status.showMessage(msg, 5000)

        # ------------------------------------------------------- build flow
        def on_build_clicked(self):
            url = self.url_edit.text().strip()
            if not url:
                QMessageBox.warning(self, "Missing URL", "Please enter a URL first.")
                return
            self.build_btn.setEnabled(False)
            self.log.clear()
            self._log(f"Starting build for {url} ...")

            self.worker = BuildWorker(url, self.n_spin.value(), self.seq_check.isChecked())
            self.worker.progress.connect(self._log)
            self.worker.finished_ok.connect(self.on_build_finished)
            self.worker.failed.connect(self.on_build_failed)
            self.worker.start()

        def on_build_failed(self, message: str):
            self.build_btn.setEnabled(True)
            self._log(f"ERROR: {message}")
            QMessageBox.critical(self, "Build failed", message)

        def on_build_finished(self, payload: Dict):
            self.build_btn.setEnabled(True)
            self.record = payload["record"]
            self.network = payload["network"]
            self.encoder = payload["encoder"]

            self.chunk_list.clear()
            self.recall_combo_list.clear()
            for c in self.record["chunks"]:
                item_text = f"#{c['id']:02d} [{c['tag']}] {c['text'][:90]}"
                self.chunk_list.addItem(item_text)
                self.recall_combo_list.addItem(item_text)
            self.recall_combo_list.setCurrentRow(0)

            self.json_view.setPlainText(json.dumps(self.record, indent=2))
            self._log(f"Built memory with N={self.network.N} neurons and "
                      f"{len(self.network.patterns)} stored patterns.")
            self._refresh_diagnostics()

        # ------------------------------------------------------------ recall
        def on_run_recall(self):
            if self.network is None or self.encoder is None:
                QMessageBox.information(self, "No memory yet", "Build a memory first.")
                return
            row = self.recall_combo_list.currentRow()
            if row < 0:
                row = 0
            pattern = self.network.patterns[row].cpu().numpy()
            noise_frac = self.noise_slider.value() / 100.0
            rng = np.random.RandomState(row * 97 + 1)
            cue = BipolarEncoder.corrupt(pattern, noise_frac, rng=rng)

            steps = self.recall_steps_spin.value()
            result = self.network.recall(cue, steps=steps, gate_val=0.0)

            target_label = self.network.labels[row]
            got_idx = result["final_best_idx"]
            got_label = self.network.labels[got_idx] if got_idx >= 0 else "?"
            correct = (got_idx == row)
            conv = result["converged_at"]
            conv_str = f"step {conv}" if conv is not None else "did not fully converge in the given steps"

            self.recall_result_label.setText(
                f"Cue = chunk #{row} with {int(noise_frac*100)}% of bits flipped.\n"
                f"Recalled: chunk #{got_idx} (overlap {result['final_overlap']:.3f}), "
                f"{'MATCH' if correct else 'MISMATCH'} vs. target chunk #{row}. Converged at {conv_str}.\n"
                f"Target: {target_label}\nRecalled: {got_label}"
            )

            ax_overlap, ax_energy = self.recall_canvas.axes
            ax_overlap.clear()
            ax_overlap.plot(result["overlap_trace"], color="#2f6fed")
            ax_overlap.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
            ax_overlap.set_title("Overlap with best-matching pattern")
            ax_overlap.set_xlabel("step t")
            ax_overlap.set_ylabel("overlap m(t)")
            ax_overlap.set_ylim(-1.05, 1.05)

            ax_energy.clear()
            ax_energy.plot(result["energy_trace"], color="#e0622b")
            ax_energy.set_title("Attractor energy H(x(t))  (Eq. context Sec 3.1)")
            ax_energy.set_xlabel("step t")
            ax_energy.set_ylabel("H(x)")

            self.recall_canvas.draw()

        # -------------------------------------------------------------- replay
        def on_run_replay(self):
            if self.network is None:
                QMessageBox.information(self, "No memory yet", "Build a memory first.")
                return
            steps = self.replay_steps_spin.value()
            result = self.network.replay_sequence(steps=steps)

            ax_gate, ax_idx = self.replay_canvas.axes
            ax_gate.clear()
            ax_gate.plot(result["gate_trace"], color="#8a3ffc")
            ax_gate.set_title("Gate g(t)  (Eq. 12)")
            ax_gate.set_ylabel("g(t)")

            ax_idx.clear()
            ax_idx.step(range(len(result["index_trace"])), result["index_trace"], where="mid", color="#0f9d58")
            ax_idx.set_title("Best-matching stored chunk over time")
            ax_idx.set_xlabel("step t")
            ax_idx.set_ylabel("chunk index")

            self.replay_canvas.fig.tight_layout()
            self.replay_canvas.draw()

            lines = []
            prev = None
            for t, idx in enumerate(result["index_trace"]):
                if idx != prev:
                    label = self.network.labels[idx] if 0 <= idx < len(self.network.labels) else "?"
                    lines.append(f"t={t:3d}  ->  #{idx:02d}  {label}")
                    prev = idx
            self.replay_text.setPlainText("\n".join(lines))

        # ---------------------------------------------------------- diagnostics
        def _refresh_diagnostics(self):
            if self.network is None:
                return
            net = self.network
            caps = net.capacity_estimates()
            try:
                rho = net.spectral_radius_at(net.patterns[0])
                rho_str = f"{rho:.3f}"
            except Exception:
                rho_str = "n/a"

            self.diag_label.setText(
                f"N = {net.N} neurons   |   P = {len(net.patterns)} stored chunks   |   "
                f"beta = {net.beta}   |   gamma = {net.gamma}\n"
                f"Estimated pattern capacity Pmax ~ 0.14N = {caps['Pmax_hebbian']:.1f}   |   "
                f"Estimated sequence capacity Smax ~ N/(c ln N) = "
                f"[{caps['Smax_low_c4']:.1f}, {caps['Smax_high_c2']:.1f}] for c in [4, 2]\n"
                f"Spectral radius rho(J) at stored chunk #0 = {rho_str} "
                f"(should be < 1 for a stable attractor, Eq. 3-4)"
            )
            ax = self.diag_canvas.axes
            ax.clear()
            im = ax.imshow(net.W_A.cpu().numpy(), cmap="coolwarm", vmin=-0.05, vmax=0.05)
            ax.set_title("Attractor weight matrix W_A")
            ax.set_xlabel("neuron j")
            ax.set_ylabel("neuron i")
            self.diag_canvas.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self.diag_canvas.fig.tight_layout()
            self.diag_canvas.draw()

        # ------------------------------------------------------------- saving
        def on_save_json(self):
            if self.record is None:
                QMessageBox.information(self, "Nothing to save", "Build a memory first.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save JSON record", "memory_record.json", "JSON (*.json)")
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.record, f, indent=2)
            self._log(f"Saved JSON record to {path}")

        def on_save_memory(self):
            if self.network is None:
                QMessageBox.information(self, "Nothing to save", "Build a memory first.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save memory", "memory.ptmem", "PyTorch memory (*.ptmem)")
            if not path:
                return
            torch.save({"state": self.network.to_state(), "record": self.record}, path)
            self._log(f"Saved memory (weights + JSON record) to {path}")

        def on_load_memory(self):
            path, _ = QFileDialog.getOpenFileName(self, "Load memory", "", "PyTorch memory (*.ptmem)")
            if not path:
                return
            try:
                payload = torch.load(path, weights_only=False)
                self.network = DynamicMemoryNetwork.from_state(payload["state"])
                self.record = payload.get("record")
                self.encoder = BipolarEncoder(dim=self.network.N)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Load failed", str(exc))
                return

            self.chunk_list.clear()
            self.recall_combo_list.clear()
            if self.record:
                for c in self.record["chunks"]:
                    item_text = f"#{c['id']:02d} [{c['tag']}] {c['text'][:90]}"
                    self.chunk_list.addItem(item_text)
                    self.recall_combo_list.addItem(item_text)
                self.json_view.setPlainText(json.dumps(self.record, indent=2))
            else:
                for i, lbl in enumerate(self.network.labels):
                    self.chunk_list.addItem(f"#{i:02d} {lbl}")
                    self.recall_combo_list.addItem(f"#{i:02d} {lbl}")
            if self.recall_combo_list.count():
                self.recall_combo_list.setCurrentRow(0)
            self._log(f"Loaded memory from {path}")
            self._refresh_diagnostics()

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_() if hasattr(app, "exec_") else app.exec()


# ============================================================================
# 6. Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Memory as Dynamics -- URL memory encoder")
    parser.add_argument("--headless", metavar="URL", help="Run the pipeline on URL without the GUI (needs network).")
    parser.add_argument("--selftest", action="store_true", help="Run an offline pipeline check (no network, no GUI).")
    parser.add_argument("--neurons", type=int, default=256, help="Number of neurons N (default: 256).")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if args.headless:
        run_headless(args.headless, n_neurons=args.neurons)
        return

    try:
        sys.exit(_launch_gui())
    except ImportError as exc:
        print("Could not start the GUI -- missing a Qt binding for QtPy.")
        print("Install one with e.g.  pip install qtpy PyQt5")
        print(f"Details: {exc}")
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
