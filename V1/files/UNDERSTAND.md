# Understanding This Tool — In Plain Words

This page explains `memory_as_dynamics.py` without the technical jargon:
what it does, how it works step by step, how you use it, and what it's
actually good for.

---

## What is this, in one sentence?

It's a tool that reads a web page, turns its content into a kind of
"artificial memory" inspired by how brains store memories, and then lets you
watch that memory recall things — even from incomplete or scrambled input,
and in the order the page originally presented them.

---

## The idea behind it

Most computer memory works like a filing cabinet: information is stored in a
specific labeled slot, and you retrieve it by asking for that exact label.
Brains don't seem to work that way. Instead, a memory seems to be a *pattern
of activity* that settles into a stable shape — a bit like a marble rolling
into one of several dips in a landscape and staying there. Give the brain a
partial or noisy version of that pattern (a blurry photo, a half-remembered
name), and it tends to "fall into" the same dip and complete the memory.
Sequences of memories (like a song, or the steps of a task) seem to work
similarly — each memory pulls the next one into place in order.

This tool builds a small, simplified version of that idea and lets you watch
it work on real content pulled from a real web page.

---

## What it does, step by step

1. **Reads a web page you give it.**
   It fetches the page and pulls out the readable text — headings,
   paragraphs, and bullet points — in the order they appear.

2. **Turns that text into a structured record.**
   It organizes the page's title, the source URL, and each piece of text
   into a simple, ordered list — saved as a JSON file, a common, readable
   data format. Think of this as "notes taken from the page."

3. **Converts each piece of text into a "memory pattern."**
   Each chunk of text (a heading, a paragraph, a bullet) is converted into a
   long string of plus/minus values — a bit like a fingerprint for that
   piece of text. Pieces of text that share a lot of the same words end up
   with similar fingerprints; unrelated pieces end up looking very different
   from each other.

4. **Stores those fingerprints in a memory network.**
   The tool builds two things out of these fingerprints:
   - A set of **stable "resting points"** — one for each chunk of text —
     so that a corrupted or partial version of a chunk tends to settle back
     into the original.
   — A **sense of order** — a "what comes next" connection between
     consecutive chunks, so the memory can be played back in the sequence
     the page originally presented them.

5. **Lets you test and watch the memory.**
   Through the on-screen controls, you can:
   - deliberately scramble part of a memory and watch the system recover it
     (or fail to, if you scramble it too much),
   - "play back" the sequence of chunks and watch the memory step through
     them in order,
   - inspect some basic statistics about how much the memory can reliably
     hold.

---

## What you'll actually see on screen

- **A box to type in a URL**, plus a button to build the memory from it.
- **A running log** telling you what's happening (fetching the page,
  building the memory, etc.).
- **Four tabs** to explore what was built:
  1. **Chunks / JSON** — the plain list of text the page was broken into,
     and the underlying JSON notes, with the option to save them to a file.
  2. **Attractor Recall** — pick one piece of text, scramble part of it, and
     watch the memory try to recover the original.
  3. **Sequence Replay** — watch the memory step through the page's content
     in its original order, on its own.
  4. **Network Diagnostics** — a few numbers and a picture describing how
     "full" and how stable the memory currently is.

---

## How to read the data and results

### The JSON record

This is the plainest data in the tool — just notes taken from the page:

| Field | What it means |
|---|---|
| `url` | The page you gave it. |
| `scraped_at` | When it was read. |
| `title` | The page's title. |
| `num_chunks` | How many pieces of text it found and stored. |
| `chunks` | The pieces themselves, in order — each with an `id` (its position), a `tag` (was it a heading, a paragraph, a bullet?), and the `text` itself. |

There's nothing to "interpret" here beyond reading it — it's a structured
version of the page's content, in the order it appeared.

### The Attractor Recall tab

- **Target vs. recalled chunk, and MATCH / MISMATCH** — did the memory
  settle back onto the exact chunk you scrambled, or onto a different one?
  A mismatch usually means either the noise was too high, or that chunk
  looks too similar to another one stored in the same memory.
- **Overlap** — a number from -1 to 1 describing how close the current state
  is to a stored memory. **1 means an exact match, 0 means no resemblance,
  negative means "the opposite" of that memory.** Watching this number rise
  toward 1 across the recall run is watching the memory correct itself.
- **Energy** — think of this as "how settled" the memory currently is, like
  a marble's height on a hilly landscape. A line that trends downward and
  levels off means the memory is settling into a stable resting point. A
  line that stays flat or jumps around means it hasn't settled yet (or is
  stuck between two resting points).
- **"Converged at step X" / "did not fully converge"** — whether the memory
  visibly stopped changing before the run ended. Not converging isn't
  necessarily a failure — it can just mean it needs a few more steps, or that
  it's gently oscillating rather than settling exactly.

### The Sequence Replay tab

- **Gate signal** — a wave that rises and falls on a rhythm. Think of it as
  a metronome: while it's low, the memory holds still on one chunk; while
  it's high, the memory is nudged toward the next chunk in the sequence.
- **Chunk index over time** — this should generally look like a staircase
  going up (0, then 1, then 2, ...) as the gate advances the memory through
  the page's content in order. If it jumps around or repeats a lot instead
  of stepping cleanly, the sequence isn't being followed reliably (often
  because there are too many chunks for the memory's size).
- **The transition log** underneath spells this out in words: which chunk
  the memory was on, and when it moved to the next one.

### The Network Diagnostics tab

- **N** (size of the memory) and **P** (chunks actually stored) — the bigger
  N is relative to P, the more "room" the memory has and the more reliable
  recall tends to be.
- **Capacity estimates** — rough ceilings on how much this memory *should*
  be able to reliably hold, given its size. If the number of chunks you
  stored (P) is already close to or past these estimates, expect more
  mismatches and stuck sequences — the memory is simply full.
- **Spectral radius** — a single stability number. Below 1 is a good sign
  (the stored memories are genuinely stable); at or above 1 suggests the
  memory may not settle reliably.
- **The heatmap** — a picture of the memory's internal connections. Bright
  blocks or visible clusters usually mean groups of chunks that share a lot
  of vocabulary (and are therefore more likely to get confused with each
  other during recall).

---

## How to use it

1. Install the required software once (see `README.md` / `USAGE.md` for the
   exact commands).
2. Run the program — a window opens.
3. Paste in a URL (an article, a blog post, a documentation page — anything
   with real, readable text works well).
4. Click **"Scrape & Build Memory."**
5. Explore the tabs: try scrambling a memory and recalling it, or replaying
   the sequence, or just reading the JSON notes it built.

There's also a way to run it without opening a window at all — either as a
quick offline check that nothing is broken (`--selftest`), or by pointing it
at a real URL from a command line and reading the results as text
(`--headless`). These are covered in `USAGE.md`.

---

## What is this useful for?

- **Learning and demonstration.** It's a hands-on way to see, rather than
  just read about, how "memory as a stable pattern" and "sequence as an
  ordered chain of patterns" behave — useful if you're studying neuroscience-
  inspired computing, associative memory, or recurrent neural networks.
- **Exploring real content through this lens.** Instead of only testing the
  idea on random made-up data, you can feed it a real article and see how it
  handles real, messy, correlated language — including where it starts to
  struggle, which is itself informative.
- **A starting point for experimentation.** Because it's a single,
  readable file, it's easy to change: swap in a different way of turning
  text into patterns, try different page sizes, tune how "sticky" the memory
  or its sequence-following behavior is, and immediately see the effect in
  the visualizations.
- **Stress-testing a page's content for repetitiveness.** If two sections of
  a page confuse the memory (recall keeps landing on the wrong one, or the
  heatmap shows a bright cluster), that's a signal those sections are
  worded very similarly to each other — potentially useful feedback for
  someone editing the page, independent of the memory-science angle.
- **Prototyping "content-addressable" and "order-based" recall ideas** before
  committing to a heavier framework — this is a small, inspectable sandbox
  for that, not a production system.
- **Teaching or presenting** biologically-inspired / neuromorphic computing
  concepts, since it turns an otherwise abstract paper (attractors, energy
  landscapes, gated sequences) into something you can point a real web page
  at and watch happen on screen.
- **Getting a structured summary of a page's headings and paragraphs** as a
  side effect — the JSON record alone can be useful even before you look at
  the memory behavior at all.

---

## How this is different from other kinds of "memory"

- **A filing cabinet or database** stores each item at an exact address and
  retrieves it by that exact address (a filename, a key, a row ID). This
  tool's memory has no addresses at all — you give it a *rough* or *damaged*
  version of a memory, and it iteratively nudges itself toward the full
  version, the way a marble rolls downhill into a dip rather than being
  placed there directly.

- **The "memory" in most AI chat tools** (often called a vector database or
  embedding search) stores a fixed list of items and, when asked, does a
  single one-shot comparison to find the closest match — no back-and-forth,
  no settling process. This tool instead runs an internal loop, updating its
  state step by step until it stabilizes (or doesn't) — closer to "thinking
  it over" than "looking it up."

- **An AI chat model's context window** isn't really persistent memory at
  all — it's just the recent conversation text being re-read each time, and
  it's gone once the conversation ends or the window fills up. This tool's
  memory is a fixed set of connections you build once and can save to a file,
  independent of any conversation.

- **A classic Hopfield network** (the closest relative to this tool) only
  does the "settle into a stable pattern" part. This tool adds a second
  layer on top: a rhythmic "gate" that can deliberately push the memory from
  one settled pattern to the next, so it can also replay an *ordered
  sequence* of memories — not just recall isolated ones.

- **Human memory**, which inspired all of this, is vastly more complex: it
  consolidates over time, forgets selectively, and retrieves differently
  depending on context and emotion. This tool borrows only the specific
  ideas of "memory as a stable settling pattern" and "sequence as a
  gate-driven chain of those patterns" — a deliberately small slice of the
  real thing, not a simulation of it.

The common thread across all of these comparisons: most familiar systems
separate *where things are stored* from *how they're looked up*. In this
tool, storage and retrieval are the same thing — the connections that hold
the memories are also the mechanism that recalls them, which is the core
idea the underlying paper is built around.

## What it is *not*

- It is not a production-grade knowledge base, search engine, or chatbot
  memory system — it doesn't do meaning-based search or question answering.
- It doesn't guarantee perfect recall. Real text chunks often share
  vocabulary, which can cause the memory to confuse similar pieces of
  text — a limitation of this style of memory in general, not just a flaw
  in this tool, and part of what makes it interesting to explore.
- It doesn't store anything permanently on its own — memories only persist
  between runs if you explicitly save them to a file.
