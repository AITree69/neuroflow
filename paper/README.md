# NeuroFlow Paper

This directory contains the LaTeX source for the NeuroFlow paper,
intended for submission to arXiv (cs.MS / cs.LG / cs.PL).

## Files

- `paper.tex` — main LaTeX source (single-file, no separate `.bib`)
- `paper.bib` — not used; references are inline via `\begin{thebibliography}`
- `README.md` — this file

## Compile to PDF

### Option A — TeX Live (recommended)

```bash
# Linux (Debian/Ubuntu)
sudo apt install texlive-latex-extra texlive-fonts-recommended

# macOS (Homebrew)
brew install --cask mactex

# Windows
winget install --id MiKTeX.MiKTeX -e
# or
winget install --id TeXLive.TeXLive -e
```

Then:

```bash
cd paper
pdflatex paper.tex
pdflatex paper.tex      # second pass for cross-references
```

The result is `paper.pdf`.  No `bibtex` step is needed.

### Option B — tectonic (single-binary, fast, auto-runs as many passes as needed)

```bash
# Linux / macOS / Windows
winget install --id Tectonic.Tectonic -e
# or
curl -fsSL https://drop-sh.fullyjustified.net | sh

cd paper
tectonic paper.tex
```

`tectonic` will fetch missing packages on the fly and run enough
LaTeX passes automatically.  Recommended for CI.

### Option C — Overleaf (no install)

1. Go to <https://www.overleaf.com>.
2. New project → Upload Project → upload `paper.tex`.
3. Hit "Recompile".

## Submission to arXiv

arXiv accepts `.tex` source + figures.  Steps:

1. Compile to `paper.pdf` (any of the above).
2. Go to <https://arxiv.org/submit>.
3. Category: **cs.MS** (Mathematical Software) — primary.
   Cross-list to **cs.LG** and **cs.PL** if relevant.
4. Upload `paper.tex`.  All packages used (`geometry`, `microtype`,
   `times`, `amsmath`, `tikz`, `algorithm2e`, etc.) are in standard
   TeX Live `texlive-latex-extra`; arXiv will compile automatically.
5. Title in the metadata field: `NeuroFlow: An Industrial-Grade Open
   Framework for Neural Operator-Based PDE Solving`.
6. Abstract: copy from the `\begin{abstract}` block.
7. Authors: add a single anonymous author or your real name and
   affiliation.  For the initial anonymous submission use
   `Anonymous NeuroFlow Contributors`.

## Source-level claims to verify before submission

The numbers in Table 1 (Sec.~\ref{sec:eval}) come from running
`examples/02_export_and_infer.py` after training with
`examples/01_train_burgers1d.py --epochs 5`.  The headline numbers
to expect on the developer's machine:

- C{++} vs \py{} max abs diff: $\sim 5.2 \times 10^{-5}$
- C{++} latency (batch=1, $n=256$, width=64, 4 layers): $\sim 11$\,ms
- \py{} latency: $\sim 6$--$8$\,ms

The `0.6x` C{++}/Python speedup in the headline is honest
disclosure: Stage~1's C{++} is intentionally a line-by-line
translation; the speedup story is Stage~4.

## License

The LaTeX source is released under CC~BY~4.0.
The code described in the paper is Apache-2.0.
