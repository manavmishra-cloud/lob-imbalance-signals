# Paper — lob-imbalance-signals

LaTeX source for the paper accompanying this repository.

**Working title:** *Directional Accuracy in Limit Order Book Prediction:
Sequence Models vs Tabular ML on BTC/USDT*

**Target venue:** arXiv `q-fin.ST` (Statistical Finance) or `cs.LG`
(Machine Learning). Workshop-paper length (~8-10 pages).

**Status:** Draft v1 — complete abstract, intro, related work, data,
methodology, results (with real numbers), discussion, conclusion.
Limitations and future-work sections present.

## Files

```
paper/
├── README.md          # this file
├── main.tex           # paper source
└── references.bib     # bibliography
```

## Sections

| Section | Status | Word count (approx) |
|---|---|---|
| Abstract | ✓ Complete | 250 |
| 1. Introduction | ✓ Complete | 500 |
| 2. Related Work | ✓ Complete | 350 |
| 3. Data | ✓ Complete | 300 |
| 4. Methodology | ✓ Complete | 550 |
| 5. Results | ✓ Complete with real numbers | 500 |
| 6. Discussion | ✓ Complete with limitations | 450 |
| 7. Conclusion | ✓ Complete | 250 |
| References | ✓ 7 entries (all verified) | -- |

Total: ~3,200 words, expected PDF length ~8-10 pages.

## Headline finding

Across 72{,}001 BTC/USDT snapshots at 100~ms cadence, the five-model
comparison shows that overall classification accuracy and trading-
relevant directional accuracy rank models nearly inversely under
realistic LOB class imbalance (73.5% no-move).

| Model | Overall acc | Directional acc |
|---|---|---|
| Linear OFI | 78.0% | 10.4% |
| XGBoost | 79.0% | 23.9% |
| LSTM | 59.2% | 58.5% |
| Transformer | 53.0% | **63.3%** |

The takeaway: practitioners and researchers should report both
metrics; headline accuracy alone systematically prefers degenerate
dominant-class predictors over models with genuine trading-relevant
signal.

## How to build the PDF

### Overleaf (easiest)
1. Create a free account at https://overleaf.com
2. New Project → Upload Project → upload the ZIP (build via the
   helper command below)
3. Click Recompile

### Local LaTeX (Linux)
```bash
sudo apt install texlive-latex-recommended texlive-fonts-extra texlive-bibtex-extra biber
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Next iteration

Recommended additions before submission:

1. **Multi-asset replication** (BTC, ETH, SOL) for cross-asset
   robustness of the result
2. **Statistical significance test** on pairwise directional-
   accuracy differences (requires longer capture or panel)
3. **Trade-stream capture** for genuine trade-flow features
4. **Sensitivity analysis** on sequence length and hidden size
5. **Attention-pattern interpretability** for the Transformer

## arXiv submission

Endorsement chain: same flow as `nifty-vol-forecast`. The first paper's
endorsement is per-archive (econ.EM for nifty, likely q-fin.ST or
cs.LG for this one), so a separate endorsement may be required unless
we pick an archive where the user is already endorsed at submission time.
