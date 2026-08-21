# NeurIPS 2026 workshop papers

This directory contains two double-blind variants of the same study, built from one shared evidence base:

- `taieval/`: measurement validity, auditability, parser/missingness checks, and evaluation protocol.
- `lp4fm/`: multilinguality, language-conditioned reporting, and transfer across foundation models.

Both use the official `neurips_2026.sty` workshop format and keep the main text to six pages. References, the technical appendix, and the official checklist follow the main text.

Compile from the repository root with:

```bash
mkdir -p papers/neurips2026/build/taieval
cd papers/neurips2026/taieval
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=../build/taieval main.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=../build/taieval main.tex
```

Replace `taieval` with `lp4fm` for the second paper. The checked PDFs are in `build/` locally; the source of truth is the two `main.tex` files plus `shared/`.

Before submission, replace the anonymous author block only after the workshop's review policy permits it, and upload an anonymous artifact rather than the public repository link during double-blind review.
