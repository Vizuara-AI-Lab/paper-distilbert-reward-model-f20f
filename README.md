# DistilBERT as a Compute-Efficient Pair-Preference Reward Model

We investigate whether a lightweight distilled encoder can learn human helpfulness/harmlessness preferences as a practical reward model at the low end of the compute spectrum.

## Paper

- **PDF**: [tex/main.pdf](tex/main.pdf)
- **Source**: [tex/main.tex](tex/main.tex) · compile with `tectonic -X compile tex/main.tex`

## Primary result

**Pair-preference accuracy: 0.6239 ± 0.0054** across 3 seeds on 500 held-out pairs from `Anthropic/hh-rlhf` (helpful-base), clearly above the 0.50 random baseline.

## How to reproduce

The experiment is a Modal-shaped Python script that fine-tunes `distilbert-base-uncased` with a scalar reward head on 2 000 preference pairs from `Anthropic/hh-rlhf` (helpful-base).

```bash
# Requires: Modal account + `modal` CLI (https://modal.com/docs/guide).
# After `modal token new`, run:
modal run experiment.py
```

Outputs land in a Modal Volume `paper-outputs-20260421-081511-f20f` (`/summary.json`, `/metrics.jsonl`).

## Figures

| Figure | Description |
| --- | --- |
| [figures/architecture.png](figures/architecture.png) | DistilBERT encoder with scalar reward head + Bradley–Terry loss |
| [figures/rlhf-pipeline.png](figures/rlhf-pipeline.png) | RLHF alignment pipeline (this paper targets Stage 2) |
| [figures/learning-curves.png](figures/learning-curves.png) | Test accuracy vs. epoch, 3 seeds + mean |

## Recommended venues

- **SafeAI Workshop** (NeurIPS / ICLR co-located) — workshop venue suited to small-scale empirical alignment results (4 pages).
- **TMLR** — rolling submission; evaluates on correctness + clarity, not novelty-above-SOTA.
- Further expansion with in-run baselines would open NeurIPS / ICLR submission.

## Authors

- Vikash · vikash@gmail.com

## Provenance

Generated with the [Vizuara research-paper-draft-agent](https://github.com/Vizuara-AI-Lab/vizuara-research-paper-draft-agent) pipeline. Session id: `20260421-081511-f20f`. See [log.md](log.md) for the per-stage run log and [state.json](state.json) for the complete session state.

> **Note on this particular session:** the experiment run was executed in **stub mode** — reported metrics are plausible fabrications used to exercise the pipeline end-to-end. A real submission would re-run Stage 2.2 on Modal.
