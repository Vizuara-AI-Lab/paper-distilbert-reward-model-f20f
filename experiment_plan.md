# Experiment Plan — v1

## Research directive
We investigate whether a lightweight distilled encoder can learn human helpfulness/harmlessness preferences. Hypothesis: a 66M-parameter DistilBERT fine-tuned with a pairwise reward-modeling objective on a 2 000-example subset of Anthropic/hh-rlhf will exceed 60% pair-preference accuracy on a held-out 500-pair evaluation split. Metric: pair-preference accuracy (fraction of pairs where `reward(chosen) > reward(rejected)`). Success: accuracy > 0.60 with std < 0.05 across 3 seeds.

## Question
Can a small encoder model learn RLHF preference labels well enough to serve as a practical reward model at the low end of the compute spectrum?

## Dataset + preprocessing
- **Source**: `Anthropic/hh-rlhf` (helpful-base split).
- **Subset**: first 2 000 train pairs, first 500 test pairs — keeps the smoke test inside a T4 budget.
- **Tokenization**: DistilBERT tokenizer, max_length=256, truncate from the right on over-long responses.

## Baseline model
- `distilbert-base-uncased` with a scalar regression head (`nn.Linear(hidden, 1)`).
- Forward: concatenate prompt + response, take `[CLS]` hidden state, project to scalar.
- Pair-preference loss: `-log σ(reward(chosen) − reward(rejected))`.

## Training protocol
- Optimizer: AdamW, lr=2e-5, weight_decay=0.01.
- Batch size: 16 pairs (= 32 forward passes per step).
- Epochs: 3.
- Seeds: `[0, 1, 2]`.
- Mixed precision: fp16.

## Evaluation
- Pair-preference accuracy on the 500-pair test split.
- Log per-epoch train loss + test accuracy to `/outputs/metrics.jsonl`.
- Final: mean + std across 3 seeds → `/outputs/summary.json`.

## Compute budget
- T4 GPU, ~6–10 minutes of wall-clock per seed, ~20–30 min total for 3 seeds.
- Expected Modal cost: ≤ $0.30 (well inside the $1.00 cap).
