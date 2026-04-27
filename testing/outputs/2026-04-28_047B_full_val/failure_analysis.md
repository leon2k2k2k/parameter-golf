# Post-hoc analyses on 047B val NLLs

Source: `/tmp/047B_full_val.npz`. 40540159 scored positions.

## (A) Top-1 confidence calibration

Does the model's softmax probability match its accuracy? Group positions by top-1 confidence, see how often top-1 is correct.

| Top-1 confidence bucket | count | % positions | top-1 accuracy | mean NLL (bits) |
|---|---:|---:|---:|---:|
| 0.00–0.10 | 2,963,580 | 7.3% | 6.8% | 7.80 |
| 0.10–0.30 | 11,167,252 | 27.5% | 19.2% | 5.70 |
| 0.30–0.50 | 7,203,006 | 17.8% | 38.2% | 3.98 |
| 0.50–0.70 | 4,758,522 | 11.7% | 58.2% | 2.68 |
| 0.70–0.90 | 4,590,898 | 11.3% | 79.7% | 1.48 |
| 0.90–0.99 | 5,720,472 | 14.1% | 95.4% | 0.42 |
| 0.99–1.00 | 4,136,429 | 10.2% | 99.6% | 0.05 |

## (B) Overconfident-wrong: model said `p_top1 > 0.9` but was wrong

- **279,328 positions** (0.69% of total)
- Mean NLL on these: **8.10 bits/token**
- Total bits in these positions: 2,263,104
- **1.6% of total NN val_bpb comes from overconfident-wrong positions** ← surprising failure mode

### Top-15 overconfident-wrong (NLL = bits 'wasted' on a wrong sure thing)

| pos | NLL | top-1 prediction (p) | actual | left context (last 50 chars) |
|---|---:|---|---|---|
| 17394259 | 41.31 | `'ipp'` (0.990) | `'stage'` | `e daily inquirer. ang was a juror for the phil` |
| 15566987 | 41.04 | `'y'` (0.994) | `'night'` | ` paris, paris, paris, paris. rendez-bount` |
| 20780721 | 37.92 | `'og'` (0.999) | `'gers'` | ` one of the uk's most popular psychics and astrol` |
| 26575488 | 36.61 | `'hes'` (0.996) | `'hensive'` | `ent: 2 sheets/per set material: paper, pvc, ad` |
| 18037155 | 36.03 | `'▁'` (0.902) | `'fix'` | `-3 days. - product information eylure superfix` |
| 14350 | 36.03 | `'x'` (0.998) | `'stery'` | `and the planning of rooms and passages in a phalan` |
| 38061069 | 35.50 | `'ues'` (0.992) | `'uge'` | ` in the jewish quarter are the spanish synagog` |
| 37332139 | 35.47 | `'▁egg'` (0.992) | `'k'` | ` cook until brown. dig out a little hole for each` |
| 29447765 | 35.22 | `'mon'` (0.997) | `'om'` | `er example of john plugging his book during a ser` |
| 1068650 | 35.21 | `'ver'` (1.000) | `'er'` | `st canadian job request. applying to the vancou` |
| 40239301 | 35.04 | `'neys'` (0.921) | `'▁attention'` | ` box so that we can ignore it and thus return jour` |
| 25281581 | 34.83 | `'at'` (0.999) | `'ache'` | `ly a handful of buildings in the province of sask` |
| 31515484 | 34.60 | `'as'` (1.000) | `'ase'` | `n to advance specific legislative and policy agend` |
| 5566118 | 34.58 | `'ien'` (0.924) | `'▁times'` | `mnation was a certain pharisaism to be found in al` |
| 18478958 | 34.51 | `'bl'` (0.953) | `'▁etc'` | `hopping in his studio, the tour continued to glass` |

## (C) Confident-and-right (the 'free' bytes)

- **9,577,573 positions** (23.62% of total) — model knew exactly what was coming
- Mean NLL on these: **0.0372 bits/token** (effectively free)
- Total bits: 356,425 (0.3% of total)

## (D) Confidence × correctness joint distribution

Cross-tabulating where the bits/positions actually live.

| | confident wrong | confident right | unsure wrong | unsure right |
|---|---:|---:|---:|---:|
| count | 279,328 | 9,577,573 | 19,155,139 | 11,528,119 |
| % positions | 0.7% | 23.6% | 47.2% | 28.4% |
| mean NLL (bits) | 8.10 | 0.0372 | 6.39 | 1.09 |
| total bits | 2,263,104 | 356,425 | 122,464,402 | 12,541,238 |
| % of val_bpb | 1.6% | 0.3% | 89.0% | 9.1% |

## (E) Document boundaries (first byte after BOS)

- **42,170 positions** are the first byte of a new document
- Mean NLL: **1.35 bits** vs overall mean 3.39 bits
- 0.10% of positions, contribute 0.04% of val_bpb

## (F) Cumulative loss contribution (sorted positions)

What fraction of total bits comes from the worst N% of positions?

| Worst N% of positions | bits accounted for |
|---|---:|
| top 1% | 5.0% of val_bpb |
| top 5% | 20.0% of val_bpb |
| top 10% | 34.9% of val_bpb |
| top 20% | 57.8% of val_bpb |
| top 30% | 74.1% of val_bpb |
| top 50% | 92.9% of val_bpb |
