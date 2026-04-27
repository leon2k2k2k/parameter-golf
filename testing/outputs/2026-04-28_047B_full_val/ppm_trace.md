# PPM byte-trace on first ~104827 bytes of val

## Setup
- Order=4, λ_hi=0.9 (NN dominant when PPM unsure), λ_lo=0.05 (PPM dominant when confident)
- Gate threshold: PPM max-prob ≥ 0.9 → HIGH regime

## Big PPM wins — PPM correct, NN spread or wrong

| pos | context (last 40 chars) | NN top-1 token (prob) | PPM said (prob) | actual byte | NN bits | PPM bits | mix bits | **saved** |
|---|---|---|---|---|---:|---:|---:|---:|
| 64133 | `��why, are you both joking?” razumi` | `_asked` (0.28) | `h` (0.97) | `h` | 9.75 | 0.05 | 0.12 | **+9.63** |
| 57340 | `r very nature again, and to my mind they` | `_are` (0.19) | ` ` (0.91) | ` ` | 8.70 | 0.13 | 0.21 | **+8.50** |
| 72729 | `�foo! i have muddled it!” por` | `ters` (0.41) | `f` (0.98) | `f` | 8.23 | 0.03 | 0.10 | **+8.13** |
| 42053 | `s telling me to-day,” put in porfir` | `io` (0.35) | `y` (0.96) | `y` | 7.87 | 0.05 | 0.13 | **+7.74** |
| 72491 | `” “what do you mean?” razumi` | `_asked` (0.40) | `h` (0.97) | `h` | 7.74 | 0.04 | 0.11 | **+7.62** |
| 64390 | `s, and discourteous sarcasm of porfir` | `i` (0.07) | `y` (0.97) | `y` | 7.65 | 0.04 | 0.11 | **+7.54** |
| 50231 | `e influence of environment.” razumi` | `’` (0.15) | `h` (0.96) | `h` | 7.53 | 0.06 | 0.13 | **+7.40** |
| 42194 | `r.” “and there,” said razumi` | `,` (0.71) | `h` (0.94) | `h` | 7.53 | 0.08 | 0.16 | **+7.37** |
| 72732 | `�foo! i have muddled it!” porfir` | `in` (0.09) | `y` (0.98) | `y` | 7.38 | 0.03 | 0.10 | **+7.28** |
| 95226 | `ard towards men, and valiant against his` | `_own` (0.58) | ` ` (0.95) | ` ` | 7.36 | 0.07 | 0.15 | **+7.22** |
| 35163 | `eeded in taking a good look at porfir` | `io` (0.21) | `y` (0.91) | `y` | 7.29 | 0.14 | 0.21 | **+7.08** |
| 46289 | `n in quite a different tone, laughing to` | `_himself` (0.92) | ` ` (0.95) | ` ` | 7.16 | 0.07 | 0.14 | **+7.01** |
| 60043 | `skolnikov, raising his eyes to porfir` | `’` (0.09) | `y` (0.97) | `y` | 7.09 | 0.04 | 0.11 | **+6.98** |
| 92368 | ` a slap on his snout, a slap on sagoi` | `’` (0.78) | `n` (0.92) | `n` | 7.08 | 0.13 | 0.20 | **+6.88** |
| 64387 | `vous, and discourteous sarcasm of por` | `os` (0.19) | `f` (0.97) | `f` | 6.92 | 0.04 | 0.11 | **+6.81** |
| 48406 | `ively dislike history, ânothing but` | `_the` (0.18) | ` ` (0.93) | ` ` | 6.76 | 0.10 | 0.18 | **+6.59** |
| 72490 | `.” “what do you mean?” razum` | `ov` (0.16) | `i` (0.97) | `i` | 6.62 | 0.04 | 0.11 | **+6.51** |
| 49660 | `ch hold of him, do!” laughed porfir` | `.` (0.13) | `y` (0.97) | `y` | 6.59 | 0.04 | 0.11 | **+6.47** |
| 49657 | `catch hold of him, do!” laughed por` | `k` (0.09) | `f` (0.97) | `f` | 6.47 | 0.04 | 0.11 | **+6.35** |
| 71010 | ` know very well,” he turned to rask` | `ed` (0.74) | `o` (0.98) | `o` | 6.44 | 0.03 | 0.10 | **+6.34** |
| 71011 | `know very well,” he turned to rasko` | `ed` (0.74) | `l` (0.98) | `l` | 6.44 | 0.03 | 0.10 | **+6.34** |
| 50230 | `he influence of environment.” razum` | `’` (0.07) | `i` (0.96) | `i` | 6.44 | 0.06 | 0.13 | **+6.31** |
| 60040 | `�raskolnikov, raising his eyes to por` | `os` (0.13) | `f` (0.97) | `f` | 6.21 | 0.04 | 0.11 | **+6.09** |
| 42050 | ` was telling me to-day,” put in por` | `k` (0.24) | `f` (0.96) | `f` | 6.13 | 0.05 | 0.13 | **+6.01** |
| 9738 | ` be untipped, the existential threats to` | `_the` (0.17) | ` ` (0.92) | ` ` | 6.00 | 0.12 | 0.19 | **+5.82** |

## PPM losses (HIGH regime, hurt by >2 bits)

| pos | NN bits | PPM bits | mix bits | lost | PPM thought (p) | actual | ← context |
|---|---:|---:|---:|---:|---|---|---|
| 2401 | 0.07 | 30.11 | 4.40 | **-4.32** | `\x81` (0.98) | `\x82` | ` i loved everything about it (3:00 �` |
| 78851 | 0.46 | 28.37 | 4.78 | **-4.32** | ` ` (0.99) | `\xee` | ` find installations of icewarp:”` |
| 74334 | 0.54 | 25.92 | 4.86 | **-4.32** | `h` (0.97) | `n` | `dditional time in order to interview wit` |
| 19064 | 0.01 | 24.85 | 4.33 | **-4.32** | ` ` (0.94) | `?` | `��where is this article plagiarized from` |
| 72450 | 0.90 | 24.75 | 5.22 | **-4.32** | ` ` (0.99) | `.` | `a flat open anywhere, no, there wasn’t` |
| 81269 | 1.29 | 24.80 | 5.61 | **-4.32** | ` ` (0.93) | `'` | `e and late late nights. they` |
| 78485 | 0.46 | 20.93 | 4.78 | **-4.32** | ` ` (0.93) | `:` | `bmitted: 2004-09-23 00:00:00 added by` |
| 38212 | 2.19 | 22.54 | 6.51 | **-4.32** | ` ` (0.94) | `!` | `e in despair! you know what women are` |
| 27238 | 0.95 | 21.16 | 5.27 | **-4.32** | ` ` (0.93) | `n` | ` your 'press release'? we know it was` |
| 78971 | 3.39 | 23.08 | 7.72 | **-4.32** | ` ` (0.97) | `h` | `�web mail” inurl:”:32000/mail/”` |
| 88535 | 0.03 | 19.56 | 4.35 | **-4.32** | ` ` (0.96) | `i` | `and i are talking in private:” ’t` |
| 61603 | 0.33 | 19.47 | 4.65 | **-4.32** | `f` (0.92) | `v` | `even to the cow, like to imagine themsel` |
| 29270 | 0.52 | 19.47 | 4.84 | **-4.32** | `\xee` (0.91) | `-` | `russia: the russian messenger. ` |
| 51854 | 5.23 | 23.49 | 9.55 | **-4.32** | `\xee` (0.90) | `\xc3` | ` yours which interested me at the time. ` |
| 20142 | 0.41 | 17.96 | 4.73 | **-4.32** | `a` (0.93) | `p` | `e no recent comments.monday, se` |

## Aggregate per regime

| Regime | count | mean NN bits | mean mix bits | mean Δ |
|---|---:|---:|---:|---:|
| HIGH wins | 1924 | 1.97 | 0.14 | +1.83 |
| HIGH neutral | 6910 | 0.26 | 0.09 | +0.17 |
| HIGH losses | 319 | 1.48 | 5.37 | -3.89 |