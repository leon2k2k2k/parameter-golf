# Frontier dependency map - snapshot 2026-04-25

Record-track PRs on `openai/parameter-golf`, clustered by code lineage rather
than by date. Numbers are claimed 3-seed mean `val_bpb`.

## TL;DR

- **Only #1493 is merged.** Official SOTA remains **1.0810**.
- Best **clean** open PR: **#1756** @ **1.06505**.
- Best **tokenizer-disputed / likely-legal** open PR: **#1797** @ **1.06157**.
- Best **pre-quant-TTT-disputed** PR: **#1758** @ **1.02840**.
- Best **byte-bug-suspect** PR: **#1791** @ **1.0339**.
- Beyond official SOTA (**1.0810**) on the clean track: **#1756**, **#1775**, **#1776**, **#1780**, **#1784**, **#1790**, **#1792**, **#1802**, **#1809**, **#1812**.
- The live public action is overwhelmingly on **Trunk A / CaseOps descendants**.
- `#1795 @ 0.95165` is not part of our frontier. Treat it as a banned eval-cache / online-memory branch, not as a real target.

## Trunks

### Trunk A - classic SP8192 recurrence / TTT lineage

```text
#1493  MERGED 1.0810  bigbag   official SOTA
 │
 ├─ #1530  open 1.0734  samacqua   VarLen attention + fused MLP + doc-indep LoRA TTT
 │   ├─ #1610  open 1.0728  romeerp   +phased global-SGD eval pass
 │   │   └─ #1767  open 1.07209  renqianluo   +LoRA warm-start A + alpha/rank scaling + WD=1.0
 │   │       ├─ #1784  open 1.07081  renqianluo   +GatedAttn composition
 │   │       ├─ #1790  open 1.06991  miaoyuxun   +SmearGate + AttnOutGate(w24) + LoRA-TTT polish
 │   │       └─ #1792  open 1.07006  renqianluo   +PolarNS + MIN_LR on the alpha-LoRA/gate stack
 │   └─ #1586  open 1.0749  dexhunter   +per-layer GPTQ clip / quant-side tuning
 │       └─ #1626  open 1.0719  dexhunter   +multi-phase global-SGD TTT
 │           └─ #1729  open 1.0678  romeerp   +CaseOps tokenizer + tapered WD  [DISPUTED: tokenizer]
 │               └─ #1736  open 1.06549  dexhunter   +GatedAttn + QuantGate + Loop45 + PhasedTTT  [DISPUTED: tokenizer]
 │                   ├─ #1756  open 1.06505  romeerp   +recurrence depth curriculum
 │                   ├─ #1766  open pending  tashapais   +Recur-Alpha buffer  [DISPUTED: tokenizer]
 │                   ├─ #1769  open 1.06453  dexhunter   +MLPClip12  [DISPUTED: tokenizer]
 │                   ├─ #1771  open 1.06513  bigbag   +depth curriculum + warm-start-A synthesis  [DISPUTED: tokenizer]
 │                   ├─ #1779  open 1.06421  leon2k2k2k   +frozen recurrent carry  [DISPUTED: tokenizer]
 │                   │   └─ #1801  open 1.06287  leon2k2k2k   +sparse gate + updated frozen carry  [DISPUTED: tokenizer]
 │                   └─ #1787  open 1.06335  nprime06   +PolarNS + MIN_LR + sparse gate + fused CE  [DISPUTED: tokenizer]
 │                       └─ #1797  open 1.06157  dexhunter   +SmearGate + LQER asym  [DISPUTED: tokenizer]
 │
 ├─ #1799  open 1.2073  jamesEmerson112   headwise gated attention on older SP8192 legal-TTT stack  (lineage unclear)
 ├─ #1802  open 1.0771  aamodbhatt   +Polar Express NS + MIN_LR warmdown on global MP-TTT
 ├─ #1809  open 1.0800  PranavViswanath   Gram-NS + Polar Express + 3L recurrence + parallel residuals + QK5.25 + legal TTT  (lineage unclear)
 └─ #1812  open 1.0729  EthanNing   SP8192 + 4-epoch score-first eval-time TTT  (lineage unclear)
```

### Trunk C - pre-quant-TTT family and descendants

```text
#1735  open 1.0429  AjAnubolu   parallel pre-quant TTT  [DISPUTED: pre-quant TTT]
 ├─ #1738  open 1.03540  alertcat   +CaseOps Tokenizer V15  [DISPUTED: pre-quant TTT]
 │   ├─ #1758  open 1.02840  kilojoules   LR retune + unfreeze-all  [DISPUTED: pre-quant TTT]
 │   └─ #1807  open 1.07037  davie2009kh   +Huber WD Muon (L1/L2 hinge)  [DISPUTED: pre-quant TTT]
 └─ #1794  open 1.08488  Programmerryoki   per-layer clip + unfrozen score-first TTT + eval guard
```

### Trunk D - GatedDeltaNet / FLA

```text
#1687  CLOSED 1.04090  resouer   K_KVShare_Wider FLA base
 ├─ #1698  open 1.00995  arsenis-cmd   GatedDeltaNet + score-first TTT  [byte-bug]
 └─ #1791  open 1.0339  genji0306   Opensens reproduction of K_KVShare_Wider  [byte-bug]
```

### Unclassified lineage

```text
#1796  open 1.08057  simon-marcus   Scylla tokenizer + legal score-first TTT  [DISPUTED: tokenizer]
 └─ #1813  open 0.94166*  djeidy   +depth recurrence L3-5 + QK-gain 5.25  [DISPUTED: tokenizer]
```

*#1813 inherits buggy byte-accounting from PR #1184 (base_bytes=3 for 27 fallback tokens; should be 1). Corrected estimate ~1.120 bpb. Depth recurrence mechanism is architecturally real; BPB claim is not credible.

The `#1796` / `#1813` cluster uses the Scylla tokenizer and is not clearly attached to the CaseOps or pre-quant trunks.

## Caveats

- Verdict labels reflect our local scan heuristics, not organizer rulings.
- `tokenizer-disputed` here means "non-standard tokenizer, likely legal if the
  lossless-roundtrip and exact-byte-accounting claims hold."
- `byte-bug` means the diff lands in the `from fla.` family and needs an
  independent denominator audit before we should trust the headline `bpb`.
