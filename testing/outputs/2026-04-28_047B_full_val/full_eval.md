# PPM full-val per-byte evaluation

**N bytes scored**: 139,063,300
**Aggregate**: NN=0.98840 bpb, PPM-only=2.06617 bpb, mix=0.93128 bpb. PPM gain = **-0.05711 bpb**
**Gate-fire rate**: 0.1846 (18.5%)

## Quartile breakdown of NN difficulty

Sort all 139,063,300 bytes by NN bits descending. Split into quartiles. For each quartile, see what fraction of NN's loss PPM rescues.

| Quartile | bytes | NN total bits | Mix total bits | PPM saved | rescue rate | gate-fire % | mean NN bits/byte | mean mix bits/byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 (worst NN — top 25%) | 34,765,825 | 86,583,280 | 77,879,776 | +8,703,504 | **+10.1%** | 11.0% | 2.49 | 2.24 |
| Q2 (25-50%) | 34,765,825 | 35,203,276 | 32,020,776 | +3,182,500 | **+9.0%** | 16.1% | 1.01 | 0.92 |
| Q3 (50-75%) | 34,765,825 | 14,388,835 | 15,095,881 | -707,046 | **-4.9%** | 17.6% | 0.41 | 0.43 |
| Q4 (best NN — bottom 25%) | 34,765,825 | 1,274,454 | 4,511,081 | -3,236,627 | **-254.0%** | 29.0% | 0.04 | 0.13 |

## Decile breakdown

| Decile | bytes | NN bits | Mix bits | PPM saved | rescue rate | gate-fire % |
|---|---:|---:|---:|---:|---:|---:|
| D1 (0-10%) | 13,906,330 | 49,781,392 | 45,006,000 | +4,775,392 | **+9.6%** | 7.7% |
| D2 (10-20%) | 13,906,330 | 26,453,632 | 23,653,776 | +2,799,856 | **+10.6%** | 12.6% |
| D3 (20-30%) | 13,906,330 | 19,378,510 | 17,291,724 | +2,086,786 | **+10.8%** | 15.0% |
| D4 (30-40%) | 13,906,330 | 14,879,914 | 13,452,695 | +1,427,219 | **+9.6%** | 16.1% |
| D5 (40-50%) | 13,906,330 | 11,293,120 | 10,496,356 | +796,764 | **+7.1%** | 16.6% |
| D6 (50-60%) | 13,906,330 | 8,059,721 | 7,880,160 | +179,561 | **+2.2%** | 17.1% |
| D7 (60-70%) | 13,906,330 | 4,956,735 | 5,399,964 | -443,229 | **-8.9%** | 17.7% |
| D8 (70-80%) | 13,906,330 | 2,131,024 | 3,137,714 | -1,006,690 | **-47.2%** | 19.7% |
| D9 (80-90%) | 13,906,330 | 473,089 | 1,777,499 | -1,304,410 | **-275.7%** | 27.4% |
| D10 (90-100%) | 13,906,330 | 42,719 | 1,411,624 | -1,368,905 | **-3204.4%** | 34.8% |

## Big wins and losses

- Saved >1 bit/byte: **6,164,604** positions (4.43%)
- Saved >5 bits/byte: **62,942** positions (0.045%)
- Lost >1 bit/byte: **770,829** positions (0.55%)
- Lost >3 bits/byte: **531,907** positions (0.382%)
- Total bits saved by helping positions: 18,890,640
- Total bits lost by hurting positions: 10,948,302
- **Net savings**: +7,942,338 bits = +5.78% of NN val_bpb

## Where PPM hijacks (HIGH regime, NN was right)

Gate fired HIGH AND NN bits < 1.0 (NN was confident-right) AND mix > NN by >1 bit:
- **474,232** hijack positions (0.34% of bytes)
- Total cost from hijacks: **1,668,083** bits (8.8% of total wins)

## Where post-PPM loss concentrates (sorted by mix bits desc)

| Worst N% of positions (post-PPM) | bits | % of total mix |
|---|---:|---:|
| top 0.1% | 1,443,196 | 1.1% |
| top 1.0% | 9,136,464 | 7.1% |
| top 5.0% | 30,240,182 | 23.4% |
| top 10.0% | 48,380,920 | 37.4% |
| top 20.0% | 74,048,768 | 57.2% |
| top 30.0% | 92,565,856 | 71.5% |
| top 50.0% | 116,483,680 | 89.9% |

## By byte category

| category | bytes | % | NN bits/byte | mix bits/byte | PPM saved bits/byte | gate-fire % |
|---|---:|---:|---:|---:|---:|---:|
| alpha | 100,058,956 | 72.0% | 1.007 | 0.986 | +0.021 | 13.1% |
| digit | 1,338,235 | 1.0% | 2.343 | 2.424 | -0.081 | 3.5% |
| space | 21,324,956 | 15.3% | 1.113 | 0.790 | +0.323 | 24.9% |
| punct | 2,629,703 | 1.9% | 1.970 | 2.110 | -0.140 | 4.1% |
| quote | 257,590 | 0.2% | 2.116 | 2.518 | -0.402 | 9.1% |
| other | 13,453,860 | 9.7% | 0.305 | 0.341 | -0.035 | 52.3% |
