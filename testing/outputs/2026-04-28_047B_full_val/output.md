# PPM-D byte-mixture inspection

**Checkpoint:** `/workspace/runs/047B-loop-kv-shrink-screen/final_model.pt`
**Tokens scored:** 40540159 (bytes: 139063300)
**PPM config:** order=4 λ_hi=0.9 λ_lo=0.05 threshold=0.9

## Headline

| Metric | bits/byte |
|---|---:|
| **NN only** (`nn_byte_bpb`) | **0.98840** |
| PPM only (`ppm_only`) | 2.16005 |
| **NN + PPM mix** (`mix_bpb`) | **0.93975** |
| **Δ from PPM** | **-0.04864** |
| Gate high-confidence fraction | 0.1724 (17.2%) |
| Token-level reference (`token_bpb`) | 0.98840 |

## Comparison to dexhunter's #1857

| | #1857 | this run |
|---|---:|---:|
| nn_byte_bpb | 1.10020 | 0.98840 |
| ppm_only | 2.34028 | 2.16005 |
| mix_bpb | 1.03176 | 0.93975 |
| gate_high_frac | 0.14241 | 0.17237 |
| Δ from PPM | -0.06844 | -0.04864 |

## Per-category contribution to NN-only val_bpb

| category | count | % positions | mean bits/byte | bits | % NN val_bpb |
|---|---:|---:|---:|---:|---:|
| PROSE | 35160215 | 86.7% | 1.12 | 130090332.3 | 94.5% |
| empty | 3929062 | 9.7% | 0.00 | 3896392.6 | 2.8% |
| NUMERIC | 1338235 | 3.3% | 2.34 | 3135542.2 | 2.3% |
| PATH | 48124 | 0.1% | 3.81 | 183917.1 | 0.1% |
| CODE | 33508 | 0.1% | 4.71 | 164671.6 | 0.1% |
| HEX | 23391 | 0.1% | 1.37 | 136414.4 | 0.1% |
| URL | 7624 | 0.0% | 0.66 | 17897.9 | 0.0% |

**PPM-addressable categories (URL+NUMERIC+CODE+HEX+PATH): 2.6% of NN val_bpb**

## NLL distribution (NN only, bits/byte)

| bucket | count | % | mean | contribution |
|---|---:|---:|---:|---:|
| 0.0–0.5 | 16934936 | 41.8% | 0.119 | 0.0499 (4.3%) |
| 0.5–1.5 | 10513498 | 25.9% | 0.964 | 0.2501 (21.4%) |
| 1.5–3.0 | 7573352 | 18.7% | 2.126 | 0.3971 (34.0%) |
| 3.0–5.0 | 3598968 | 8.9% | 3.796 | 0.3370 (28.8%) |
| 5.0–∞ | 1919405 | 4.7% | 7.180 | 0.3400 (29.1%) |

## Top-50 most catastrophic NN predictions

These are the bytes where the model was most surprised — high NLL means the model assigned ~0% probability to what actually came next.

| pos | NLL (bits) | actual | top-1 prediction (prob) | category | left context (last 50 chars) |
|---|---:|---|---|---|---|
| 19975736 | 42.96 | `'net'` | `'▁'` (0.617) | PROSE | `aquatic therapy for newborns also known as neonatal hydrotherapy aquatic` |
| 33543940 | 42.48 | `'ale'` | `'▁use'` (0.441) | PROSE | `><0x91><0xA4>i like this. she loves it, i gave it as a gift. it's so easy to` |
| 30144567 | 42.11 | `'oil'` | `'▁site'` (0.554) | PROSE | `e the following about links and posts on this site: any/all of the links on the` |
| 38454641 | 41.72 | `'other'` | `'▁my'` (0.342) | PROSE | `on wednesday, sept. 18, the friends of the art circle library will host a` |
| 17394259 | 41.31 | `'stage'` | `'ipp'` (0.990) | PROSE | `y for the newspaper philippine daily inquirer. ang was a juror for the phil` |
| 15566987 | 41.04 | `'night'` | `'y'` (0.994) | PROSE | `offers in paris, paris, paris, paris, paris, paris, paris. rendez-bount` |
| 30511920 | 40.61 | `'i'` | `'▁appropriate'` (0.218) | PROSE | `e. it is the duty of any person or buyer using this mls listing to perform the` |
| 39246260 | 39.44 | `'▁fly'` | `'re'` (0.435) | PROSE | `he dmz and at the same time saigon was being threatened from the west. so we’` |
| 35366293 | 39.17 | `'els'` | `'ful'` (0.425) | PROSE | ` specific overt act that was a substantial step toward re-entry, constitute harm` |
| 21875795 | 39.14 | `'another'` | `'.'` (0.610) | PROSE | `it accepts most interchangeable pentax 35-mm film camera lenses and accessories` |
| 32246589 | 38.63 | `'▁read'` | `'pon'` (0.540) | PROSE | `of the customers from that source are purchasing product a, all the deals cou` |
| 23369652 | 38.13 | `'atch'` | `'▁is'` (0.453) | PROSE | `ent bands sound so good! what caused all this stir about dirty pictures? below` |
| 32740101 | 38.13 | `'dep'` | `'▁loans'` (0.430) | PROSE | `ems if useful long-lasting functions. the particular owner and driver of payday` |
| 20780721 | 37.92 | `'gers'` | `'og'` (0.999) | PROSE | `>david wellsdavid wells is one of the uk's most popular psychics and astrol` |
| 19213685 | 37.20 | `'bac'` | `'▁of'` (0.727) | PROSE | `sort of autologous stem cell therapies presently envisaged. that would be a set` |
| 14415616 | 37.20 | `'dra'` | `'▁to'` (0.203) | PROSE | `ot a part of the school community, by all means, keep it! remember, it’s “fun”` |
| 9481873 | 37.18 | `'les'` | `'▁'` (0.803) | PROSE | ` service logistics back office spare parts sales operations accessories` |
| 3000677 | 37.14 | `'about'` | `'.'` (0.594) | PROSE | `ed and charged with first-degree felony counts of murder and aggravated burglary` |
| 4857560 | 37.11 | `'▁pict'` | `'-'` (0.371) | PROSE | `championship game at macon county high. tournament mvp kenandre bates (3` |
| 16876515 | 37.08 | `'uture'` | `'ulture'` (0.816) | PROSE | `farming - a cow based farming methodology - which believes in 100% natural agric` |
| 35341058 | 37.08 | `'old'` | `'▁me'` (0.563) | PROSE | `e from the get-go to have people along the way that have really been encouraging` |
| 38950532 | 37.02 | `'allow'` | `'.'` (0.587) | PROSE | `was done over a 10-year period late in his life when he was relatively drug-free` |
| 11275507 | 36.98 | `'aff'` | `'ructure'` (0.747) | PROSE | `world, and they have to re-engineer their processes, retool, and sometimes, rest` |
| 17514358 | 36.83 | `'in'` | `'▁f'` (0.678) | PROSE | `de high quality services for your fencing needs. if you cannot find the perfect` |
| 12232143 | 36.80 | `'▁coast'` | `'▁co'` (0.576) | PROSE | `rted. the story played with my emotions and had me going on an emotional roller` |
| 4295024 | 36.79 | `'▁try'` | `'ll'` (0.627) | PROSE | `tions among their favored picks, but there seems to be growing speculation they'` |
| 19397712 | 36.65 | `'dur'` | `'-'` (0.732) | PROSE | `ques, hk’s extrusion department can produce continuous strip, in single or dual` |
| 16078790 | 36.61 | `'fav'` | `'ion'` (0.817) | PROSE | `loud computing will remain sexy in 2012 and until there is a major, major insert` |
| 26575488 | 36.61 | `'hensive'` | `'hes'` (0.996) | PROSE | `erent sheets of designs. content: 2 sheets/per set material: paper, pvc, ad` |
| 36538134 | 36.37 | `'▁whole'` | `'row'` (0.785) | PROSE | `nd press the auto retaliate button. toggles on or off the fog of war: th` |
| 23226772 | 36.22 | `'atin'` | `'tain'` (0.869) | PROSE | ` weird american superheroe figures that were super deformed, a batman and cap` |
| 27647183 | 36.22 | `'tle'` | `'.'` (0.708) | PROSE | `blue, missile-size antibiotics, and i began to improve a little. but not a lot` |
| 4200388 | 36.08 | `'▁“'` | `'"'` (0.796) | PROSE | `stinks! why should established companies who have been paying their "fair-share` |
| 18037155 | 36.03 | `'fix'` | `'▁'` (0.902) | PROSE | `you save despatched within 1-3 days. - product information eylure superfix` |
| 17592633 | 36.03 | `'▁brought'` | `'ie'` (0.837) | PROSE | ` a future ahead of him. nml needed some help on the other end of the line . ob` |
| 14350 | 36.03 | `'stery'` | `'x'` (0.998) | PROSE | `hing to the building of walls and the planning of rooms and passages in a phalan` |
| 22355552 | 35.68 | `'ight'` | `'▁independent'` (0.495) | PROSE | ` reviews at our discretion. all the submitted reviews become the property of an` |
| 38061069 | 35.50 | `'uge'` | `'ues'` (0.992) | PROSE | `nd relics. of particular note in the jewish quarter are the spanish synagog` |
| 37332139 | 35.47 | `'k'` | `'▁egg'` (0.992) | PROSE | ` a skillet, add hashbrowns and cook until brown. dig out a little hole for each` |
| 22336427 | 35.38 | `'▁debut'` | `'▁coverage'` (0.584) | PROSE | `up for bcbsnd coverage through the employer. another way to purchase insurance` |
| 31143042 | 35.32 | `'giving'` | `'▁your'` (0.220) | PROSE | `the final version is available at http://dx.doi.org/10.1215/00318108-3772018 for` |
| 22084697 | 35.29 | `'ade'` | `'ius'` (0.782) | PROSE | `e-coconut ice cream—is perhaps the most beautiful version i’ve ever seen, a gen` |
| 29447765 | 35.22 | `'om'` | `'mon'` (0.997) | PROSE | `in church at 3:13 mark) (another example of john plugging his book during a ser` |
| 1068650 | 35.21 | `'er'` | `'ver'` (1.000) | PROSE | ` s. thompson's brutally honest canadian job request. applying to the vancou` |
| 30593455 | 35.16 | `'ode'` | `'elling'` (0.393) | PROSE | `ine following motor vehicle accident, dry needling of the extremities, sw` |
| 40239301 | 35.04 | `'▁attention'` | `'neys'` (0.921) | PROSE | `experience away in the neutral box so that we can ignore it and thus return jour` |
| 29900156 | 35.01 | `'▁academ'` | `'.'` (0.396) | PROSE | ` a person’s voice when it sounds rough, often because of a sore throat or a cold` |
| 31487044 | 35.01 | `'ann'` | `'.'` (0.385) | PROSE | `executive secretary of the sco, zhang deguangsorry, i do not speak chinese` |
| 10479014 | 34.95 | `'com'` | `'.'` (0.530) | PROSE | `material. it definitely gives elegance and class if seen tagged on your luggage` |
| 25281581 | 34.83 | `'ache'` | `'at'` (0.999) | PROSE | `lls landing will be one of only a handful of buildings in the province of sask` |

## Worst-10 per category

Where each kind of byte fails most. PPM-addressable categories (URL/NUMERIC/CODE) are exactly where PPM helps.

### URL (7624 positions, mean NLL 2.35 bits)

| pos | NLL | actual | top-1 (prob) | left context |
|---|---:|---|---|---|
| 8801125 | 24.07 | `'▁http'` | `'▁she'` (0.254) | `diana’s untimely death and of the warmth and generosity she gives to the people` |
| 31871530 | 24.01 | `'▁https'` | `'▁age'` (0.264) | `d 22 years of age. they found that immediately following the minimum authorized` |
| 33996505 | 22.19 | `'▁https'` | `'▁ch'` (0.184) | ` and we have now gone a record 15 straight years without a repeat world series` |
| 31389857 | 21.43 | `'▁http'` | `'▁to'` (0.844) | ` late one night and then sleeping in excessively the next day is a sure-fire way` |
| 18692198 | 21.15 | `'▁http'` | `'▁'` (0.158) | ` this game! as a thank you for being so patient, we have decided to release our` |
| 33996900 | 21.02 | `'▁https'` | `'▁'` (0.584) | `ers 29. pepsi already has significant presence as a cornerstone partner, in the` |
| 13138504 | 20.81 | `'▁http'` | `'.'` (0.511) | `g at the western conservative summit on sabbath i knew that i had my proof` |
| 31549101 | 20.57 | `'▁http'` | `'▁in'` (0.143) | ` 1965, but the negations went nowhere. the sixth wto ministerial conference` |
| 974738 | 20.41 | `'▁https'` | `','` (0.360) | ` idaho on klr, marco from canada on f650gs single and pedro from redmond` |
| 31577153 | 20.28 | `'▁https'` | `'▁the'` (0.175) | ` what to do, what to do? i intend to begin learning everything i can to bring` |

### NUMERIC (1338235 positions, mean NLL 2.34 bits)

| pos | NLL | actual | top-1 (prob) | left context |
|---|---:|---|---|---|
| 12873641 | 25.20 | `'8'` | `'s'` (0.708) | `orth devon. 12 bowl – cereal or soup 12 plate – large & small 6 egg cup` |
| 32419579 | 25.19 | `'0'` | `'▁our'` (0.408) | `uch that the idea of a god who is angry with sin is something that is outside of` |
| 7563110 | 24.49 | `'4'` | `'▁says'` (0.216) | ` used. (see education week, sept. 12, 1990.) "when appropriate," the report` |
| 32115543 | 24.27 | `'6'` | `'▁are'` (0.167) | `which will help the passengers to assure their safety and security whenever they` |
| 19248588 | 24.18 | `'7'` | `'▁'` (0.992) | `scene assessing the situation, police said. copyright <0xC2><0xA9> 2013 yahoo!` |
| 30643453 | 23.68 | `'1'` | `'▁by'` (0.389) | ` bridegroom. when the time finally came at midnight, and there was a “cry made` |
| 15302634 | 23.40 | `'0'` | `'),'` (0.485) | `xplore other great areas of ecuador (such as the amazon or galapagos islands` |
| 16162931 | 23.35 | `'3'` | `'ainless'` (0.544) | `ily salted roadways. 304 stainless steel is the most common form of stainless st` |
| 28955256 | 22.89 | `'2'` | `'▁a'` (0.141) | `w things, i'm always trying to imagine how it would look as a photograph. it's` |
| 12631622 | 22.83 | `'8'` | `'uk'` (0.776) | `lls from whole brain tissues of adult mice.curr protoc stem cell biole` |

### CODE (33508 positions, mean NLL 4.91 bits)

| pos | NLL | actual | top-1 (prob) | left context |
|---|---:|---|---|---|
| 8511273 | 23.88 | `';'` | `'’'` (0.327) | `to anyone interested in learning more about other cultures and time periods. it` |
| 26519313 | 23.76 | `';'` | `'year'` (0.707) | `through world culture and history in word, image and sound. based upon the 160–` |
| 20525395 | 23.19 | `');'` | `'\ue001'` (0.772) | `yler young, t., "the iranian migration into the zagros," iran v (1967); ` |
| 8431421 | 22.78 | `';'` | `'▁is'` (0.312) | `ve it. and we know the feeling. we are fans of music as well. we know what it` |
| 24427777 | 22.66 | `';'` | `'ll'` (0.559) | `> related: 6 hotel ideas you can rob to the home’s food quotient you’` |
| 5374564 | 22.00 | `');'` | `','` (0.473) | `ly 12 note to have a dome! a very good looking building. we ... ... 51, 51, 51` |
| 6059381 | 22.00 | `';'` | `'1'` (0.253) | `ntrance is on south side of the building. dates: (fridays & saturdays <0x40> ` |
| 34637611 | 21.64 | `');'` | `'▁'` (0.730) | `censing during amoa's international expo in las vegas (sept. 30 , oct. 2,` |
| 8587333 | 20.90 | `';'` | `'rect'` (0.100) | `estival. they attacked american-indian protesters who had hung columbus in e` |
| 8431497 | 20.31 | `';'` | `"'"` (0.598) | `up right now, which is going to be the exact opposite of the opposites one. it` |

### HEX (23391 positions, mean NLL 5.83 bits)

| pos | NLL | actual | top-1 (prob) | left context |
|---|---:|---|---|---|
| 5862903 | 30.67 | `'▁face'` | `'ro'` (0.650) | `k poorly of any humman being have put the horse before the cart. they accept ar` |
| 15735803 | 30.30 | `'ffee'` | `'▁vulner'` (0.249) | ` vary due to the nature of disability. a womn who is visually impaired can be a` |
| 16903378 | 27.28 | `'▁dead'` | `'est'` (0.751) | `l. at downholme, we strive to keep people looking their best, even in the cold` |
| 25546386 | 26.37 | `'▁beef'` | `'▁attorney'` (0.185) | ` case. when a lawsuit has incorporated medical experts and witnesses, then that` |
| 37994128 | 26.09 | `'dead'` | `'▁'` (0.164) | `ganic chemistry from the university of san carlos in guatemala.<s>construc` |
| 27626201 | 25.02 | `'▁beef'` | `'▁'` (0.940) | `veterinary medical license: 5211 texas department of criminal justice and` |
| 9866393 | 24.78 | `'▁faced'` | `'▁the'` (0.091) | `rth the fees. like any other full time job, the mua the same cost of living to` |
| 30935033 | 24.78 | `'▁added'` | `'▁are'` (0.383) | `s much colder than the air. as a general rule, if the air and water temperature` |
| 25567811 | 24.55 | `'▁dead'` | `'▁of'` (0.511) | `er: 60 million metric tons or about 66 million u.s. tons. that's an awful lot` |
| 20862312 | 24.55 | `'face'` | `'force'` (0.926) | `ees across the island of ireland. <0xC2><0xB7> the majority of the intel work` |

### PATH (48124 positions, mean NLL 3.82 bits)

| pos | NLL | actual | top-1 (prob) | left context |
|---|---:|---|---|---|
| 1868022 | 24.90 | `'//'` | `'▁'` (0.992) | `g some big special reports. we’re gonna put a lot of work into these next week.` |
| 38802492 | 23.15 | `'▁/'` | `'▁children'` (0.213) | `revention sudden infant death syndrome (sids) is the primary cause of death for` |
| 31002083 | 21.40 | `'//'` | `'▁'` (0.525) | `ooking 0047 701 77 480 booking in germany 0049 40 54 76 52 94 in the news` |
| 2512628 | 21.36 | `'/'` | `'▁this'` (0.976) | `ice: $14.99 your price: $11.95 (worth 1,195 funagain points!) notify me if` |
| 1867744 | 20.90 | `'▁/'` | `'’'` (0.398) | ` nature are concerned. why don’t they have a problem with all these things? we` |
| 6663049 | 20.64 | `'//'` | `'▁'` (0.944) | `g finding the perfect rocker boyfriends whilest getting the hell out of bodeen.` |
| 4343702 | 20.53 | `'▁/'` | `'▁'` (0.901) | `er must have been there. so what had happened? why hadn’t duo been mentioned?` |
| 38751856 | 20.41 | `'/'` | `'▁'` (0.977) | `ay card you can send your friends and loved ones from the mail box on this page.` |
| 34818013 | 20.10 | `'▁/'` | `'▁the'` (0.106) | `article on the popular slashdot news service. the term is quite widely used by` |
| 37992770 | 20.06 | `'/'` | `'▁-'` (0.553) | ` other. in general, 1 mhz is the recommended minimum spacing between systems.` |

### PROSE (35160215 positions, mean NLL 3.70 bits)

| pos | NLL | actual | top-1 (prob) | left context |
|---|---:|---|---|---|
| 19975736 | 42.96 | `'net'` | `'▁'` (0.617) | `aquatic therapy for newborns also known as neonatal hydrotherapy aquatic` |
| 33543940 | 42.48 | `'ale'` | `'▁use'` (0.441) | `><0x91><0xA4>i like this. she loves it, i gave it as a gift. it's so easy to` |
| 30144567 | 42.11 | `'oil'` | `'▁site'` (0.554) | `e the following about links and posts on this site: any/all of the links on the` |
| 38454641 | 41.72 | `'other'` | `'▁my'` (0.342) | `on wednesday, sept. 18, the friends of the art circle library will host a` |
| 17394259 | 41.31 | `'stage'` | `'ipp'` (0.990) | `y for the newspaper philippine daily inquirer. ang was a juror for the phil` |
| 15566987 | 41.04 | `'night'` | `'y'` (0.994) | `offers in paris, paris, paris, paris, paris, paris, paris. rendez-bount` |
| 30511920 | 40.61 | `'i'` | `'▁appropriate'` (0.218) | `e. it is the duty of any person or buyer using this mls listing to perform the` |
| 39246260 | 39.44 | `'▁fly'` | `'re'` (0.435) | `he dmz and at the same time saigon was being threatened from the west. so we’` |
| 35366293 | 39.17 | `'els'` | `'ful'` (0.425) | ` specific overt act that was a substantial step toward re-entry, constitute harm` |
| 21875795 | 39.14 | `'another'` | `'.'` (0.610) | `it accepts most interchangeable pentax 35-mm film camera lenses and accessories` |

## Top-25 best NN predictions (contrast)

| pos | NLL (bits) | actual | category | left context |
|---|---:|---|---|---|
| 3295343 | 0.00 | `'\ue001'` | PROSE | `out / change ) you are commenting using your google<0x2B> account. ( log ` |
| 5516381 | 0.00 | `'\ue001'` | PROSE | `out / change ) you are commenting using your google<0x2B> account. ( log ` |
| 38924675 | 0.00 | `'\ue001'` | PROSE | `by rampage jackson in forum cycle inforeplies: 24last ` |
| 39232514 | 0.00 | `'\ue001'` | PROSE | `13: published: mar 21, 2016. publicly ` |
| 39941280 | 0.00 | `'s'` | PROSE | ` the queens warrant squad. get the this week'` |
| 35482992 | 0.00 | `'ed'` | PROSE | `ars of conflict, with ocalan finally in custody, popular frustration was uncork` |
| 39232308 | 0.00 | `'\ue001'` | PROSE | `75: published: feb 25, 2016. publicly ` |
| 23810887 | 0.00 | `'th'` | PROSE | `: 3 - fixed door bins: 1, full-wid` |
| 34192177 | 0.00 | `'\ue001'` | PROSE | `physician or other health providers posting on or otherwise referred to on this ` |
| 5124326 | 0.00 | `'\ue001'` | PROSE | `mw 5 series forum (e34)replies: 0last ` |
| 5124386 | 0.00 | `'\ue001'` | PROSE | `mw 3 series forum (e46)replies: 0last ` |
| 6675733 | 0.00 | `'\ue001'` | PROSE | `working papers 0055, national bureau of economic ` |
| 6675175 | 0.00 | `'\ue001'` | PROSE | `working papers 4304, national bureau of economic ` |
| 31700214 | 0.00 | `'\ue001'` | PROSE | `ology, washington university, st. louis, mo, united ` |
| 23906536 | 0.00 | `'\ue001'` | PROSE | ` have decided to go through with filing a lawsuit in a albuquerque, new ` |
| 3235465 | 0.00 | `'\ue001'` | PROSE | ` colorado, georgia, maryland, massachusetts, new ` |
| 38700698 | 0.00 | `'elt'` | PROSE | `25 percent in this case - died among the 12 mothers who were not wearing a seatb` |
| 14046733 | 0.00 | `'pected'` | PROSE | `admission to the nh <0x5B>nursing home<0x5D> solely based on a confirmed or sus` |
| 37065753 | 0.00 | `'.'` | PROSE | `tact the editor responsible for this story: robin ajello at email<0x40>example` |
| 23310948 | 0.00 | `'\ue001'` | PROSE | `out / change ) you are commenting using your google<0x2B> account. ( log ` |
| 29233705 | 0.00 | `'t'` | PROSE | ` to a companywide level.”<s>what people are saying - write a review we haven'` |
| 7360047 | 0.00 | `'\ue001'` | PROSE | `working papers 4084, national bureau of economic ` |
| 23906286 | 0.00 | `'\ue001'` | PROSE | ` you should not make a final decision without talking to a albuquerque, new ` |
| 5834152 | 0.00 | `'\ue001'` | PROSE | ` gnomey in forum apple notebooksreplies: 15last ` |
| 35449306 | 0.00 | `'\ue001'` | PROSE | `physician or other health providers posting on or otherwise referred to on this ` |