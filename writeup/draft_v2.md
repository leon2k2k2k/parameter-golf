# Parameter Golf: Six Weeks to Build the Best LLM

---

> **What does a 14% compression improvement actually look like?**
> Both models below were trained on the same dataset, in the same 10 minutes,
> on the same hardware. The only difference is what the community built in six weeks.

*From the FineWeb validation set, same 150-token context fed to both models:*

| | SP1024 Baseline · 1.22 BPB | Near-SOTA Model · 1.06 BPB |
|---|---|---|
| *"…his driver license and auto insurance had been cancelled in October. This poses a problem for"* | the estate, though internalieness to gets the pass. It is a fat check that's gives ges auto insurance company… that's a **fairyyyyyyyyyyyyyyyyyyyyy** | the estate's creditors who need to work out a payment, especially with the new insurance regulations in october… |
| *"…New host Steve Byrnes was feeling his way through this"* | season. He had just had a short, **bad weekkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk** | season's show, and while he played one of the best nights of his career at the same time, he hadn't been able to really shine since the first Nashville game… |
| *"…Prosecutors now have until"* | Jan. 2, the **Supreen House** prosecutor sentences on a **murded** case… **Mazzzzaglia**… **charies's''''''** | Dec. 18 to sign off on a case after the new hampshire attorney general's office says they have more time… |

*Prompted with "You are a helpful AI assistant. Please answer the following question." (not very good at general conversation):*

| | SP1024 Baseline · 1.22 BPB | Near-SOTA Model · 1.06 BPB |
|---|---|---|
| *"The history of the internet began in the 1960s when"* | the Netherlands had established a strong supply of information about the **Canadian government and its supply chains**. The German government, in its own right, is commonly established as a matter of legal… | the internet became the **first electronic commerce tool** that was used by many people, including children, to connect online. the internet is now used by more than 3 billion users worldwide… |
| *"What is 15 multiplied by 13? Let me work it out:"* | **1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 14. 13.** 14. 15. 16. 17. 18. 19. **19.** 20.1 | 15 multiples 13 have 15, for example, 15, for **1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000** |
| *"To bake a chocolate cake, you need flour, eggs, butter, and"* | sugar… you need to have **an infinite wooden cake**. Otherwise, you need to be taken as close to an… | salt… Add 1 tablespoon of flour to 1/4 cup of milk and pour over your cake. Add 1/4 teaspoon of sugar… Cook a lightly toasted chocolate cake. Make cupcakes Ã 3 |

---

In March 2026, OpenAI released a public competition with a deceptively simple premise: train the best language model you can, but it has to fit in 16,000,000 bytes, and you only get 10 minutes of training time on 8 H100s. Call it parameter golf — every byte counts, every second counts.

What followed over the next six weeks looked, from the outside, like just another competitive coding challenge. It turned out to be something more: techniques stacking on each other in ways nobody planned, innovations that shouldn't have worked but did, controversial submissions that looked like miracles, mayhem on the last day, and a picture-perfect finish. In the end, starting from a model that produces the gibberish above, the community built one that speaks coherently — just don't ask it anything that isn't in the training data. This post goes through some of the highlights — both the technical and the dramatic. 

---

## 1. The Competition

At the core of this competition is a simple question: how well can a model predict text?

A language model is, at its heart, a probability distribution over text. Given a sequence of words — or more precisely, tokens — the model assigns a probability to every possible next token. A well-trained model should assign high probability to tokens that actually appear in real text, and low probability to tokens that don't. Think of it like a well-read person trying to complete sentences. Given "The president signed the —", they would confidently predict "bill" or "order" and be surprised by "banana." A bad model treats every next word as equally likely. A good model has internalized the patterns of language well enough to be right, or at least close, most of the time.

The models in this competition are trained and scored on FineWeb, a large dataset of cleaned web text. The score is computed on a held-out validation slice that the models never see during training. For each token in that slice, we ask: what probability did the model assign to the token that actually appeared? The cost for a single token is $-\log_2 p(t)$, where $t$ is the correct token. If the model is perfectly confident — $p(t) = 1$ — it pays zero cost. If it assigns $p(t) = 0.5$, it pays 1 bit. If it assigns $p(t) = 0.01$, it pays about 6.6 bits. The total score is this cost summed across all tokens, normalized by the number of bytes in the original text:

$$\text{BPB} = \frac{-\sum_k \log_2 p(t_k \mid t_1, \ldots, t_{k-1})}{\text{number of bytes}}$$

This is called bits-per-byte (BPB). To put it in perspective: a model that assigns completely uniform probability across all 256 possible bytes — knowing nothing at all — scores exactly 8 BPB. The baseline OpenAI provided started at **1.2244 BPB**, already far below that, meaning the model had learned real structure in language. Six weeks later, the community had pushed it to **1.0565** — a 14% reduction, achieved purely through algorithmic improvements with no change to the hardware or the data.

---

## 2. The Model

*(draft pending)*

---

## 3. Too Good to Be True

*(draft pending)*

---

## 4. Drama on the Last Day

*(draft pending)*
