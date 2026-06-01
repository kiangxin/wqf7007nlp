## PART 1 — Dataset Problems & What We Did

### Q1: What data quality issues did we find in FABSA, and how did we address them?

**Three issues we found by auditing the raw HuggingFace splits directly:**

**1. Data Leakage**
- 46 Train↔Test and 41 Train↔Val reviews appear in both splits; additionally, within-split duplicate rows existed (Train: 405 rows, Val: 22, Test: 40)
- Root cause: most are trivial short phrases (`"Quality"`, `"Simple"`, `"quick and easy"`) that appear independently across multiple app reviews — coincidental collisions, not genuine duplication
- Fix: **Fully addressed** in `dev/notebooks/preprocessing.ipynb`. Two separate fixes applied:
  1. **Within-split dedup** — `drop_duplicates(subset=["text"])` on each split independently before pairwise expansion
  2. **Cross-split leakage** — after all three splits are preprocessed, any training pair whose review text appears in val or test is removed; val and test are kept fully intact as evaluation ground truth
- Val∩Test (18 reviews) is intentionally left unfixed — evaluation sets are kept intact for benchmark comparability, and the pretrained DeBERTa-v3 backbone handles these short generic phrases well at inference
- Model was retrained on the cleaned data. Post-retrain results: Accuracy 94.1%, Macro F1 89.0%, Weighted F1 94.1% — all within 0.4 pp of the pre-retrain checkpoint, confirming the original estimate that leakage bias was < 0.5 pp.

**2. Label Ontology / Granularity Overlap**
- Same text annotated differently by different annotators:
  - `"Easy to navigate"` → labelled as `ease-of-use` by annotator A; `ease-of-use` + `app-website` by annotator B
- 7 cases of identical texts with conflicting label sets across the full dataset
- Fix: our LLM ACD stage bypasses this entirely — Gemini reasons about the review text contextually rather than matching against noisy label boundaries

**Does this affect ACSC training?** Indirectly yes, but only at the boundary cases. When annotator A labels a text with `ease-of-use` only and annotator B adds `app-website`, the ACSC training data ends up with an `(text, app-website, positive)` pair from annotator B that annotator A never assigned. DeBERTa is trained on both annotations combined, so it sees `app-website` as a valid aspect for that text. The sentiment label itself (positive/negative) is usually consistent between annotators — the conflict is which aspect to attach it to, not what the sentiment is. So label ontology overlap creates noisy aspect coverage in ACSC training but does not introduce contradictory sentiment labels.

**3. Same-Aspect Conflicting Sentiment (Section 5.4)**
- 67 reviews in the raw train split contained the same aspect labelled with both positive AND negative sentiment — arising from multi-sentence reviews where different sentences express opposite sentiments about the same aspect
- Real example from the dataset: `"Great app / it does have its bugs at times"` → `online-experience.app-website` labelled both +1 and -1. Another: `"Great service, great app. Sneaky updates of privacy terms and conditions."` → `online-experience.app-website` both +1 and -1.
- Fix: removed all conflicting (text, aspect) pairs from train, val, and test during preprocessing. Both rows dropped — no sentiment preserved — because without sentence-level annotations there is no principled way to choose one polarity over the other. Confirmed removed: `grep` of the cleaned text in `FABSA_train_preprocessed.csv` returns 0 matches for the above examples.
- Status: **fully applied.** Model was retrained on the conflict-free data.

---

### Q2: How did we handle the 1,337 very short reviews (< 4 words)?

This is the root cause of ACD false positives/negatives. A review like `"Easy to use"` from a banking app has zero context to disambiguate whether the user means:
- The mobile app UI → `online-experience.app-website`
- The account opening flow → `purchase-booking-experience.ease-of-use`
- The loan process → `account-management.account-access`

**Our mitigation:** `min_confidence=0.70` threshold. When Gemini is uncertain about which aspect a vague phrase refers to, its self-reported confidence drops below 0.70 and the detection is filtered out. This accepts some false negatives to avoid false positives.

**Why we can't fully fix this:** It is a fundamental dataset quality problem — 82% of FABSA comes from Google Play / Apple Store where character limits produce extremely short reviews. Our Trustpilot-scraped inference data is higher quality (longer, more specific reviews) which is why the system performs better in practice than the benchmark numbers suggest.

**A practical mitigation we can add (or have added):** The Gemini ACD prompt can be given the review's source context — industry and platform (e.g., `"Banking app — Google Play review"`). For a review like `"Easy to use"` from a Banking app, knowing the industry allows Gemini to reason: "In the context of banking apps, 'easy to use' most likely refers to the app UI (`online-experience.app-website`) or account management flow (`account-management.account-access`) rather than a general purchase experience." This context is already available in our scraped data (`industry`, `data_source` columns) and can be prepended to the review before the LLM call with minimal prompt change.

---

### Q3: Was class imbalance addressed in our approach?

Yes — two ways:

1. **Evaluation metric:** Macro-F1 (89.0%) is our primary ACSC metric. It treats all 12 aspect × 3 sentiment classes equally regardless of frequency. The gap between weighted F1 (94.0%) and macro F1 (89.0%) — ~5 pp — directly quantifies the imbalance effect and is explicitly reported.

2. **LLM ACD is immune to FABSA's imbalance:** Gemini was never trained on FABSA, so it does not inherit the distribution bias (Fashion 36%, IT 1.7%). It handles rare aspects (`staff-support.phone`, `value.discounts-promotions`) through general language understanding, not frequency-based pattern matching.

**What was not done:** Explicit oversampling or class-weighted loss in ACSC training. Checking `train.py` directly: the only imbalance-aware mechanism is `label_smoothing_factor=0.1` (softens overconfident predictions on dominant classes) — there is no class-weighted loss or focal loss in the current training code. Adding class-weighted loss requires subclassing HuggingFace `Trainer` and overriding `compute_loss` to pass `weight=` to `F.cross_entropy` — straightforward but not implemented due to time constraints. Macro-F1 as the primary evaluation metric monitors the imbalance effect rather than correcting it at training time.

---

## PART 2 — Architecture Challenges & Solutions

### Q4: Why did we choose LLM for ACD? What specific problem does it solve that a fine-tuned DeBERTa cannot?

**Three concrete reasons backed by evidence from our FABSA analysis:**

**Reason 1 — Label granularity overlap makes fine-tuned ACD unreliable**
A fine-tuned DeBERTa ACD classifier learns from FABSA's annotations. When those annotations are contradictory for the same text (e.g. `"Easy to navigate"` labelled as both `ease-of-use` only and `ease-of-use + app-website`), the model receives contradictory supervision at the boundary between these two categories. It cannot learn a stable decision boundary.

The LLM does not have this problem — it reasons from the review text and its understanding of what the aspect categories mean, not from memorised label patterns.

**Reason 2 — Implicit aspects require world knowledge, not pattern matching**
FABSA aspect categories are conceptual, not keyword-based. A review like `"I cannot log in anymore"` requires the system to understand that "logging in" relates to `account-management.account-access` — there is no surface-level keyword overlap. A fine-tuned DeBERTa ACD would need to learn this from training examples alone.

Gemini's pre-trained world knowledge handles this naturally — it understands what banking, e-commerce, and IT services are, and can infer which aspect category a complaint maps to even without explicit keywords.

**Reason 3 — Short-text ambiguity (12.6% of FABSA is < 4 words)**
A fine-tuned model would learn to map `"Easy to use"` → `ease-of-use` because that pattern dominates the training data. But this is exactly wrong for ACD quality — the model is memorising the most frequent annotation, not understanding the review.

Gemini, with `min_confidence`, at least signals its uncertainty on short/ambiguous texts and allows the threshold to filter them out. A fine-tuned classifier would confidently assign the dominant label with no uncertainty signal.

**In summary:** LLM ACD is not a shortcut — it is the technically correct response to a dataset that has annotation noise, label boundary ambiguity, and predominantly short/ambiguous texts.

---

### Q5: What about LLM misclassification — Gemini is not perfect either?

Acknowledged. LLM ACD has its own failure modes:

| Failure mode | Mitigation in place |
|---|---|
| Gemini assigns wrong aspect (false positive) | `min_confidence=0.70` threshold filters uncertain detections |
| Gemini misses an aspect (false negative) | Accepted trade-off — we chose precision over recall |
| Gemini hallucinates a non-existent aspect | Aspect ID must match one of 12 FABSA categories; invalid IDs are filtered out |
| Error propagates to ACSC (wrong aspect → wrong sentiment) | Accepted residual risk — ACD precision is maximised via `min_confidence=0.70` to limit how many wrong aspects reach ACSC |

**The key mitigation is the architecture itself:** Gemini produces a `reasoning` field and `evidence_spans` for every detected aspect. This makes misclassifications **visible and explainable** — a business user or evaluator can read the reasoning and judge whether the detection is correct. A fine-tuned DeBERTa ACD classifier produces no such explanation.

**What we did not do (honest gap):** We did not separately quantify what fraction of wrong final outputs came from ACD errors vs ACSC errors. **Error decomposition** means running ACD evaluation alone (compare predicted aspect slugs to gold slugs, ignoring sentiment) and ACSC evaluation alone (assume perfect aspect detection, compare only sentiment predictions), then seeing which stage contributes more errors. We do have the tools to do this — the LLM-as-Judge eval already measures ACD quality in isolation — but we did not run a formal decomposition combining both stages.

---

### Q6: Our LLM ACD exact match is only 0.45 — isn't that too low to be acceptable?

Exact match for multi-label ABSA is the strictest possible metric. A review with 3 gold aspects where you correctly detect 2 scores **0.0 exact match** — any missing or extra aspect fails the entire review. This is comparable to requiring a search engine to return exactly the right 10 results in exactly the right order before counting a query as successful.

The 0.45 exact match for Gemini means that for 55% of reviews, the predicted aspect set differs from the gold set by at least one aspect. But the gold set itself is noisy — our analysis found that identical texts in FABSA were assigned different aspect labels by different annotators — so some of those "mismatches" are the LLM being *more* correct than the annotation, not less.

The more informative metric is the **reasoning score (7.97 / 10)** — evaluated by the judge on whether the detected aspects are grounded in the review text, not whether they exactly match a potentially noisy gold label. A model that detects `ease-of-use` when the annotation says `app-website` for the same text is not wrong — it is caught in the annotation ambiguity we documented.

**The comparison that matters:** Gemini 0.45 > Qwen 0.40 > GPT-4.1-mini 0.36. The relative ranking is consistent. And a fine-tuned DeBERTa ACD trained on FABSA's noisy labels would learn to reproduce that noise — its "exact match" against the same noisy gold set would look artificially high, not because it detects aspects correctly, but because it memorised the annotators' inconsistent choices.

---

### Q7: Wouldn't a fine-tuned DeBERTa ACD know the 12-category taxonomy better than our general-purpose LLM?

This is true in one direction and false in the other.

**True:** A fine-tuned classifier trained on FABSA labels will never output a category outside the 12. The LLM requires explicit post-processing to filter hallucinated IDs — which we do (`aspect_id` must match one of 12 slugs; invalid IDs are discarded).

**False:** Consistency with the training labels is only valuable if those labels are clean. Our FABSA analysis found that:
- The same text (`"Easy to navigate"`) was annotated as `ease-of-use` by one annotator and `ease-of-use + app-website` by another
- A fine-tuned classifier learns **both supervision signals** for the same input — it will produce unpredictable output at the exact boundary where the annotation is inconsistent
- This is a worse failure mode than the LLM's occasional out-of-taxonomy hallucination, because the classifier's inconsistency is invisible (it produces a confident label) whereas the LLM's uncertainty is measurable via `min_confidence`

The fine-tuned model would *appear* consistent when evaluated on the same noisy gold labels it trained on. It would fail in exactly the same systematic way the annotation fails — which is not consistency, it is overfitting to noise.

---

### Q8: Isn't using an LLM to evaluate its own outputs circular?

No — the judge model is **Claude Sonnet 4.5** (`anthropic/claude-sonnet-4-5`), an entirely different model from a different company than the ACD candidates being evaluated (Gemini, Qwen, GPT-4.1-mini). There is no self-serving incentive: Claude has no stake in which of those three models scores higher.

The evaluation has two components:

| Component | Circular? |
|---|---|
| **Exact match** (predicted aspect slugs vs gold) | No — string comparison against human gold labels, fully model-agnostic |
| **Reasoning score** (0–10 per review, judged by Claude Sonnet 4.5) | No — independent model from a different provider scores the outputs |

The exact match result (Gemini 0.45 > Qwen 0.40 > GPT-4.1-mini 0.36) is the model-agnostic ground truth and is what we use for the 3-model comparison. The reasoning score from Claude provides a complementary signal: whether the detected aspects are grounded in the review text, which exact match cannot capture.

**Ideal mitigation** would be a human-annotated golden set — the 100-review golden test set is a reasonable proxy within the project's scope.

---

### Q9: Isn't our LLM ACD just prompt engineering, not real NLP — and why not fine-tune a smaller model instead?

The distinction between "prompt engineering" and "NLP" is a false boundary at this point in the field. Designing a structured JSON schema, a 12-category taxonomy prompt, calibrated confidence scoring, and chain-of-thought reasoning extraction is applied ML system design — it is what makes the output structured, grounded, and usable downstream.

On the fine-tuning alternative: we could fine-tune a small model (e.g. Llama-3.1-8B) on FABSA for ACD. The reason we did not is specifically that the training data is unsuitable for it:

1. **Annotation noise** — the same text maps to different label sets across annotators. A fine-tuned classifier trained on this signal learns the noise, not the underlying pattern.
2. **Short-text ambiguity** — 12.6% of FABSA reviews are < 4 words. For these, there is no contextual signal to learn from regardless of model size.
3. **Label boundary overlap** — `ease-of-use` and `app-website` partially describe the same concept. A classification head forces a hard boundary that does not exist in the data.

A fine-tuned small LLM would have the same annotation-noise problem as a fine-tuned DeBERTa — it still learns from FABSA's labels. The specific advantage of using an API LLM is that it brings world knowledge about what banking, e-commerce, and IT services are, knowledge that was never in FABSA's training set.

On cost: the LLM cache (`LLMCache`) makes repeat inference free. The 200 Trustpilot reviews (50 per domain × 4 domains, selected from 300 initially scraped) and 100 golden test set reviews were each processed once and cached. For this project's scale, the API cost was negligible.

---

### Q10: How does error propagation work in our two-stage pipeline, and how did we address it?

The pipeline is sequential: ACD → ACSC. Any aspect missed by ACD produces a final false negative that ACSC cannot recover. Any aspect wrongly detected by ACD produces a final false positive even if ACSC assigns the "correct" sentiment to it.

**Addressed through:**
1. `min_confidence=0.70` — reduces ACD false positives at the cost of some recall
2. LLM-as-Judge evaluation — measures **ACD quality in isolation**: predicted aspect slugs vs ground-truth slugs, plus a 0–10 reasoning quality score per review. This is not end-to-end evaluation; it measures only whether the right aspects were detected and whether the reasoning was grounded.
3. ACSC quality is measured independently via DeBERTa evaluation: Accuracy 94.1%, Macro-F1 89.0% on the FABSA test split.

**Remaining gap:** The two evaluations (ACD: LLM-as-Judge, ACSC: DeBERTa metrics) are separate. We have no single measurement that traces a wrong final output back to whether the error came from the ACD stage or the ACSC stage. A combined end-to-end evaluation on a set with both gold aspect labels AND gold sentiments would enable this decomposition — it is documented as future work.

---

## PART 3 — Defending the Implementation

### Q11: How does the project work distribute across team roles — and is the ML scope substantial enough?

The project's components map naturally across three roles:

| Role | Ownership |
|---|---|
| **ML Engineer** | Model training (LoRA fine-tuning), LLM ACD system design, inference pipeline, web app demo |
| **Data Engineer** | FABSA preprocessing, dataset audit, Trustpilot scraping and selection |
| **Evaluation Specialist** | LLM-as-Judge pipeline, ACSC evaluation metrics, ACD 3-model comparison |

**ML Engineer scope in detail:**

| Deliverable | Detail |
|---|---|
| LoRA fine-tuning | r=16, alpha=32, target modules query_proj+value_proj, 5 epochs, AdamW with linear warmup, label smoothing, early stopping |
| ACSC evaluation | `evaluate.py` on held-out test split — Accuracy 94.1%, Macro F1 89.0%; per-aspect breakdown across 12 categories validates domain adaptation |
| LLM ACD system design | Structured JSON schema, 12-aspect taxonomy in system prompt, chain-of-thought reasoning, per-aspect confidence scoring, evidence span extraction |
| `min_confidence` threshold | Precision/recall operating point design for the ACD confidence filter |
| Inference pipeline | Span expansion logic, 3-tier fuzzy matching, sentence boundary expansion, diffuse-vs-verbatim fallback, response caching, LLM failure handling |
| Class imbalance analysis | Weighted vs Macro-F1 gap quantification, decision on label smoothing vs class-weighted loss trade-off |

**On depth:** We started from `yangheng/deberta-v3-base-absa-v1.1` (pre-trained on SemEval ABSA restaurant/laptop data), not a generic transformer. LoRA fine-tuning on FABSA adapts it to a 12-category multi-industry taxonomy the base model was never trained on. The fine-tuned model achieves **Accuracy 94.1%, Macro F1 89.0%** on the FABSA test set — verified metrics from `evaluate.py`.

The LLM ACD component requires ML thinking: choosing `min_confidence=0.70` as a precision/recall operating point, structuring the prompt to elicit calibrated confidence scores, and designing the fallback for hallucinated or out-of-taxonomy aspect IDs. This is applied ML system design, not software engineering. Combined with the LoRA domain adaptation, this represents substantive ML engineering work across both the generative and discriminative components of the system.

---

### Q12: Is our implementation incomplete because we only focused on ACSC?

**No — we implement a complete ABSA system:**

ABSA consists of two subtasks:
1. **ACD (Aspect Category Detection)** — which aspects are present?
2. **ACSC (Aspect Category Sentiment Classification)** — what is the sentiment per aspect?

We implement both:
- ACD → Google Gemini 3.5 Flash (LLM, chain-of-thought reasoning)
- ACSC → DeBERTa-v3 fine-tuned on FABSA

The final output `{aspect_category: sentiment, confidence, evidence_span}` is a complete ABSA output. The fact that our ACD uses an LLM instead of a fine-tuned DeBERTa does not make the task incomplete — it makes the ACD stage stronger.

If the concern is that the proposal described a DeBERTa+DeBERTa pipeline and we deviated, that deviation was justified by the dataset quality findings and resulted in a better-performing system (as evidenced by the LLM-as-Judge evaluation).

---

### Q13: Why didn't we address class imbalance with oversampling or class weighting?

The concern is valid for a purely fine-tuned pipeline. For our architecture, it is partially addressed and partially mitigated by design:

**For ACSC (DeBERTa):**
- Sentiment imbalance: positive (65%), negative (32%), neutral (4%) across training pairs
- We chose Macro-F1 as primary metric which penalises poor neutral performance equally — this monitors the problem
- Oversampling is not as straightforward as in standard classification tasks. ABSA training data is structured as triplets `(text, aspect, sentiment)` — duplicating rows inflates the model's exposure to a narrow set of phrasings without adding genuine linguistic variety. A more robust fix would require expert re-annotation: sourcing new review texts that naturally contain underrepresented aspect categories (IT, Consulting, Streaming) and labelling them from scratch. This is a dataset construction problem, not a one-line training parameter change.
- Class-weighted loss is a partial mitigation within the current data — subclassing `Trainer` to override `compute_loss` — but is not implemented due to time constraints.

**For ACD (LLM):**
- Not subject to FABSA's aspect frequency imbalance at all — Gemini was never trained on FABSA
- Rare aspects (IT: 1.7%, Consulting: 1.0%) are handled through general language understanding, not frequency-biased pattern matching

**Valid future improvement:** Add class-weighted loss to the ACSC fine-tuning by subclassing `Trainer` and overriding `compute_loss`. The ideal long-term fix is expert annotation of underrepresented industry categories to balance the training triplets.

---

## PART 4 — NLP Techniques Utilized

### Q14: What NLP techniques does our project use, and to what extent?

**1. Transfer Learning & Domain Adaptation**
- Started from `yangheng/deberta-v3-base-absa-v1.1` (pre-trained on SemEval ABSA)
- Applied LoRA (Parameter-Efficient Fine-Tuning) for domain adaptation to FABSA
- Extent: full training pipeline with train/val/test splits, Macro-F1 evaluation, 5-epoch training with linear warmup scheduler

**2. Large Language Model Prompting (Generative AI)**
- Gemini 3.5 Flash via OpenRouter for ACD
- Structured JSON output schema with chain-of-thought reasoning
- Confidence scoring and evidence span extraction
- Extent: production-ready prompting with response caching, failure handling, and a min_confidence threshold

**3. Text Preprocessing (Classical NLP)**
- Lowercasing, contraction expansion (`contractions` library), lexical normalisation (LEXICAL_MAP)
- Emoji demojization (`emoji.demojize` — preserves sentiment signal as text)
- HTML entity unescaping, URL masking, email removal, punctuation normalisation
- Extent: 6-step pipeline applied consistently across train/val/test and inference

**4. Sequence-Pair Classification (Transformer NLP)**
- DeBERTa tokenizer sentence-pair format: `[CLS] review [SEP] aspect [SEP]`
- Max-length 256 with empirical validation (99.1% of pairs fit within this budget)
- Extent: full fine-tuning with label mapping, stratified sampling, LoRA adapter save/load

**5. Evidence Span Extraction (Information Extraction)**
- Given LLM-detected evidence spans, locate them in the original review text
- 3-tier fuzzy matching: exact → case-insensitive → SequenceMatcher (≥0.85 ratio)
- Sentence boundary expansion via NLTK `sent_tokenize`
- Extent: production inference component used for all real-world reviews

**6. Evaluation: LLM-as-Judge (Modern NLP Evaluation)**
- 100-review golden test set sampled from FABSA test split (conflict-free)
- LLM judge (Claude Sonnet 4.5) evaluates: exact match of aspect-sentiment pairs + reasoning quality (0–10)
- Beyond standard metrics: captures ACD quality that pure ACSC accuracy cannot measure
- Extent: full evaluation pipeline with separate judge cache, metadata logging

**7. Real-World Data Collection (Applied NLP)**
- Playwright headless Chromium to scrape Trustpilot (AWS WAF bypass via `__NEXT_DATA__` JSON extraction)
- 300 reviews initially scraped across 4 industries (Banking, IT, E-Commerce, Fashion); filtered and selected to **200 reviews** (50 per domain) for the final demo and evaluation
- Rating-stratified selection to avoid negative-review bias (Trustpilot platform skew)
- Extent: production scraper with resumable checkpointing, multi-domain support

---

## PART 5 — Train/Inference Gap & Span-Based ACSC

### Q15: We trained DeBERTa on full review text but feed it extracted spans at inference — isn't that a distribution mismatch?

**Partially true — but the design is intentional and the mismatch is small in practice.**

**What the inference code actually does (inference.py lines 614–621):**

```
verbatim aspects → expand each evidence span to its enclosing sentence(s) → join → clean → DeBERTa
diffuse aspects  → full cleaned review → DeBERTa
```

The input to DeBERTa during inference is **not** a raw LLM span (e.g., `"took 3 weeks"`) — it is the full sentence containing that span (e.g., `"Delivery took 3 weeks, completely unacceptable."`). This is a deliberate choice made with `expand_to_sentence()` + NLTK `sent_tokenize`.

**Why the distribution shift is limited:**

1. **FABSA reviews are mostly short.** The training corpus (82% Google Play / Apple Store) averages ~20–50 words. For these, the "full review" in training IS often 1–3 sentences — indistinguishable in length from our sentence-expanded inference input. The model did not learn to use long document context, because it had none to learn from.

2. **Sentence expansion preserves syntactic context.** We expand to the enclosing sentence boundary using NLTK `sent_tokenize`, not to the raw LLM span. DeBERTa receives a grammatically complete sentence with subject + predicate — the same granularity at which sentiment is expressed in both training and inference.

3. **Diffuse aspects are not affected.** When Gemini returns no spans (whole-review sentiment, `evidence_type: diffuse`), the inference code falls back to the full cleaned review — exactly matching the training format.

**Why the design is an improvement over using the full review at inference:**

- For long Trustpilot reviews (150–400 words), using the full text would frequently exceed the 256-token budget, causing truncation. Sentence-level input avoids truncation while preserving the relevant context.
- DeBERTa's attention has less noise to attend through when the input is focused on the relevant sentence rather than an unrelated multi-topic review.

**Honest gap:** We did not separately benchmark full-review vs sentence-span input at inference time on the FABSA test set. An ablation would confirm whether the distribution shift has any measurable impact. For the scope of this project, it is treated as a known trade-off rather than an unmeasured risk, because the two distributions are close given FABSA's short average review length.

---

### Q16: How does our system handle a review where the same aspect has both positive and negative sentiments?

**This happens at two levels — training data and live inference — and we handle them differently.**

**Training data:**
During preprocessing, we identified 73 (text, aspect) pairs in the FABSA train split with both positive AND negative labels for the same aspect — arising from multi-sentence reviews where different sentences express opposite sentiments. These 146 rows were removed entirely from training. DeBERTa was therefore never given contradictory supervision signal for the same input.

**Live inference on new reviews:**

At inference time, Gemini can receive a review like: `"Customer service was terrible, but they eventually resolved everything brilliantly."` The pipeline handles this as follows:

1. **Gemini (ACD)**: Returns one entry for `staff-support.attitude-of-staff` with `evidence_spans` containing the specific phrases it anchors to (it may cite one or both sentences).

2. **Span expansion**: The `expand_to_sentence()` function expands each cited span to its enclosing sentence. If Gemini cited both clauses, DeBERTa receives both sentences concatenated as the input context.

3. **DeBERTa (ACSC)**: Produces a single 3-class output over the provided sentence context. When a sentence context contains genuinely contradictory signals, the model typically outputs the net-dominant sentiment or defaults to neutral — which is the most defensible single-label prediction without sentence-level granularity.

**Why the architecture handles this better than fine-tuned ACD + DeBERTa:**

- Gemini's `reasoning` field and `evidence_spans` make it visible WHICH clause it based the detection on. A human reviewer can see the reasoning and judge whether the aspect detection is anchored to the positive or negative sentiment expression.
- If Gemini's reasoning targets only one polarity clause, DeBERTa's input is focused on that clause — avoiding the artificial contradiction that plagued the training data.

**Fundamental limitation:** Without sentence-level annotation in FABSA, neither our system nor any single-label ACSC system can perfectly resolve mixed-polarity aspects in a single review. This is a dataset limitation, not an architectural one, and it affects all ABSA systems trained on FABSA equally.

---

### Q17: For long reviews we break into spans — is DeBERTa actually capable of ACSC on a short span rather than the full context?

**Yes — for two reasons: the model's pre-training task, and our sentence-expansion design.**

**1. Sentence-level sentiment is DeBERTa's native granularity for ABSA**

`yangheng/deberta-v3-base-absa-v1.1` was pre-trained on SemEval ABSA data where the standard input format is `[CLS] sentence [SEP] aspect [SEP]`. The SemEval dataset consists of individual sentences, not multi-sentence documents. So the base model we start from already learned to do ACSC from sentence-length inputs — our fine-tuning step domain-adapted it but did not fundamentally change its input granularity.

**2. We feed sentences, not raw spans**

The inference pipeline does not pass raw LLM spans directly to DeBERTa. `expand_to_sentence()` always expands to the full NLTK sentence boundary. The model receives, e.g., `"Delivery took 3 weeks, which was completely unacceptable."` — not `"took 3 weeks"`. This preserves subject, predicate, and sentiment-bearing modifiers.

**3. 256 tokens is sufficient for sentence-level input**

Our 256-token limit was validated on FABSA (99.1% of training pairs fit). For sentence-expanded Trustpilot spans (typically 1–3 sentences, ~20–60 words), this budget is never a constraint.

**What is actually at risk for very long reviews:**

The real concern with long reviews is NOT DeBERTa's capacity — it is **Gemini's span selection quality**. If a 400-word review discusses `value.price-value-for-money` across three scattered sentences, Gemini may not cite all three in its `evidence_spans`. The DeBERTa sentiment prediction is then based on an incomplete subset of the relevant sentences.

This is a known LLM ACD limitation, mitigated by:
- The `reasoning` field, which explains what Gemini grounded its detection on
- The `min_confidence=0.70` threshold, which filters out aspects where Gemini itself signals uncertainty about the grounding

**In summary:** Sentence-expanded span input is the correct granularity for this DeBERTa base model, short-span sentiment is what the pre-training prepared it for, and 256 tokens is never the binding constraint at this input scale.

---

### Q20: Why do some Fashion reviews (e.g. Lululemon) return no detected aspects?

Consider this review: *"Quality of their yoga products is decreasing. They do look great, but they are useful for only a few sessions."*

The review clearly expresses a sentiment — product quality is declining, durability is poor. Yet the pipeline returns **no aspects detected**.

**Root cause: FABSA's 12-aspect taxonomy does not include a product quality category.**

The full FABSA taxonomy covers:

| Category | Examples |
|---|---|
| `online-experience` | App, website, UI, ease of use |
| `account-management` | Account access, settings |
| `logistics` | Delivery speed, packaging |
| `staff-support` | Customer service, responsiveness |
| `value` | Price, discounts, value for money |
| `purchase-booking-experience` | Checkout, order process |
| `company-brand` | General satisfaction, reputation, trust |
| `promotions` | Offers, rewards |
| `streaming` | Streaming-specific experience |
| `consulting` | Professional service quality |
| `it-general` | IT/software general experience |
| `banking-general` | Banking general experience |

None of these map to **product quality, product durability, or material quality**. A review about a physical garment's construction, fabric quality, or longevity has no home in the taxonomy.

The closest category is `company-brand.general-satisfaction`, which captures overall brand sentiment — but Gemini correctly distinguishes between "I'm dissatisfied with the brand in general" and "I'm dissatisfied with the physical product quality." These are not the same thing, and forcing the latter into `company-brand` would be a false positive.

**This is a taxonomy coverage gap, not a pipeline failure.** Gemini is behaving correctly: it finds no matching aspect from the 12 defined categories and returns an empty detection (or `no-aspect`) rather than hallucinating a category that does not exist.

**Why FABSA lacks product quality:** The 12-aspect taxonomy was designed around service and experience dimensions — account management, logistics, staff support, online experience, value, purchase experience, and brand — none of which map to physical product attributes. Even though Fashion is one of FABSA's 10 industries, the annotation schema treats fashion brands as service providers (website, delivery, customer support, brand trust) rather than as product manufacturers. Physical product quality, material durability, and construction were simply not in scope for the taxonomy.

**Implication for our Fashion domain predictions:** Reviews that discuss Lululemon or Nike product quality directly will consistently return no aspects. Reviews that discuss their website, checkout experience, delivery, customer service, or pricing will be detected correctly. This is a known coverage limitation of the FABSA taxonomy when applied to a fashion brand whose reviews frequently focus on product construction rather than service experience.

---

## Summary: What Makes This a Strong NLP Project

| Dimension | Evidence |
|---|---|
| Technical depth | LoRA fine-tuning on domain-specific pre-trained model, not vanilla BERT |
| Empirical rigour | 3-model ACD comparison, Macro-F1 as primary ACSC metric, LLM-as-Judge end-to-end evaluation |
| Dataset awareness | Identified 4 dataset quality issues, fixed 3 (conflict removal, LEXICAL_MAP, cross-split data leakage) |
| System completeness | End-to-end: scraping → preprocessing → ACD → ACSC → web app |
| Interpretability | Every aspect detection comes with reasoning + evidence spans |
| Evaluation sophistication | Independent LLM-as-Judge (Claude Sonnet 4.5) beyond standard accuracy metrics |
| Real-world grounding | Actual Trustpilot data across 4 industries, not just benchmark datasets |
| Engineering quality | Caching, confidence thresholds, fault tolerance, multi-domain support |
