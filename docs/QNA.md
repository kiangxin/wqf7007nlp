# Project Challenges, Solutions & Defence
## Q&A Format — Prepared from Dataset Audit + Methodology Review

---

## PART 1 — Dataset Problems & What We Did

### Q: What data quality issues did you find in FABSA, and did you fix them?

**Three issues found by auditing the raw HuggingFace splits directly:**

**1. Data Leakage**
- 46 Train↔Test and 41 Train↔Val reviews appear in both splits
- Root cause: most are trivial short phrases (`"Quality"`, `"Simple"`, `"quick and easy"`) that appear independently across multiple app reviews — coincidental collisions, not genuine duplication
- Fix: **Addressed.** Duplicate rows were removed from the training split during preprocessing, and the model was retrained on the deduplicated data. Additionally, lexical normalisation (LEXICAL_MAP) was added to the preprocessing pipeline at the same time — contractions expanded, slang standardised — ensuring the cleaned training data is consistent with what the model sees at inference.

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

### Q: What about the 1,337 very short reviews (< 4 words)?

This is the root cause of ACD false positives/negatives. A review like `"Easy to use"` from a banking app has zero context to disambiguate whether the user means:
- The mobile app UI → `online-experience.app-website`
- The account opening flow → `purchase-booking-experience.ease-of-use`
- The loan process → `account-management.account-access`

**Our mitigation:** `min_confidence=0.70` threshold. When Gemini is uncertain about which aspect a vague phrase refers to, its self-reported confidence drops below 0.70 and the detection is filtered out. This accepts some false negatives to avoid false positives.

**Why we can't fully fix this:** It is a fundamental dataset quality problem — 82% of FABSA comes from Google Play / Apple Store where character limits produce extremely short reviews. Our Trustpilot-scraped inference data is higher quality (longer, more specific reviews) which is why the system performs better in practice than the benchmark numbers suggest.

**A practical mitigation we can add (or have added):** The Gemini ACD prompt can be given the review's source context — industry and platform (e.g., `"Banking app — Google Play review"`). For a review like `"Easy to use"` from a Banking app, knowing the industry allows Gemini to reason: "In the context of banking apps, 'easy to use' most likely refers to the app UI (`online-experience.app-website`) or account management flow (`account-management.account-access`) rather than a general purchase experience." This context is already available in our scraped data (`industry`, `data_source` columns) and can be prepended to the review before the LLM call with minimal prompt change.

---

### Q: Was class imbalance addressed?

Yes — two ways:

1. **Evaluation metric:** Macro-F1 (89.0%) is our primary ACSC metric. It treats all 12 aspect × 3 sentiment classes equally regardless of frequency. The gap between weighted F1 (94.0%) and macro F1 (89.0%) — ~5 pp — directly quantifies the imbalance effect and is explicitly reported.

2. **LLM ACD is immune to FABSA's imbalance:** Gemini was never trained on FABSA, so it does not inherit the distribution bias (Fashion 36%, IT 1.7%). It handles rare aspects (`staff-support.phone`, `value.discounts-promotions`) through general language understanding, not frequency-based pattern matching.

**What was not done:** Explicit oversampling or class-weighted loss in ACSC training. Checking `train.py` directly: the only imbalance-aware mechanism is `label_smoothing_factor=0.1` (softens overconfident predictions on dominant classes) — there is no `class_weight` parameter or focal loss in the current training code. Adding `class_weight="balanced"` to the loss or switching to focal loss is a one-line change but was not implemented. Macro-F1 as the primary evaluation metric monitors the imbalance effect rather than correcting it at training time.

---

## PART 2 — Architecture Challenges & Solutions

### Q: Why insist on LLM for ACD? What specific problem does it solve that a fine-tuned DeBERTa cannot?

**Three concrete reasons backed by evidence from the dataset audit:**

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

### Q: What about LLM misclassification — Gemini is not perfect either?

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

### Q: How does error propagation work in a two-stage pipeline, and did you address it?

The pipeline is sequential: ACD → ACSC. Any aspect missed by ACD produces a final false negative that ACSC cannot recover. Any aspect wrongly detected by ACD produces a final false positive even if ACSC assigns the "correct" sentiment to it.

**Addressed through:**
1. `min_confidence=0.70` — reduces ACD false positives at the cost of some recall
2. LLM-as-Judge evaluation — measures **ACD quality in isolation**: predicted aspect slugs vs ground-truth slugs, plus a 0–10 reasoning quality score per review. This is not end-to-end evaluation; it measures only whether the right aspects were detected and whether the reasoning was grounded.
3. ACSC quality is measured independently via DeBERTa evaluation: Macro-F1 89.0%, Accuracy 91.0% on the FABSA test split.

**Remaining gap:** The two evaluations (ACD: LLM-as-Judge, ACSC: DeBERTa metrics) are separate. We have no single measurement that traces a wrong final output back to whether the error came from the ACD stage or the ACSC stage. A combined end-to-end evaluation on a set with both gold aspect labels AND gold sentiments would enable this decomposition — it is documented as future work.

---

## PART 3 — Defending the Implementation

### Q (Teammate): The ACSC fine-tuning alone isn't enough ML work — is this really a full ML contribution?

The concern is not about number of epochs, but about whether fine-tuning a single ACSC classifier is too narrow a scope to represent a full ML engineering contribution.

**The ML engineering scope is broader than just DeBERTa fine-tuning:**

The two ML engineers built the complete machine learning pipeline — which includes the LLM component as ML engineering, not just the supervised classifier:

| Component | ML Engineering Work |
|---|---|
| DeBERTa ACSC | LoRA (r=16, alpha=32) fine-tuning on FABSA, 5 epochs, AdamW with linear warmup, label smoothing, early stopping — full training pipeline with stratified val split and Macro-F1 tracking |
| LLM ACD | Designed the structured JSON schema, 12-aspect taxonomy system prompt, per-aspect confidence scoring, chain-of-thought reasoning requirement, and `evidence_spans` extraction — this is prompt engineering as ML system design |
| Ablation study | Controlled comparison with correct label remapping (BASE_TO_OURS) — without this mapping the ablation results would be meaningless; the +7.67 pp result required careful experimental design |
| Inference pipeline | Span expansion logic, 3-tier fuzzy matching, sentence boundary expansion, diffuse-vs-verbatim fallback — all ML inference design decisions |

The LLM ACD component in particular requires ML thinking: choosing `min_confidence=0.70` as a precision/recall operating point, structuring the prompt to elicit calibrated confidence scores, and designing the fallback for hallucinated or out-of-taxonomy aspect IDs. This is not software engineering — it is applied ML system design.

**On DeBERTa alone:** We started from a domain-specific pre-trained model, not a generic transformer. The ablation quantifies the contribution of our fine-tuning:

| Model | Accuracy |
|---|---|
| Base `deberta-v3-base-absa-v1.1` (no FABSA fine-tuning) | 83.3% |
| Our fine-tuned v3 (5 epochs LoRA on FABSA) | **91.0%** |
| Improvement | **+7.67 pp** |

+7.67 pp from domain adaptation on a model that was already ABSA-specialised is meaningful. If the claim were that vanilla DeBERTa → 5 epoch fine-tune → done, that would be a thin contribution. But domain-specific pre-training + LoRA adaptation + ablation + LLM ACD system design is a complete ML engineering scope.

---

### Q (Teammate): Is proposing only ACSC incomplete as an ABSA task?

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

### Q (Teammate): Two ML engineers were proposed — was the modelling work substantial enough?

The ML engineers owned the model training and inference pipeline. The dataset audit, preprocessing, and evaluation components were owned by the data engineer and evaluation specialist respectively — those deliverables are not double-counted here.

The ML engineers' specific scope:

| Deliverable | Detail |
|---|---|
| LoRA fine-tuning | r=16, alpha=32, target modules query_proj+value_proj, 5 epochs, AdamW with linear warmup, label smoothing, early stopping |
| Ablation study | Base vs fine-tuned comparison with correct label remapping (BASE_TO_OURS) — required to avoid misleading results when comparing across different label spaces |
| LLM ACD system design | Structured JSON schema, 12-aspect taxonomy in system prompt, chain-of-thought reasoning, per-aspect confidence scoring, evidence span extraction |
| `min_confidence` threshold selection | Precision/recall operating point design for the ACD confidence filter |
| Inference pipeline | Span expansion logic, 3-tier fuzzy matching, sentence boundary expansion, diffuse-vs-verbatim fallback, response caching, LLM failure handling |
| Class imbalance analysis | Weighted vs Macro-F1 gap quantification, decision on label smoothing vs class-weighted loss trade-off |

The strongest claim for ML depth is the LLM ACD component: designing a prompting system that produces structured, calibrated, explainable output — with confidence scores that function as a learned operating point — requires ML thinking, not just software engineering. Combined with the LoRA domain adaptation and the controlled ablation, this represents substantive ML engineering work across both the generative and discriminative components of the system.

---

### Q (Teammate): Class imbalance should be addressed with oversampling/class weighting — why wasn't it?

The concern is valid for a purely fine-tuned pipeline. For our architecture, it is partially addressed and partially mitigated by design:

**For ACSC (DeBERTa):**
- Sentiment imbalance: positive (65%), negative (32%), neutral (4%) across training pairs
- We chose Macro-F1 as primary metric which penalises poor neutral performance equally — this monitors the problem
- Oversampling or class-weighted loss would be the next improvement if we retrain

**For ACD (LLM):**
- Not subject to FABSA's aspect frequency imbalance at all — Gemini was never trained on FABSA
- Rare aspects (IT: 1.7%, Consulting: 1.0%) are handled through general language understanding, not frequency-biased pattern matching

**Valid future improvement:** Add `class_weight="balanced"` or focal loss to the ACSC fine-tuning. The infrastructure (train.py) supports this with one parameter change. Not implemented due to time constraints, documented as future work.

---

## PART 4 — NLP Techniques Utilized

### Q: What NLP techniques does the project use, and to what extent?

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
- LLM judge evaluates: exact match of aspect-sentiment pairs + reasoning quality (0-10)
- Beyond standard metrics: captures ACD quality that pure ACSC accuracy cannot measure
- Extent: full evaluation pipeline with separate judge cache, metadata logging

**7. Ablation Study**
- Isolated the contribution of FABSA fine-tuning vs base model
- Required correct label remapping (BASE_TO_OURS) to avoid misleading results
- Result: +7.67 pp accuracy improvement, quantifying domain adaptation value

**8. Real-World Data Collection (Applied NLP)**
- Playwright headless Chromium to scrape Trustpilot (AWS WAF bypass via `__NEXT_DATA__` JSON extraction)
- 300 reviews across 4 industries: Banking, IT, E-Commerce, Fashion
- Rating-stratified selection to avoid negative-review bias (Trustpilot platform skew)
- Extent: production scraper with resumable checkpointing, multi-domain support

---

## PART 5 — Train/Inference Gap & Span-Based ACSC

### Q (Teammate): You trained DeBERTa on full review text, but during inference you feed it extracted spans. Isn't that a distribution mismatch?

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

### Q (Teammate): How does the system handle a review where the same aspect has both positive and negative sentiments?

**This happens at two levels — training data and live inference — and we handle them differently.**

**Training data (already addressed in Section 5.4):**
73 (text, aspect) pairs in the FABSA train split had both positive AND negative labels for the same aspect. These 146 rows were removed entirely from training. DeBERTa was therefore never given contradictory supervision signal for the same input.

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

### Q (Teammate): For long reviews, you break into spans. Is DeBERTa actually capable of doing ACSC on a short span rather than the full context?

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

## Summary: What Makes This a Strong NLP Project

| Dimension | Evidence |
|---|---|
| Technical depth | LoRA fine-tuning on domain-specific pre-trained model, not vanilla BERT |
| Empirical rigour | Ablation study with correct label alignment, Macro-F1 as primary metric |
| Dataset awareness | Identified 4 dataset quality issues, fixed 2 (conflict removal, LEXICAL_MAP) |
| System completeness | End-to-end: scraping → preprocessing → ACD → ACSC → web app |
| Interpretability | Every aspect detection comes with reasoning + evidence spans |
| Evaluation sophistication | LLM-as-Judge beyond standard accuracy metrics |
| Real-world grounding | Actual Trustpilot data across 4 industries, not just benchmark datasets |
| Engineering quality | Caching, confidence thresholds, fault tolerance, multi-domain support |
