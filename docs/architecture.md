# System Architecture

## High-Level Pipeline

```mermaid
flowchart TD
    subgraph A["1 — Data Preprocessing"]
        A1[("FABSA Dataset\nHuggingFace\n~10,574 reviews")]
        A2["Clean\ndedup · contraction expansion\nemoji demojize · URL masking\nconflict removal · leakage fix"]
        A3["Pairwise Expansion\none row per\n(review, aspect) pair"]
        A4[("Train / Val / Test CSVs\ndev/data/")]
        A1 --> A2 --> A3 --> A4
    end

    subgraph B["2 — Model Fine-Tuning  (ACSC)"]
        B1["Base Model\nyangheng/deberta-v3-base-absa-v1.1\npre-trained on SemEval ABSA"]
        B2["LoRA Fine-Tuning\nr=16, alpha=32\nquery_proj + value_proj\n5 epochs · early stopping"]
        B3["Evaluation\nAccuracy 94.1%\nMacro F1  89.0%"]
        B4[("Saved LoRA Adapter\ndev/model/deberta_absa_finetuned/")]
        B1 --> B2
        A4 -->|train / val split| B2zs
        A4 -->|test split| B3
        B2 --> B4
        B2 --> B3
    end

    subgraph C["3 — Inference Pipeline"]
        C1["Customer Review\n(raw text)"]
        C2["Stage 1 — ACD\nGemini 3.5 Flash via OpenRouter\nStructured JSON · chain-of-thought\naspect list + evidence spans + confidence"]
        C3{"min_confidence\n>= 0.70?"}
        C4["Stage 2 — ACSC\nDeBERTa-v3 LoRA\nsentence-pair classification\npositive / negative / neutral"]
        C5["Structured Output\naspect · sentiment · confidence\nevidence span · reasoning"]
        C1 --> C2 --> C3
        C3 -->|pass| C4
        C3 -->|filter| C5
        C4 --> C5
        B4 -->|load adapter| C4
    end

    subgraph D["4 — Web Demo  (app/)"]
        D1["Browse Tab\nPre-analyzed Trustpilot reviews\nBanking · IT · E-Commerce · Fashion"]
        D2["Live Analysis Tab\nReal-time inference\noptional company / industry context"]
        C5 --> D1
        C5 --> D2
    end
```

---

## Component Summary

| Stage | Component | Technology |
|---|---|---|
| Preprocessing | Clean + pairwise expand FABSA | Python · pandas · `contractions` · `emoji` |
| Fine-tuning (ACSC) | LoRA adapter on DeBERTa-v3 | HuggingFace PEFT · PyTorch |
| ACD (inference) | Aspect category detection | Gemini 3.5 Flash via OpenRouter |
| ACSC (inference) | Sentiment per detected aspect | Fine-tuned DeBERTa-v3 |
| Web demo | Browse + Live Analysis | FastAPI · vanilla HTML/JS |

---

## Why Two-Stage?

The ACD and ACSC tasks require fundamentally different strengths:

- **ACD needs world knowledge** — deciding whether *"I cannot log in"* maps to `account-management.account-access` requires understanding what a banking service is, not pattern matching. An LLM handles this through reasoning; a fine-tuned classifier memorises annotation noise.
- **ACSC needs precision** — given a specific aspect, classifying sentiment (positive / negative / neutral) is a well-scoped 3-class problem that a fine-tuned discriminative model handles with high accuracy (94.1%).

Splitting the tasks lets each component do what it is best at.
