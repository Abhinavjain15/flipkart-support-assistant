# Flipkart Order Intelligence & Support Assistant

A single connected system: a return-risk model (Part 1), a product-image categoriser
via transfer learning (Part 2), and a LangGraph support agent (Part 3) that loads both
as real, callable tools on top of a retrieval-augmented policy knowledge base.

All AI-API calls default to a deterministic, offline `MOCK_LLM` mode -- zero API keys,
zero network calls required to run or grade this repo.

## Repository Structure
```
├── part1_return_risk/       # return-risk scoring pipeline
├── part2_image_classifier/  # Fashion-MNIST transfer-learning categoriser
├── part3_agent/              # LangGraph support agent (RAG + 2 tools)
├── models/                   # saved artifacts consumed across Parts
├── data/sample_images/       # real exported .png test images
└── transcripts/              # 8 required test-conversation transcripts
```

## Setup
```bash
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Part 1 -- Return-Risk Scoring Pipeline

**Regenerate the dataset and model from scratch:**
```bash
cd part1_return_risk
python3 generate_orders.py       # -> orders_dataset.csv (6000 rows, seed=42)
python3 verify_data.py           # data quality + MAR justification
python3 train_baseline.py        # DummyClassifier: acc=0.7725, F1(class1)=0.0
python3 train_logreg.py          # LogReg: default ROC-AUC=0.6253, threshold sweep -> t*=0.44
python3 train_rf.py              # RF GridSearchCV: best CV ROC-AUC=0.6178, test=0.6143
python3 explain_model.py         # impurity vs permutation importance
python3 subgroup_analysis.py     # recall/precision by category & payment method
python3 save_artifact.py         # -> models/return_risk_model.pkl, t*_rf=0.46
```

**Key results:**
- Overall return rate: 22.75%. `rating_given` missing on 13.05% of rows -- **MAR**,
  conditional on `payment_method` (COD missing rate 22.83% vs non-COD 6.06%, a 16.77pp gap).
- Baseline `DummyClassifier`: accuracy 0.7725, F1(class 1) = 0.0 -- the "high accuracy,
  zero recall" trap.
- Tuned Random Forest (`max_depth=6, n_estimators=100`): best CV ROC-AUC 0.6178,
  test ROC-AUC 0.6143 (gap 0.0036, no overfitting).
- Top-5 features: `payment_method_COD`, `price_inr`, `customer_tenure_days`,
  `delivery_distance_km`, `discount_pct`. Permutation importance shows
  `customer_tenure_days` drops to near-zero (rank #18/18) -- impurity-based importance
  overrated it because RF split-selection favors high-cardinality continuous columns
  regardless of true generalizing signal.
- Weakest subgroup: `Prepaid_Card` payment method, recall = 0.0000 (vs overall 0.5092).
  Proposed fix: a payment-method-specific decision threshold, since the global 0.5 cut
  point is calibrated around COD's much higher base return rate.
- **`t*_rf = 0.46`** (F1-maximizing threshold on the saved RF's own `predict_proba`,
  recomputed independently of the Logistic Regression's t*=0.44).

---

## Part 2 -- Product Image Categoriser (Transfer Learning)

**Run training/evaluation from scratch:**
```bash
cd part2_image_classifier
python3 data_loader.py             # Fashion-MNIST: train=55000, val=5000, test=10000
python3 extract_features.py        # cache frozen ResNet-18 features (single pass)
python3 train_feature_extraction.py # train head only: val_acc=0.9102 (>=80%, no fine-tune needed)
python3 evaluate.py                # test_acc=0.9013, confusion matrix, per-class P/R
python3 save_artifact.py           # -> models/product_classifier.pt
python3 export_sample_images.py    # -> data/sample_images/*.png (7 real test images)
```

**Key results:**
- Backbone: pretrained ResNet-18, frozen, features cached once (512-d), head trained
  15 epochs, Adam, lr=1e-3, batch_size=128.
- Feature extraction alone reached **91.02% val accuracy** -- fine-tuning was **not required**.
- Test accuracy: **90.13%**.
- Top confusion pairs (from the real confusion matrix):
  - **T-shirt/top ↔ Shirt** (217 misclassifications): near-identical outline at 28x28
    resolution; the distinguishing collar/placket detail collapses at this scale.
  - **Coat ↔ Shirt** (144 misclassifications): both present as a rectangular torso block
    with sleeves; length/thickness cues that normally separate them are lost to
    low-resolution downsampling.

---

## Part 3 -- Flipkart Support Agent

**Run the agent (default MOCK_LLM mode, zero API keys, zero network calls):**
```bash
cd part3_agent
python3 embed_index.py             # build/verify the Faiss index
python3 graph.py                   # run the demo multi-turn + fresh-conversation example
python3 retrieval_eval.py          # Precision@3 / Recall@3 over 6 queries
python3 generate_transcripts.py    # regenerate all 8 saved transcripts
```

**Architecture:**
- 14 policy documents, sentence-wise chunked -> 42 chunks, embedded with
  `all-MiniLM-L6-v2` (local, free), indexed with Faiss (`IndexFlatIP` on
  L2-normalized vectors = cosine similarity).
- `check_return_risk(order_features)` loads `models/return_risk_model.pkl` directly
  and buckets risk relative to `t*_rf=0.46`: **Low if prob < 0.46, High if prob >= 0.61,
  else Medium** -- anchored to the saved RF's own threshold, not fixed cut points.
- `classify_product_image(image_path)` loads `models/product_classifier.pt` and runs
  real inference against the `.png` files in `data/sample_images/`.
- LangGraph: 5 nodes (`input_guard`, `intent_classification`, `rag_retrieval`,
  `tool_calling`, `response_generation`), 2 conditional edges (post-guard routing,
  post-intent routing), state persisted via `MemorySaver` checkpointer keyed by `thread_id`.
- Guardrails: input-side prompt-injection pattern filter; output-side groundedness
  check refusing policy answers when top retrieval similarity < 0.35.

**Retrieval evaluation (document-level, 6 queries):**

| Query | P@3 | R@3 |
|---|---|---|
| Shirt return window | 1/3 = 0.3333 | 1/1 = 1.0000 |
| COD refund timeline | 1/3 = 0.3333 | 1/1 = 1.0000 |
| Same-day/next-day delivery | 0/3 = 0.0000 | 0/1 = 0.0000 |
| Broken laptop | 1/3 = 0.3333 | 1/2 = 0.5000 |
| Reverse pickup | 1/2 = 0.5000 | 1/1 = 1.0000 |
| Cancel after shipped | 1/1 = 1.0000 | 1/1 = 1.0000 |

**Average Precision@3 = 0.4167, Average Recall@3 = 0.7500**

### Test Transcripts (all run in MOCK_LLM mode)
1. [Policy Q1: Apparel return window](transcripts/01_policy_q1.md)
2. [Policy Q2: COD refund timeline](transcripts/02_policy_q2.md)
3. [Return-risk tool call](transcripts/03_return_risk.md)
4. [Product image classification](transcripts/04_image_classify.md)
5. [Multi-turn: state carried across turns](transcripts/05_multiturn_state.md)
6. [Fresh conversation: state correctly absent](transcripts/06_fresh_conversation.md)
7. [Guardrail: prompt-injection blocked](transcripts/07_prompt_injection.md)
8. [Guardrail: ungrounded question refused](transcripts/08_ungrounded_refusal.md) --
   similarity 0.3234 vs threshold 0.35, correctly refused rather than fabricated.

### Optional Live-LLM Extension
Not implemented -- out of scope for grading. `USE_LIVE_LLM` flag is unset by default;
every acceptance criterion is satisfied via `MOCK_LLM` alone.

---

## Git Workflow
Feature branches used per Part, each committed to multiple times and merged into `main`:
- `feature/part1-return-risk`
- `feature/part2-image-classifier`
- `feature/part3-agent`

View history: `git log --graph --all`
