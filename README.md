# LLM Question Answering & Hallucination Reduction on SQuAD 2.0

## Overview

This project implements a Question Answering (QA) system based on **Meta Llama 3.2 3B Instruct** and evaluates its performance on the **SQuAD 2.0** benchmark.

The main objective is to improve the model's ability to distinguish between:

- Answerable questions
- Unanswerable questions

while reducing hallucinations and ensuring that generated answers are grounded in the provided context.

This project was developed as part of an academic NLP final project.

---

## Problem Statement

Large Language Models often generate plausible answers even when the required information does not appear in the context.

This project focuses on reducing these hallucinations by introducing:

- Structured prompting
- Evidence extraction
- Answer validation
- Context grounding

---

## Methodology

### Model

- Meta Llama 3.2 3B Instruct
- Hugging Face Transformers
- PyTorch

### Dataset

- SQuAD 2.0
- Includes both answerable and unanswerable questions

### Prompt Engineering Strategy

The model is instructed to return answers using a strict structure:

```text
HAS_ANSWER: yes/no
ANSWER: ...
EVIDENCE: ...
```

This structure allows downstream validation and improves reliability.

### Hallucination Prevention

The system applies multiple validation stages:

1. Detect whether the model claims an answer exists.
2. Extract supporting evidence.
3. Verify answer consistency.
4. Reject answers that are not grounded in the provided context.
5. Return NO ANSWER when confidence is insufficient.

---

## Project Structure

```text
llm-question-answering-squad2/
│
├── main.py
├── config.json
├── README.md
│
├── utils/
│   ├── query_model.py
│   └── evaluate_results.py
│
└── data/
    ├── squad2.0-dev-1000.csv
    ├── squad2.0-dev-1000-sample.csv
    └── squad2.0-dev-1000-sample-results.csv
```

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Llama 3.2 3B Instruct
- Pandas
- NLP
- Prompt Engineering

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the QA pipeline:

```bash
python main.py
```

The system will:

1. Load the SQuAD dataset.
2. Query the LLM.
3. Apply answer validation.
4. Generate predictions.
5. Save results for evaluation.

---

## Key Concepts Demonstrated

- Large Language Models (LLMs)
- Prompt Engineering
- Hallucination Reduction
- Retrieval-Free Question Answering
- NLP Evaluation
- Structured Output Generation
- Context Grounding

---

## Contributors

- Amit Reich
- Moshe Tal
