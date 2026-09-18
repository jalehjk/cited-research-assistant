# Research Assistant with Source Citations

A small, understandable retrieval-based question answering project for selectable-text research PDFs. Upload papers, ask a question, inspect the cited passages and PDF page numbers. Local extractive mode requires no model account. An optional LLM integration is implemented and mock-tested, but has not been verified against a live provider.

## Architecture

`PDF bytes -> PyMuPDF page text -> ~140-word page-bounded passages -> TF-IDF (word unigrams + bigrams) -> cosine top 4 -> extractive answer OR LLM JSON -> citation ID validation -> Streamlit display`

**Why these choices:** TF-IDF is fast and free for a small uploaded collection, and its scores are easy to inspect. The LLM uses a configurable OpenAI-compatible chat completions endpoint via the Python standard library. No vector database or framework is needed for this prototype. Page numbers are PDF page indices, which may differ from printed page labels.

## Structure

```text
research_assistant/
├── app.py                      # Streamlit UI
├── src/research_assistant/
│   ├── __init__.py
│   └── core.py                 # ingest, retrieve, answer, validate
├── eval/
│   ├── cases.json              # four synthetic questions
│   └── run_eval.py             # reproducible smoke benchmark
├── tests/test_core.py          # behavior tests with synthetic PDFs/mock LLM
├── sample_data/sample_study.pdf # two-page hands-on example
├── notebooks/Research_Assistant_Colab.ipynb # run core workflow in Colab
├── pyproject.toml
└── README.md
```

## Install and run

Python 3.10+ recommended. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Upload `sample_data/sample_study.pdf` and ask “How much did treatment reduce pain?” Check the answer and PDF page 2. Then ask “What was the satellite orbital altitude?” and check that it abstains. For your own PDF, inspect the cited passage in the original file.

For optional LLM mode, set `LLM_API_KEY`, `LLM_MODEL` and optionally `LLM_BASE_URL` (defaults to `https://api.openai.com/v1`) in your shell, or enter key and model in the interface. The endpoint must support `/chat/completions`; API usage may incur provider charges. Do not commit keys. The app sends the question and selected PDF passages to the configured provider. Extractive mode sends nothing to a model provider.

### Google Colab

Upload `notebooks/Research_Assistant_Colab.ipynb` through Colab's **File → Upload notebook**. Download this repository as a ZIP from GitHub and supply that ZIP when the notebook's first cell requests it. The notebook runs tests, the evaluation, a sample PDF, and your own PDF uploads. It uses notebook cells as the interface rather than a hosted Streamlit web page.

## Test and evaluate

```bash
python -m pytest -q
python eval/run_eval.py
# With key/model set: python eval/run_eval.py --mode llm
```

The evaluation creates two synthetic PDFs in memory and checks expected answer text, expected file/page citation, and one unsupported question. It is a **four-case smoke benchmark**, not an estimate of accuracy on real research papers. Run the same script with `--mode llm` to measure the LLM configuration separately. The test suite mocks model output to check citation ID handling without paying for API calls.

### Results observed so far

- Local run: 5 behavior tests passed; extractive mode passed 3/4 synthetic answer checks and 3/4 citation checks. The missed case called HDL cholesterol a “biomarker” without using the term HDL in the question.
- Colab trial on *Attention Is All You Need*: the English-to-German question returned 28.4 BLEU with a page citation; an unrelated satellite question abstained.
- On the same paper, the English-to-French question returned 41.0 BLEU with a citation to the results paragraph on PDF page 8. The abstract and Table 2 say 41.8. This is a source conflict that the assistant did not flag; it is **not** counted as an unambiguous accuracy success.
- No live LLM API run has been performed, so there is no LLM answer quality result.

## Safeguards and limits

- Rejects non-PDF bytes, files over 20 MB, PDFs over 250 pages, password-protected PDFs and PDFs without selectable text. No OCR.
- Abstains when no passage passes a TF-IDF threshold plus a distinctive-term overlap check, or when extractive mode cannot find a matching sentence.
- LLM instructions restrict answers to passages and request sentence-level citation IDs. The app rejects missing/unknown IDs and responses with no lexical overlap. These checks **do not prove that a cited passage entails the claim**. Manually inspect citations before relying on an answer.
- Lexical retrieval can miss paraphrases, acronyms, figures, tables and equations. A keyword match can also produce a false answer on an unrelated topic. PDF extraction and paragraph boundaries are imperfect; long paragraphs may split mid-sentence.
- Uploaded documents are treated as untrusted content, but prompt injection remains a risk in LLM mode. Use non-sensitive test papers and review answers.

## Interview walkthrough

Explain why chunks stay within a page; show the query, TF-IDF rankings, and citation IDs; demonstrate a supported question and an abstention; then describe failure modes and how you would expand the benchmark with annotated real papers and human citation entailment review. Only report results from runs you actually completed.

