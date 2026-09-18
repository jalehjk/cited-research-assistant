"""Tiny synthetic smoke benchmark; not a research-paper performance estimate."""
import argparse
import json
import os
from pathlib import Path

import pymupdf

from research_assistant.core import Retriever, answer, ingest_pdf


def make_pdf(*page_texts):
    doc = pymupdf.open()
    for text in page_texts:
        doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["extractive", "llm"], default="extractive")
    args = parser.parse_args()
    docs = {
        "trial.pdf": make_pdf("The study enrolled 120 adults from three hospitals.",
                              "Treatment reduced pain by 35 percent after six weeks."),
        "methods.pdf": make_pdf("The investigators measured HDL cholesterol at baseline."),
    }
    passages = [p for name, data in docs.items() for p in ingest_pdf(name, data)]
    retriever = Retriever(passages)
    cases = json.loads((Path(__file__).parent / "cases.json").read_text())
    results = []
    for case in cases:
        result = answer(case["question"], retriever, mode=args.mode,
                        api_key=os.getenv("LLM_API_KEY"), model=os.getenv("LLM_MODEL"),
                        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
        correct = (result.abstained if case.get("abstain") else
                   not result.abstained and case["expected_text"].lower() in result.text.lower())
        citation = (result.abstained if case.get("abstain") else
                    any(h.passage.document == case["document"] and h.passage.page == case["page"]
                        and f"[{h.id}]" in result.text for h in result.hits))
        results.append({"question": case["question"], "answer": result.text,
                        "answer_check": bool(correct), "citation_check": bool(citation)})
    print(json.dumps({"mode": args.mode, "count": len(cases),
                      "answer_checks_passed": sum(r["answer_check"] for r in results),
                      "citation_checks_passed": sum(r["citation_check"] for r in results),
                      "cases": results}, indent=2))


if __name__ == "__main__":
    main()
