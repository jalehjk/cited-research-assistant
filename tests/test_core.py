import json

import pymupdf
import pytest

from research_assistant.core import ABSTAIN, Passage, Retriever, answer, ingest_pdf


def pdf(*pages):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_pages_are_preserved_and_scans_rejected():
    passages = ingest_pdf("study.pdf", pdf("Cohort includes 120 adults.", "Treatment reduced pain by 35 percent."))
    assert [(p.page, p.document) for p in passages] == [(1, "study.pdf"), (2, "study.pdf")]
    with pytest.raises(ValueError, match="No selectable text"):
        ingest_pdf("blank.pdf", pdf(""))


def test_answer_cites_relevant_page_and_abstains_on_unrelated_question():
    passages = ingest_pdf("trial.pdf", pdf("Enrollment criteria include adults over 18.", "Treatment reduced pain by 35 percent."))
    retriever = Retriever(passages)
    result = answer("How much did treatment reduce pain?", retriever)
    assert "35 percent" in result.text
    assert result.hits[0].passage.page == 2
    assert f"[{result.hits[0].id}]" in result.text
    unknown = answer("What was the satellite orbital altitude?", retriever)
    assert unknown.abstained and unknown.text == ABSTAIN


def test_llm_rejects_invented_citation(monkeypatch):
    passages = [Passage("paper.pdf", 1, "Treatment reduced pain by 35 percent.", 0)]

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def fake_urlopen(request, timeout):
        assert b"Treatment reduced pain" in request.data
        response = FakeResponse()
        response.read = lambda: json.dumps({"choices": [{"message": {"content": json.dumps({"sentences": [
            {"text": "Treatment reduced pain by 35 percent.", "source_ids": ["S999"]}
        ]})}}]}).encode()
        return response

    monkeypatch.setattr("research_assistant.core.urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="answer withheld"):
        answer("How much did treatment reduce pain?", Retriever(passages), mode="llm", api_key="test", model="test")


def test_llm_accepts_only_returned_source_id(monkeypatch):
    passages = [Passage("paper.pdf", 4, "Treatment reduced pain by 35 percent.", 0)]

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({"sentences": [
                {"text": "Treatment reduced pain by 35 percent.", "source_ids": ["S1"]}
            ]})}}]}).encode()

    monkeypatch.setattr("research_assistant.core.urlopen", lambda *args, **kwargs: FakeResponse())
    result = answer("How much did treatment reduce pain?", Retriever(passages), mode="llm", api_key="test", model="test")
    assert not result.abstained and result.hits[0].passage.page == 4


def test_rejects_non_pdf():
    with pytest.raises(ValueError, match="Expected a PDF"):
        ingest_pdf("false.pdf", b"not a PDF")
