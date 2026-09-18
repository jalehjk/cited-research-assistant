import os

import streamlit as st

from research_assistant.core import Retriever, answer, ingest_pdf

st.set_page_config(page_title="Research Assistant", page_icon="📄")
st.title("Research Assistant with Source Citations")
st.caption("Ask questions about selectable-text PDFs. Citations name a file and its PDF page.")

uploads = st.file_uploader("Upload research PDFs", type="pdf", accept_multiple_files=True)
mode = st.radio("Answer mode", ["Extractive (free, no API)", "LLM (API key required)"], horizontal=True)
model = os.getenv("LLM_MODEL", "")
key = os.getenv("LLM_API_KEY", "")
if mode.startswith("LLM"):
    model = st.text_input("Model ID", value=model)
    key = st.text_input("API key", value=key, type="password")
    st.caption("PDF passages and your question are sent to the configured model provider.")

if uploads:
    passages, errors = [], []
    for upload in uploads:
        try:
            passages.extend(ingest_pdf(upload.name, upload.getvalue()))
        except ValueError as exc:
            errors.append(f"{upload.name}: {exc}")
    for error in errors:
        st.warning(error)
    if passages:
        st.success(f"Indexed {len(passages)} passages from {len(uploads) - len(errors)} PDF(s).")
        question = st.text_input("Question about these papers")
        if st.button("Find answer", disabled=not question.strip()):
            try:
                result = answer(question, Retriever(passages), mode="llm" if mode.startswith("LLM") else "extractive",
                                api_key=key, model=model, base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
                st.write(result.text)
                if result.hits:
                    st.subheader("Cited passages")
                    for hit in result.hits:
                        with st.expander(f"[{hit.id}] {hit.passage.document}, PDF page {hit.passage.page}"):
                            st.write(hit.passage.text)
            except Exception as exc:
                st.error(f"Answer unavailable: {exc}")
else:
    st.info("Upload at least one PDF to start.")
