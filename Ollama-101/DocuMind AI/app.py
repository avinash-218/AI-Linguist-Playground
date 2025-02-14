import streamlit as st
from utils import add_doc_chunks_to_vectorDB, chunk_documents, find_related_documents, generate_answer, load_pdf_documents, save_uploaded_file

st.title("DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File upload section
upload_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False)

if upload_pdf:
    saved_path = save_uploaded_file(upload_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    add_doc_chunks_to_vectorDB(processed_chunks)

    st.success("Document Processed successfully! Ask your questions below")

    user_input = st.chat_input("Enter your question about the document...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Analyzing Document..."):
            relevant_docs = find_related_documents(user_input)
            answer = generate_answer(user_input, relevant_docs)

        with st.chat_message("assistant"):
            st.write(answer)
