# Streamlit Web App Version of Research Paper Summarizer + Q&A Tool

import streamlit as st
import fitz
import re
import os
import json
from datetime import datetime
from textwrap import dedent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize Models
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Prompt Template
def get_prompt():
    return ChatPromptTemplate.from_template(dedent("""\
        You are a research paper summarizer. Given the following content, summarize it in structured format:

        IMPORTANT: Ignore references to other research papers, abstracts of other papers, or unrelated text. Only summarize the core content of THIS paper.

        Title:
        Context:
        Problem:
        Proposed Solution:
        Key Highlights:
        Application:

        Paper:
        \"\"\"{text}\"\"\"        
    """))

# Chain
parser = StrOutputParser()
chain = get_prompt() | llm | parser

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for i, page in enumerate(doc):
        text += page.get_text("text")
    return text

# Title extraction
def extract_title(text):
    lines = text.split("\n")
    for line in lines:
        if re.match(r'^\s*[A-Z][A-Za-z0-9\s:\-(),]+$', line.strip()) and len(line.strip().split()) > 3:
            return line.strip()
    return "Unknown Title"

# Chunking

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Create Q&A chain
def create_qa_chain(text_chunks):
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding_model)
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

# Streamlit App
st.set_page_config(page_title="Research Paper Summarizer & QA", layout="wide")
st.title("📄 Research Paper Summarizer + Q&A with Gemini")

uploaded_file = st.file_uploader("Upload your research paper (.pdf or .txt)", type=["pdf", "txt"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "pdf":
        paper_content = extract_text_from_pdf(uploaded_file)
    elif file_type == "txt":
        paper_content = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type.")
        st.stop()

    title = extract_title(paper_content)
    st.subheader(f"📝 Title: {title}")

    if st.button("🔍 Generate Summary"):
        with st.spinner("Generating summary..."):
            summary = chain.invoke({"text": paper_content})
        st.markdown("### ✨ Summary")
        st.markdown(summary)
        st.download_button("Download Summary", summary, file_name="summary.md")

    if st.checkbox("💬 Ask questions about the paper"):
        st.info("Q&A Mode enabled. Type your questions below.")
        text_chunks = chunk_text(paper_content)
        qa_chain = create_qa_chain(text_chunks)

        query = st.text_input("Ask your question:")
        if query:
            with st.spinner("Thinking..."):
                answer = qa_chain({"query": query})["result"]
            st.markdown("#### 🤖 Answer:")
            st.write(answer)

        if 'qa_log' not in st.session_state:
            st.session_state.qa_log = []

        if query and answer:
            st.session_state.qa_log.append({"question": query, "answer": answer})

        if st.session_state.qa_log:
            with st.expander("📜 Q&A History"):
                for i, qa in enumerate(st.session_state.qa_log, 1):
                    st.markdown(f"**Q{i}:** {qa['question']}")
                    st.markdown(f"**A:** {qa['answer']}\n")

        if st.button("📥 Download Q&A Log"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            qa_json = json.dumps(st.session_state.qa_log, indent=4)
            st.download_button("Download JSON", qa_json, file_name=f"qa_log_{timestamp}.json")

            md_lines = [f"### Q{i+1}. {qa['question']}\n**A:** {qa['answer']}\n" for i, qa in enumerate(st.session_state.qa_log)]
            md_text = f"# Q&A Log\n\n{''.join(md_lines)}"
            st.download_button("Download Markdown", md_text, file_name=f"qa_log_{timestamp}.md")
