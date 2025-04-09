# 🧠 Gemini Research Paper Summarizer & Q&A Tool

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-green?logo=streamlit)](https://summarygenerator-9ohffqczfqdtwxlqxdykas.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

An intelligent web app powered by **Google's Gemini 2.0 Flash** and **LangChain**, designed to **summarize academic papers** and **answer questions** — all through a slick **Streamlit** interface.

Whether you're a student, researcher, or just curious, this tool breaks down complex research into human-friendly summaries and supports real-time Q&A over the content.

---

## 🚀 Features

✅ Upload `.pdf` or `.txt` research papers  
📝 Structured summary with **Title, Context, Problem, Solution, Highlights, Application**  
💬 Ask natural-language questions about the paper content  
🎙️ Choose between **voice** or **typed queries**  
📦 Automatically logs Q&A as `.json` and `.md`  
📥 Download the summary and conversation logs  
🔒 Built with security — your API key stays hidden using Streamlit secrets

---

## 🧰 Built With

- **[Streamlit](https://streamlit.io/)** – Web app framework  
- **[LangChain](https://www.langchain.com/)** – Orchestration for LLM + vector DB  
- **[Google Gemini](https://ai.google.dev/)** – For both summarization and Q&A  
- **[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)** – PDF parsing and extraction  
- **[FAISS](https://github.com/facebookresearch/faiss)** – Vector similarity search  
- **[Google Generative AI Embeddings](https://ai.google.dev/gemini-api/docs/embed-text)** – Text embedding  
- **[SpeechRecognition](https://pypi.org/project/SpeechRecognition/)** – For voice input support

---

## 🖥️ How It Works

1. Upload your `.pdf` or `.txt` research paper.
2. Get an instant, structured summary of the paper.
3. Enter or speak questions — the app understands context.
4. See answers, save logs, and download everything in Markdown or JSON.

---

## 🔐 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/SaptarshiMondal123/Summary_Generator.git
cd summary_generator
```

## 🤝 Contributing
Want to improve this project? Found a bug? PRs are welcome!
```bash
- Fork the repo
- Create your branch: git checkout -b feature/my-feature
- Commit changes: git commit -m 'Added my feature'
- Push to branch: git push origin feature/my-feature
- Create a Pull Request 🚀
```