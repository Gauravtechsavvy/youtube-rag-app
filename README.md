# Ask Questions About Any YouTube Video 🎥🤖

A **Streamlit-based Retrieval-Augmented Generation (RAG) application** that allows users to ask natural language questions about any YouTube video. The app uses the video’s subtitles as context and answers questions **strictly from the transcript**, ensuring grounded and reliable responses.

This project demonstrates a complete **LLM-powered RAG pipeline**, from data ingestion to user-facing UI.

---

## 🚀 Features

- Ask questions about **any YouTube video**
- Automatic subtitle download using **yt-dlp**
- Supports multiple languages:
  - English (en)
  - Hindi (hi)
  - Marathi (mr)
  - Kannada (kn)
  - Tamil (ta)
- Transcript cleaning and chunking
- Semantic search using **FAISS vector store**
- Context-aware answers using **Groq LLMs**
- Simple and interactive **Streamlit UI**

---

## 🧠 How It Works (RAG Pipeline)

1. User provides a **YouTube URL** and a **question**
2. Subtitles are downloaded in the selected language
3. Subtitle text is cleaned and preprocessed
4. Text is split into overlapping chunks
5. Chunks are converted into embeddings using HuggingFace models
6. FAISS retrieves the most relevant chunks
7. The LLM answers the question **only using retrieved context**

If the answer is not present in the transcript, the model responds with:

> "I don't know"

---

## 🛠 Tech Stack

- **Python** - 3.12.10
- **Streamlit** – User Interface
- **LangChain** – RAG orchestration
- **FAISS** – Vector similarity search
- **HuggingFace Embeddings** – `all-MiniLM-L6-v2`
- **Groq** – Large Language Models
- **yt-dlp** – YouTube subtitle extraction

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```


---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

---

## 📸 Screenshots (Optional)

_Add screenshots or a short demo GIF here to showcase the UI._

---

## ⚠️ Known Limitations

- Some YouTube videos do not have subtitles available
- Auto-generated subtitles may contain errors
- Answer quality depends on subtitle quality
- Very long videos may increase processing time

---

## 📌 Future Improvements

- Fallback to YouTubeTranscriptApi when subtitles are unavailable
- MultiQueryRetriever for better recall
- Caching embeddings for faster repeated queries
- Support for more languages
- Deploy on Streamlit Cloud

---

## 🙌 Learning Outcomes

This project helped me understand:
- End-to-end RAG system design
- Vector databases and semantic retrieval
- LangChain Runnable architecture
- Integrating LLMs into real applications
- Building production-style Streamlit apps

---

## 📄 License

This project is open-source and available under the **MIT License**.

---

⭐ If you find this project useful, feel free to star the repository!

