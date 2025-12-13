# ======================================================================
#  IMPORTS
# ======================================================================
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from pathlib import Path
import regex as re
import os
from dotenv import load_dotenv
import yt_dlp
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

# ======================================================================
#  USER INTERFACE SECTION
# ======================================================================
st.title('Ask question about any Youtube video')

# ----------------------------------------------------------------------
#  USER QUESTION INPUT
# ----------------------------------------------------------------------
user_question = st.text_input('Enter your question regarding youtube video')
if not user_question:
    st.warning('Please enter your question')
    st.stop()
st.success('Thanks, go ahead!')

# ----------------------------------------------------------------------
#  YOUTUBE URL INPUT
# ----------------------------------------------------------------------
url = st.text_input("Paste YouTube URL:")
if not url:
    st.warning('Please enter your YouTube link')
    st.stop()
st.success('Thanks, go ahead')

# ======================================================================
#  LANGUAGE SELECTION
# ======================================================================
langs = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Kannada": "kn",
    "Tamil": "ta"
}

user_choice = st.selectbox("Choose language", list(langs.keys()))
if not user_choice:
    st.warning('Please select a language')
    st.stop()
st.success('Press Submit')

selected_lang = langs[user_choice]

# ======================================================================
#  DOWNLOADING SUBTITLES (.vtt)
# ======================================================================
ydl_opts = {
    "quiet": True,
    "no_warnings": True,
    "writesubtitles": True,
    "writeautomaticsub": True,
    "skip_download": True,
    "subtitleslangs": [selected_lang],
    "outtmpl": "subtitle.%(ext)s",
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

# ======================================================================
#  READING SUBTITLE FILE
# ======================================================================
vtt_file = f"subtitle.{selected_lang}.vtt"
text = Path(vtt_file).read_text(encoding="utf-8", errors="ignore")

# ======================================================================
#  CLEANING THE TRANSCRIPT
# ======================================================================
cleaned = re.sub(r"\d+:\d+:\d+\.\d+ --> .*", "", text)
cleaned = cleaned.replace("\n\n", "\n")

# ======================================================================
#  SPLITTING INTO CHUNKS
# ======================================================================
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
chunks = splitter.create_documents([cleaned])

# ======================================================================
#  EMBEDDING + VECTOR STORE + RETRIEVER
# ======================================================================
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(chunks, embedding)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

# ======================================================================
#  PROMPT TEMPLATE
# ======================================================================
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer only from the provided transcript context.
If the context is insufficient just say: "I don't know".

Transcript Context:
{context}

Question:
{question}
""",
    input_variables=['context', 'question']
)

# ======================================================================
#  FORMATTING RETRIEVED DOCUMENTS
# ======================================================================
def format_docs(retriever_docs):
    return "\n\n".join(doc.page_content for doc in retriever_docs)

runnable_parallel = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

# ======================================================================
#  LOAD GROQ MODEL
# ======================================================================
model = ChatGroq(
    groq_api_key=os.getenv('groq_api'),
    model='openai/gpt-oss-20b'
)

# ======================================================================
#  FINAL CHAIN
# ======================================================================
chain = runnable_parallel | prompt | model | parser

# ======================================================================
#  ON BUTTON CLICK → GET ANSWER
# ======================================================================
if st.button('Press Submit'):
    result = chain.invoke(user_question)
    st.write(result)
