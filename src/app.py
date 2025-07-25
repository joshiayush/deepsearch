import os
from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

from state import SummaryState
from core import (
    generate_query,
    web_research,
    summarize_sources,
    reflect_on_summary,
    finalize_summary,
)
from configuration import Configuration

load_dotenv()

st.set_page_config(page_title="Deepsearch", layout="wide")
st.title("Deepsearch: AI-Powered Research Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

with st.sidebar:
    st.subheader("🔐 API & Document Setup")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )
    if st.button("📥 Process PDFs") and uploaded_files and openai_api_key:
        raw_text = ""
        for pdf in uploaded_files:
            reader = PdfReader(pdf)
            for page in reader.pages:
                raw_text += page.extract_text() or ""

        splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200
        )
        chunks = splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        st.session_state.vectorstore = vectorstore
        st.success("✅ PDFs processed successfully.")


with st.sidebar.expander("⚙️ Mode Selection", expanded=False):
    deepthinking = st.checkbox("DeepThinking", value=True, key="deepthinking_mode")
    if uploaded_files:
        st.checkbox("Search Mode", value=False, disabled=True, key="search_mode")
        st.info("Search mode is disabled when PDFs are uploaded.")
    else:
        st.checkbox("Search Mode", value=True, key="search_mode")


if "config" not in st.session_state:
    st.session_state.config = Configuration().dict()

with st.sidebar.expander("⚙️ Configuration", expanded=False):
    st.markdown("Customize the research assistant's behavior.")

    st.session_state.config["max_web_research_loops"] = st.number_input(
        label="🔁 Research Depth",
        min_value=1,
        max_value=10,
        value=st.session_state.config["max_web_research_loops"],
        help="Number of research iterations to perform.",
    )
    st.session_state.config["llm_model"] = st.selectbox(
        label="🧠 LLM Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"].index(
            st.session_state.config["llm_model"]
        ),
        help="Choose the language model to use.",
    )
    st.session_state.config["search_api"] = st.selectbox(
        label="🔍 Search API",
        options=["perplexity", "tavily", "duckduckgo", "searxng"],
        index=["perplexity", "tavily", "duckduckgo", "searxng"].index(
            st.session_state.config["search_api"]
        ),
        help="Choose the web search engine.",
    )
    st.session_state.config["fetch_full_page"] = st.checkbox(
        label="📰 Fetch Full Page Content",
        value=st.session_state.config["fetch_full_page"],
        help="Whether to retrieve full web page content.",
    )
    st.session_state.config["strip_thinking_tokens"] = st.checkbox(
        label="✂️ Strip <think> Tokens",
        value=st.session_state.config["strip_thinking_tokens"],
        help="Removes <think> tags from LLM output.",
    )


query = st.text_input("💬 Ask a question...")
if query and openai_api_key:
    responses = []

    if deepthinking:
        if not st.session_state.vectorstore:
            st.error("❌ PDFs are not processed yet.")
        else:
            docs = st.session_state.vectorstore.similarity_search(query)
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=query)
            responses.append(("📄 DeepThinking", answer))

    if st.session_state.get("search"):
        llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        search_response = llm.predict(query)
        responses.append(("🌐 Search", search_response))

    st.session_state.chat_history.append(("🧑 You", query))
    for source, resp in responses:
        st.session_state.chat_history.append((source, resp))
