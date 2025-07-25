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
    reflect_on_answer,
    finalize_summary,
    route_research,
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
    st.checkbox("DeepThinking", value=True, key="deepthinking_mode")
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

    if st.session_state.deepthinking_mode and st.session_state.search_mode:
        config = Configuration(**st.session_state.config)
        state = SummaryState(
            research_topic=query,
            search_query=query,
            web_research_results=[],
            sources_gathered=[],
            research_loop_count=0,
            running_answer=None,
        )

        while route_research(state, config) == "web_research":
            query = generate_query(state, config)
            state.search_query = query["search_query"]

            web_results = web_research(state, config)
            state.web_research_results.extend(web_results["web_research_results"])
            state.sources_gathered.extend(web_results["sources_gathered"])
            state.research_loop_count = web_results["research_loop_count"]

            summarize_result = summarize_sources(state, config)
            state.running_answer = summarize_result["running_answer"]

            reflect_result = reflect_on_answer(state, config)
            state.search_query = reflect_result["search_query"]

        finalize_summary = finalize_summary(state)
        state.running_answer = finalize_summary["running_answer"]

        st.markdown(state.running_answer)
        st.session_state.chat_history.append(("🤖 DeepThinking", state.running_answer))
        responses.append(("🤖 DeepThinking", state.running_answer))

    st.session_state.chat_history.append(("🧑 You", query))
    for source, resp in responses:
        st.session_state.chat_history.append((source, resp))
