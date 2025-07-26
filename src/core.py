import json
import uuid
import base64
from io import BytesIO
from typing import Dict, Literal, List, Tuple

from unstructured.documents.elements import Element, CompositeElement
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

from state import SummaryState
from utils import (
    deduplicate_and_format_sources,
    tavily_search,
    format_sources,
    perplexity_search,
    duckduckgo_search,
    searxng_search,
    strip_thinking_tokens,
    get_config_value,
)
from prompts import (
    text_tables_summarizer_instructions,
    query_writer_instructions,
    summarizer_instructions,
    reflection_instructions,
    get_current_date,
)
from configuration import Configuration
from utils import strip_thinking_tokens


def create_multi_vector_retriever(
    text_chunks: List[Element],
    text_summaries: List[str],
    table_chunks: List[Element],
    table_summaries: List[str],
    images_b64_chunks: List[Element],
    images_b64_summaries: List[str],
) -> MultiVectorRetriever:
    """Creates a MultiVectorRetriever for the provided text, table, and image summaries.

    Args:
        text_chunks: List of text elements.
        text_summaries: List of summaries for the text elements.
        table_chunks: List of table elements.
        table_summaries: List of summaries for the table elements.
        images_b64_chunks: List of base64 encoded image elements.
        images_b64_summaries: List of summaries for the image elements.

    Returns:
        MultiVectorRetriever instance with the provided summaries.
    """
    id_key = "doc_id"
    vectorstore = Chroma(
        collection_name="deepsearch_multi_vector_store",
        embedding_function=OpenAIEmbeddings(),
    )
    store = InMemoryStore()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    if text_chunks:
        doc_ids = [str(uuid.uuid4()) for _ in text_chunks]
        summary_texts = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]})
            for i, summary in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, text_chunks)))

    if table_chunks:
        table_ids = [str(uuid.uuid4()) for _ in table_chunks]
        summary_tables = [
            Document(page_content=summary, metadata={id_key: table_ids[i]})
            for i, summary in enumerate(table_summaries)
        ]
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, table_chunks)))

    if images_b64_chunks:
        img_ids = [str(uuid.uuid4()) for _ in images_b64_chunks]
        summary_img = [
            Document(page_content=summary, metadata={id_key: img_ids[i]})
            for i, summary in enumerate(images_b64_summaries)
        ]
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, images_b64_chunks)))
    return retriever


def text_tables_and_images_b64_summarizer(
    text_chunks: List[Element],
    table_chunks: List[Element],
    image_b64_chunks: List[Element],
) -> Tuple[List[str], List[str], List[str]]:
    """Summarizes text, tables, and images from the provided chunks.

    Args:
        text_chunks: List of text elements.
        table_chunks: List of table elements.
        image_b64_chunks: List of base64 encoded image elements.

    Returns:
        Tuple containing lists of summarized text, tables, and images.
    """
    prompt = ChatPromptTemplate.from_template(
        text_tables_summarizer_instructions,
    )

    llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    summarizer_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

    text_summaries = summarizer_chain.batch(text_chunks, {"max_concurrency": 3})

    tables_html = [table.metadata.text_as_html for table in table_chunks]
    table_summaries = summarizer_chain.batch(tables_html, {"max_concurrency": 3})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": (
                            "Describe the image in detail. Be specific about graphs."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ]
    )
    images_b64_summarizer_chain = (
        prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    )
    images_b64_summaries = images_b64_summarizer_chain.batch(image_b64_chunks)

    return text_summaries, table_summaries, images_b64_summaries


def generate_query(state: SummaryState, config: Configuration) -> Dict[str, str]:
    """Generates a search query based on the research topic.

    Uses an LLM to create an optimized search query for web research based on the user's
    research topic.
    """
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date, research_topic=state.research_topic
    )

    llm_json_mode = ChatOpenAI(
        model=config.llm_model,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    result = llm_json_mode.invoke(
        [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=f"Generate a query for web search:"),
        ]
    )

    content = result.content

    try:
        query = json.loads(content)
        search_query = query["query"]
    except (json.JSONDecodeError, KeyError):
        # If parsing fails or the key is not found, use a fallback query
        if config.strip_thinking_tokens:
            content = strip_thinking_tokens(content)
        search_query = content
    return {"search_query": search_query}


def web_research(state: SummaryState, config: Configuration) -> Dict[str, str]:
    """Performs web research based on the research topic.

    Executes a web search using the configured search API (tavily, perplexity,
    duckduckgo, or searxng) and formats the results for further processing.

    Args:
        research_topic: The topic to research.
        config: Configuration, including search API settings.

    Returns:
        Dictionary including sources_gathered, research_loop_count, and
        web_research_results
    """
    search_api = get_config_value(config.search_api)

    if search_api == "tavily":
        search_results = tavily_search(
            state.search_query,
            fetch_full_page=config.fetch_full_page,
            max_results=1,
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=1000,
            fetch_full_page=config.fetch_full_page,
        )
    elif search_api == "perplexity":
        search_results = perplexity_search(
            state.search_query, state.research_loop_count
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=1000,
            fetch_full_page=config.fetch_full_page,
        )
    elif search_api == "duckduckgo":
        search_results = duckduckgo_search(
            state.search_query,
            max_results=3,
            fetch_full_page=config.fetch_full_page,
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=1000,
            fetch_full_page=config.fetch_full_page,
        )
    elif search_api == "searxng":
        search_results = searxng_search(
            state.search_query,
            max_results=3,
            fetch_full_page=config.fetch_full_page,
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=1000,
            fetch_full_page=config.fetch_full_page,
        )
    else:
        raise ValueError(f"Unsupported search API: {config.search_api}")

    return {
        "sources_gathered": [format_sources(search_results)],
        "research_loop_count": state.research_loop_count + 1,
        "web_research_results": [search_str],
    }


def summarize_sources(state: SummaryState, config: Configuration) -> Dict[str, str]:
    """Summarizes the web research results.

    Uses an LLM to create or update a running summary based on the newest web research
    results, integrating them with any existing summary.

    Args:
        state: Current graph state containing research topic, running summary, and web
             research results.
        config: Configuration for the runnable, including LLM provider settings.

    Returns:
        Dictionary with state update, including running_answer key containing the
        updated summary.
    """
    existing_answer = state.running_answer
    most_recent_web_research = state.web_research_results[-1]

    if existing_answer:
        human_message_content = (
            f"<Existing Answer>\n{existing_answer}\n</Existing Answer>\n\n"
            f"<Recent Web Search Results>\n{most_recent_web_research}\n</Recent Web Search Results>\n\n"
            f"Your task is to critically enhance and update the Existing Answer using"
            f" only the information from the Recent Web Search Results. Ensure"
            f" accuracy, correct outdated points, and expand on missing aspects based"
            f" on the new context.\n\n"
            f"<Research Topic>\n{state.research_topic}\n</Research Topic>\n\n"
        )
    else:
        human_message_content = (
            f"<Recent Web Search Results>\n{most_recent_web_research}\n</Recent Web Search Results>\n\n"
            f"Your task is to generate a detailed, well-researched response using only"
            f" the information from the Recent Web Search Results. Ensure your answer"
            f" covers key insights, current developments, and notable perspectives.\n\n"
            f"<Research Topic>\n{state.research_topic}\n</Research Topic>\n\n"
        )

    llm = ChatOpenAI(
        model=config.llm_model,
        temperature=0.0,
    )

    result = llm.invoke(
        [
            SystemMessage(content=summarizer_instructions),
            HumanMessage(content=human_message_content),
        ]
    )
    running_answer = result.content
    if config.strip_thinking_tokens:
        running_answer = strip_thinking_tokens(running_answer)

    return {"running_answer": running_answer}


def reflect_on_answer(state: SummaryState, config: Configuration) -> Dict[str, str]:
    """Identifies knowledge gaps and generates a follow-up query.

    Analyzes the current summary to identify areas for further research and generates
    a new search query to address those gaps. Uses structured output to extract the
    follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic.
        config: Configuration for the runnable, including LLM provider settings.

    Returns:
        Dictionary with state update, including search_query key containing the
        generated follow-up query.
    """

    llm_json_mode = ChatOpenAI(
        model=config.llm_model,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    result = llm_json_mode.invoke(
        [
            SystemMessage(
                content=reflection_instructions.format(
                    research_topic=state.research_topic
                )
            ),
            HumanMessage(
                content=(
                    f"Reflect on our existing knowledge:\n === \n"
                    f"{state.running_answer}, \n === \n And now identify a knowledge"
                    " gap and generate a follow-up web search query:"
                )
            ),
        ]
    )

    try:
        reflection_content = json.loads(result.content)
        query = reflection_content.get("follow_up_query")
        if not query:
            return {"search_query": f"Tell me more about {state.research_topic}"}
        return {"search_query": query}
    except (json.JSONDecodeError, KeyError, AttributeError):
        # If parsing fails or the key is not found, use a fallback query
        return {"search_query": f"Tell me more about {state.research_topic}"}


def finalize_summary(state: SummaryState) -> Dict[str, str]:
    """Finalizes the research summary.

    This function deduplicates and formats the sources gathered during the research
    process, then combines them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered.

    Returns:
        Dictionary with state update, including running_answer key containing the
        formatted final summary with sources.
    """
    seen_sources = set()
    unique_sources = []

    for source in state.sources_gathered:
        for line in source.split("\n"):
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)

    all_sources = "\n".join(unique_sources)
    state.running_answer = f"{state.running_answer}\n\n### Sources:\n{all_sources}"
    return {"running_answer": state.running_answer}


def finalize_pdf_summary(state: SummaryState) -> Dict[str, str]:
    """Finalizes the PDF summary with formatted text and embedded base64 images.

    This function constructs a markdown report by combining textual content
    from CompositeElements and inline base64 image strings in their original order.
    It appends deduplicated sources at the end under a 'Sources' section.

    Args:
        state: SummaryState containing the research elements and gathered sources.

    Returns:
        A dictionary updating the `running_answer` key with the full markdown-formatted
        report.
    """
    markdown_blocks = []
    image_blocks = []

    for elem in state.running_elements:
        if isinstance(elem, CompositeElement):
            markdown_blocks.append(elem.text.strip())
        elif isinstance(elem, str):
            try:
                image_bytes = base64.b64decode(elem)
                image_blocks.append(BytesIO(image_bytes))
                markdown_blocks.append(None)  # Image placeholder
            except Exception as e:
                markdown_blocks.append(f"⚠️ Failed to decode image: {e}")

    return {"content": markdown_blocks, "images": image_blocks}


def route_research(
    state: SummaryState, config: Configuration
) -> Literal["finalize_summary", "web_research"]:
    """Routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count.
        config: Configuration, including max_web_research_loops setting.

    Returns:
        String literal indicating the next node to visit
        ("web_research" or "finalize_summary").
    """
    if state.research_loop_count <= config.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary"
