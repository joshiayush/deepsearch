import json
from typing import Dict

from langchain_openai import ChatOpenAI
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
    query_writer_instructions,
    summarizer_instructions,
    reflection_instructions,
    get_current_date,
)
from configuration import Configuration
from utils import strip_thinking_tokens


def generate_query(research_topic: str, config: Configuration) -> Dict[str, str]:
    """Generates a search query based on the research topic.

    Uses an LLM to create an optimized search query for web research based on the user's
    research topic.
    """
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date, research_topic=research_topic
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
