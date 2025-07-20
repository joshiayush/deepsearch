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
        Dictionary with state update, including running_summary key containing the
        updated summary.
    """
    existing_summary = state.running_summary
    most_recent_web_research = state.web_research_results[-1]

    if existing_summary:
        human_message_content = (
            f"<Existing Summary>\n{existing_summary}\n<Existing Summary>\n\n"
            f"<New Context>\n{most_recent_web_research}\n<New Context>"
            f"Update the Existing Summary with the New Context on this topic:\n"
            f"<User Input>\n{state.research_topic}\n<User Input>\n\n"
        )
    else:
        human_message_content = (
            f"<Context>\n{most_recent_web_research}\n<Context>"
            f"Create a Summary using the Context on this topic:\n"
            f"<User Input>\n{state.research_topic}\n<User Input>\n\n"
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
    running_summary = result.content
    if config.strip_thinking_tokens:
        running_summary = strip_thinking_tokens(running_summary)

    return {"running_summary": running_summary}


def reflect_on_summary(state: SummaryState, config: Configuration) -> Dict[str, str]:
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
                    f"{state.running_summary}, \n === \n And now identify a knowledge"
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
