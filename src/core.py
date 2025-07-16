import json
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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
        model_kwargs={
            "response_format": {"type": "json_object"}
        },
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
