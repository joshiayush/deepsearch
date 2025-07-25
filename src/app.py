import os
from dotenv import load_dotenv

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

if __name__ == "__main__":
    config = Configuration()
    research_topic = "Artificial Intelligence in Healthcare"
    state = SummaryState(research_topic=research_topic)
    query_result = generate_query(state, config)
    print(f"Generated Search Query: {query_result['search_query']}")

    for _ in range(config.max_web_research_loops):
        state.search_query = query_result["search_query"]
        print(f"Current Search Query: {state.search_query}")

        state.research_loop_count = int(os.getenv("MAX_WEB_RESEARCH_LOOPS", 3))
        web_research_result = web_research(state, config)

        state.web_research_results.extend(web_research_result["web_research_results"])
        state.sources_gathered.extend(web_research_result["sources_gathered"])
        state.research_loop_count = web_research_result["research_loop_count"]

        summary_result = summarize_sources(state, config)
        state.running_summary = summary_result["running_summary"]

        # Reflect on the summary to generate a follow-up query
        reflection_result = reflect_on_summary(state, config)
        print(f"Follow-up Search Query: {reflection_result['search_query']}")
        state.search_query = reflection_result["search_query"]

    finalized_result = finalize_summary(state)
    state.running_summary = finalized_result["running_summary"]
    # Final output
    print(f"Final Summary: {state.running_summary}")
