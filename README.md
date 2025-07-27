# DeepSearch

Deepsearch is a research assistant that is built on top of [langchain-ai/local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher). While having the web search ability, now it also has the ability to search through your local documents (PDF). Give it a topic and it will generate a search query, gather search results, create a brief of the search results, reflect on the answer to examine knowledge gaps, generate a new search query to address the gaps, and repeat for a user-defined number of cycles. It will provide the user a final markdown with the relevant images found during the search.

<img width="1470" height="956" alt="Screenshot 2025-07-27 at 4 29 20 PM" src="https://github.com/user-attachments/assets/f30de842-a26f-4e7b-8e0c-9185c1aff467" />

**Note:** When PDF files are uploaded the "search" mode is disabled automatically.

## 🚀 Quickstart

Clone the repository:
```shell
git clone https://github.com/joshiayush/deepsearch.git
cd deepsearch
```

Then edit the `.env` file to customize the environment variables according to your needs. These environment variables control the search tools, and other configuration settings. When you run the application, these values will be automatically loaded via `python-dotenv`.
```shell
cp .env.example .env
```

### Selecting search tool

By default, it will use [DuckDuckGo](https://duckduckgo.com/) for web search, which does not require an API key. But you can also use [SearXNG](https://docs.searxng.org/), [Tavily](https://tavily.com/) or [Perplexity](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api) by adding their API keys to the environment file. Optionally, update the `.env` file with the following search tool configuration and API keys. If set, these values will take precedence over the defaults set in the `Configuration` class in `configuration.py`. 
```shell
SEARCH_API=xxx # the search API to use, such as `duckduckgo` (default)
TAVILY_API_KEY=xxx # the tavily API key to use
PERPLEXITY_API_KEY=xxx # the perplexity API key to use
MAX_WEB_RESEARCH_LOOPS=xxx # the maximum number of research loop steps, defaults to `3`
FETCH_FULL_PAGE=xxx # fetch the full page content (with `duckduckgo`), defaults to `false`
```

### Running with Streamlit

#### Mac

1. (Recommended) Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
brew install poppler tesseract libmagic
pip install -r requirements.txt
```

3. Launch streamlit:

```bash
streamlit run src/app.py
```

#### Linux

1. (Recommended) Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
apt-get install poppler-utils tesseract-ocr libmagic-dev
pip install -r requirements.txt
```

3. Launch streamlit:

```bash
streamlit run src/app.py
```
