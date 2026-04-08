from __future__ import annotations
import os
from typing import Any
from langchain.agents import create_react_agent
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent as create_agent

MODEL = os.getenv("MODEL", "openai:gpt-4.1-mini")

SYSTEM_PROMPT = """You are a concise and accurate research assistant.
Use the web_search tool when fresh or factual information is needed.
Always cite sources in plain text where useful, summarize results clearly,
and avoid making up facts when search results are incomplete.
Limit yourself to at most 3 tool calls before answering."""

search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

model = init_chat_model(MODEL, temperature=0)

graph = create_agent(
    model=model,
    tools=[search_tool],
    prompt=SYSTEM_PROMPT,
)

if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "What are the latest LangSmith features?"}]},
        config={"recursion_limit": 5},
    )
    print(result)
