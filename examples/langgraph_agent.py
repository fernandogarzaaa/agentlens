"""Evaluate a LangGraph agent with agentlens (pip install "agentlens[langgraph]").

Any compiled LangGraph graph that takes {"messages": [...]} can be wrapped in `LangGraphAgent` and
handed to `agentlens eval --agent examples.langgraph_agent:demo_agent` or `run_eval(...)`.
The callback handler turns graph nodes, model calls and tool calls into agentlens spans, so all the
trajectory, token and cost metrics work unchanged.

This demo uses LangChain's fake chat model so it runs offline; swap in a real chat model
(ChatOpenAI(base_url=...), ChatOllama, ...) for a real evaluation.

    python examples/langgraph_agent.py
"""

from __future__ import annotations

from typing import Any

from agentlens import AgentResult, Toolset, Tracer
from agentlens.adapters.langgraph import AgentLensCallbackHandler


class LangGraphAgent:
    """Adapter from a compiled LangGraph graph to agentlens' Agent protocol."""

    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def run(self, input: Any, toolset: Toolset, system_prompt: str | None = None, max_steps: int = 8,
            tracer: Tracer | None = None) -> AgentResult:
        messages: list[Any] = [("system", system_prompt)] if system_prompt else []
        messages += [("user", input)] if isinstance(input, str) else [(m["role"], m["content"]) for m in input]
        handler = AgentLensCallbackHandler(tracer)
        out = self.graph.invoke({"messages": messages}, config={"callbacks": [handler], "recursion_limit": 2 * max_steps + 1})
        final = out["messages"][-1]
        return AgentResult(final_answer=str(final.content), steps=sum(1 for m in out["messages"] if m.type == "ai"))


def demo_agent() -> LangGraphAgent:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def get_weather(city: str, unit: str = "celsius") -> str:
        """Current weather for a city."""
        return '{"city": "%s", "temperature": 18.0, "unit": "%s"}' % (city, unit)

    class FakeToolModel(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

    replies = iter([
        AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1"}],
                  usage_metadata={"input_tokens": 120, "output_tokens": 12, "total_tokens": 132}),
        AIMessage(content="It's 18°C in Paris.", usage_metadata={"input_tokens": 160, "output_tokens": 9, "total_tokens": 169}),
    ])
    return LangGraphAgent(create_react_agent(FakeToolModel(messages=replies), [get_weather]))


if __name__ == "__main__":
    from agentlens import run_eval
    from agentlens.cases import suite_from_dict

    suite = suite_from_dict([{
        "id": "weather-paris",
        "input": "What's the weather in Paris right now?",
        "expected_tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
        "trajectory": "strict",
        "final_answer": {"number": 18},
    }])
    result = run_eval(suite, demo_agent(), agent_label="langgraph-demo", model="fake")
    case = result.cases[0]
    print(f"success={case.success} tool_calls={case.tool_calls} tokens={case.total_tokens} steps={case.steps}")
