<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h2 align="center">🤖 AI Agent Frameworks 🤖</h2>

  <p align="center">
    A hands-on comparison of modern AI agent and multi-agent frameworks. Get started with practical examples and explore the unique features of each framework.
    <br />
    <a href="./issues/new?labels=bug&template=bug-report.md">Report Bug</a>
    ·
    <a href="./issues/new?labels=enhancement&template=feature-request.md">Request Feature</a>
  </p>
</div>

This repository provides a comprehensive, hands-on comparison of modern AI agent and multi-agent frameworks. Each framework is explored through practical examples, highlighting its unique features, capabilities, and use cases.

## 🧠 Frameworks Included

- [**AG2**](https://ag2.ai/)
  ↳ Code repository: https://github.com/ag2ai/ag2
- [**Agno**](https://docs.agno.com/introduction)
  ↳ Code repository: https://github.com/agno-agi/agno
- [**Autogen**](https://microsoft.github.io/autogen/stable/)
  ↳ Code repository: https://github.com/microsoft/autogen
- [**CrewAI**](https://www.crewai.com/)
  ↳ Code repository: https://github.com/crewAIInc/crewAI
- [**LangGraph**](https://langchain-ai.github.io/langgraph/)
  ↳ Code repository: https://github.com/langchain-ai/langgraph
- [**LlamaIndex**](https://docs.llamaindex.ai/en/stable/)
  ↳ Code repository: https://github.com/run-llama/llama_index
- [**OpenAI Agents SDK**](https://openai.github.io/openai-agents-python/)
  ↳ Code repository: https://github.com/openai/openai-agents-python
- [**Pydantic-AI**](https://ai.pydantic.dev/)
  ↳ Code repository: https://github.com/pydantic/pydantic-ai
- [**SmolAgents**](https://huggingface.co/docs/smolagents/en/index)
  ↳ Code repository: https://github.com/huggingface/smolagents
  
## 📁 Structure

The repository is organized by framework, with each top-level folder containing examples, configuration, and a README for that framework. Examples range from simple agent tasks to advanced multi-agent workflows, RAG (Retrieval-Augmented Generation), API integration, and more.

**Main modules:**
- `ag2/`
- `agno/`
- `autogen/`
- `crewai/`
- `langgraph/`
- `llama-index/`
- `openai-agents-sdk/`
- `pydantic-ai/`
- `smolagents/`
- `study-agents-differences/`

Some modules are standalone, while others are PDM projects or use `requirements.txt` for dependency management. Always check the `README.md` in each module for specific setup and usage instructions.

## 🚀 Getting Started

1. **Choose a framework**: Navigate to the relevant folder for the agent framework you want to explore.
2. **Install dependencies**: See each module’s `README.md` for installation instructions.
3. **Run examples**

---

## 🧪 Comparison and Experiments

The `study-agents-differences/` folder contains scripts and utilities for comparing frameworks on common tasks, including RAG, API integration, and multi-agent workflows. It provides:
- Unified agent interfaces for Agno, LangGraph, LlamaIndex, and OpenAI
- Benchmarks for response time, token usage, and tool utilization
- Results and analysis for different agent designs and tool integrations
- A Streamlit UI for interactive comparison (`streamlit run agent-ui.py`)

---

## 🤝 Contributing
All contributions are welcome! If you have suggestions for new examples, frameworks to add, or improvements to existing content, please open an issue or submit a pull request.

--- 

#### Notes

- Some modules use PDM (`pyproject.toml`), others use `requirements.txt`. **Check each module’s `README.md` for installation and usage.**
- Install dependencies before running examples.
- Example `.env.example` files are provided where needed for API keys and settings.