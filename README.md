## AI Multi-Agent Frameworks Analysis

### Structure - [TODO]

The repository contains 4 modules: `autogen`, `crewai`, `langgraph`, and `smolagents`. Each module is a separate AI multi-agent framework that has been tested and analysed. The analysis includes an overview of the framework, its key features, use cases, and examples to demonstrate its functionality.

The repository is structured as follows:

```plaintext
agent-frameworks-analysis
├── autogen/
│   ├── autogen-simple-examples/
|   |   ├── src/
|   │   │   ├── async-human-input.py
|   │   │   ├── group-chat-with-rag.py
|   │   │   ├── hello-world.py
|   │   │   ├── nested-chats.py
|   │   ├── ...
|   │   ├── README.md
│   ├── autogen-project/
|   |   ├── knowledge_base/
│   │   ├── prompts/
│   │   ├── src/
|   │   │   ├── tools/
|   │   │   |   ├── geometric_mean_tool.py
|   │   │   |   ├── knowledge_base_seach_tool.py
|   │   │   ├── agents.py
|   │   │   ├── custom_agent.py
|   │   │   ├── index.py
|   │   │   ├── ...
|   │   │   ├── README.md
├── crewai/
├── langgraph/
├── smolagents/
```

**Note:** Some of the folders/modules are PDM projects and some contain requirements.txt files. **Make sure to install the dependencies before running the examples**, either by using PDM or pip. You can find the installation instructions in the README files of each module or sub-module.
