# Comparison and Study of the agents from different Frameworks

In this section, we explore the differences between the agents from different frameworks. We will compare the agents from **[Agno](https://docs.agno.com/introduction)**, **[LangGraph](https://www.langchain.com/langgraph)**, **[LlamaIndex](https://docs.llamaindex.ai/en/stable/)**, and **[OpenAI](https://platform.openai.com/docs/guides/function-calling?api-mode=responses#tool-choice)**.

We design the agents using a consistent structure but with different tools and evaluate them across various metrics. The comparison focuses on response time, token usage, and tool utilization. Additionally, we assess their performance with specific tools, such as RAG and API integrations.

If you want to see the results of the study, you can jump to the [Results](#results) section.

---

## Setup

First, create a python virtual environment with:

```bash
python3 -m venv .venv
```

Then, activate the virtual environment with:

```bash
source .venv/bin/activate
```

Install the dependencies with:

```bash
pip install -r requirements.txt 
```

Create a `.env` file in the root of the project based on the `.env.example` file.


Now, you can run the examples by running each file.

---

## Agents

Each agent can be run by executing the corresponding file.
Some flags can be passed to the agents to change their behavior when running independently.

### Flags
- `--provider [provider]`: The LLM provider to be used for the agent. It can be `azure`, `openai` or `other`. Default is `azure`.
- `--mode [mode]`: The mode in which the agent will run. It can be None, `metrics` or `metrics-loop`. Both these methods will return the execution time and token usage of the agent. Default is `None`, which will run a simple chatbot interface in the terminal.
- `--iter [int]`: Number of iterations the agent will run. Default is 1. (Only needed for `metrics-loop` mode).
- `--no-memory`: If you want to run the agent without memory. Default is False.
- `--create`: If you want to create an Agent's instance in every iteration. Default is False.
- `--verbose`: If you want to see the agent's logs and responses. Default is True for Normal mode and False for `metrics` and `metrics-loop` modes.
- `--file [output file]`: If you want to save the agent's responses to a file. Default is False. (Only needed for `metrics` and `metrics-loop` modes)


*Example:*
```bash
python llama_index_rag_api_agent.py --mode metrics-loop --iter 30 --create --no-memory --verbose --file tests/test100_llamaindex_rag.txt
```

---

## Results

We evaluate the agents based on the following metrics:
- **Response Time (with Memory)**: The response time of the agents when they have memory.
- **Response Time (without Memory)**: The response time of the agents when they don't have memory.
- **Tokens**: The number of tokens used by the agent to respond to a prompt.
- **RAG & API**: The response time, number of tokens and number of misses of the agents when they use RAG and API tools.

We create agents with the following tools:
- **Web Search tool**: A tool that searches the web for information.
- **RAG tool**: A tool that does RAG over a local database.
- **API tools**: Tools that query APIs for information.


> 💡 Agent Performance metrics are influenced significantly by the system prompts provided to the agents.

---

### Response Time (with Memory)  
**Prompt:** _search the web for who won the Champions League final in 2024?_  

| Metrics                 | Agno          | LangGraph      | LlamaIndex     |
|-------------------------|--------------|---------------|---------------|
| Response time - 20x     | 5.41 ± 1.19s  | 6.04 ± 2.61s  | 5.36 ± 2.02s  |
| Response time - 30x     | 5.84 ± 1.01s  | 6.17 ± 1.14s  | 5.32 ± 2.26s  |
| Response time - 50x     | 4.24 ± 0.78s  | 8.48 ± 2.56s  | 3.00 ± 3.24s  |
| Response time - 100x    | 4.39 ± 0.73s  | 9.45 ± 4.73s  | 2.64 ± 2.29s  |

#### **Comments:**  
**Agno:**
- Consistent and organized answers
- Formatted directly in markdown (if needed)
- No errors  

**LangGraph:**
- Well-structured responses
- First runs seem faster than the following
- **Bottleneck:** Memory - storing the conversation in memory negatively affects performance
- This effect scales with memory size; at iteration 100, performance drops significantly  

**LlamaIndex:**
- No unnecessary verbosity, just answers the prompt
- Consistent and direct answers
- No errors in 100 iterations  

---

### Response Time - More abstract prompt 
**Prompt:** _who won the Champions League final in 2024?_  

| Metrics                | Agno          | LangGraph      | LlamaIndex     | OpenAI         |
|------------------------|--------------|---------------|---------------|---------------|
| Response time - 100x   | 1.95 ± 1.34s  | 0.96 ± 0.72s  | 0.78 ± 0.55s  | 3.51 ± 0.83s  |

#### **Comments:**  
**Agno:**
- Recalls the tool after some iterations
- Can override by activating: `read_tool_call_history`
- Starts by calling the `web_search` tool
- Can read tool call history
- Answers are consistent - tool calling and memory work effectively  

**LlamaIndex & LangGraph:**
- Consistent short answers
- Fast information retrieval if stored in memory
- Sometimes redoes the tool call; ideally, it shouldn't and should answer correctly in one go  

---

### Response Time (without Memory)  
**(Delete and recreate the agent)**  

**Prompt:** _search the web for who won the Champions League final in 2024?_  

| Metrics                | Agno          | LangGraph      | LlamaIndex     | OpenAI         |
|------------------------|--------------|---------------|---------------|---------------|
| Response time - 50x    | 4.58 ± 1.03s  | 4.22 ± 1.11s | 4.12 ± 1.01s  | 3.83 ± 0.99s  |
| Response time - 100x   | 4.28 ± 0.76s  | 3.31 ± 0.59s  | 3.63 ± 0.66s  | 3.61 ± 0.83s  |

**Prompt:** _who won the Champions League final in 2024?_  

| Metrics                | Agno          | LangGraph      | LlamaIndex     | OpenAI         |
|------------------------|--------------|---------------|---------------|---------------|
| Response time - 100x   | 4.16 ± 0.65s  | 3.35 ± 0.56s  | 3.60 ± 0.61s  | 3.34 ± 0.51s  |

#### **Comments:**  
**All options:**
- 100% of the time, the tools are called
- Not specifying "search the web for" doesn’t affect response time  

**LlamaIndex:**
- Very straight to the point  

**LangGraph:**
- More verbose  

---

### Tokens  

| Metrics                | Agno                     | LangGraph                | LlamaIndex                | OpenAI                   |
|------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| LLM Prompt Tokens     | 1999.2                  | 1946.1                  | 2121.7                  | 1888.5                  |
| LLM Completion Tokens | 65.3                    | 53.5                    | 76.9                    | 58.3                    |
| Total LLM Token Count | 2064.5                  | 1999.7                  | 2198.6                  | 1946.7                  |

#### **Comments:**  
- Tokens heavily depend on the system prompt and agent context
- **Agno and LangGraph:** Each response contains metrics like time, tokens, time_to_first_token, etc., for **each step**
- **LlamaIndex:** Add a token counter to the LLM model constructor to track tokens ([Reference](https://docs.llamaindex.ai/en/stable/examples/observability/TokenCountingHandler/))

---

## RAG & API Times  

### **RAG**  

**Prompt:** _Ball possession in Benfica's game?_ (from `matches-1.md`)  

| Metrics                | Agno          | LangGraph      | LlamaIndex     |
|------------------------|--------------|---------------|---------------|
| Response time - 100x   | 3.30 ± 0.75s  | 2.68 ± 1.35s  | 2.86 ± 1.05s  |
| Tokens - 100x         | 4439.3       | 4877.2        | 3279.9        |
| Missed - 100x         | 2 / 100       | 4 / 100       | 2 / 100       |

**Prompt:** _Benfica's UCL match score?_  

| Metrics                | Agno          | LangGraph      | LlamaIndex     |
|------------------------|--------------|---------------|---------------|
| Response time - 100x   | 3.17 ± 0.74s  | 2.43 ± 1.09s  | 2.74 ± 0.79s  |
| Tokens - 100x         | 4515.9       | 5053.3        | 3341.0        |
| Misses - 100x         | 0 / 100       | 0 / 100       | 0 / 100       |

#### **Comments:**  
**LangGraph:** Had to modify text preprocessing before creating the VectorDB:
```python
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name=settings.embeddings_model_name,
    chunk_size=800, chunk_overlap=80
)
doc_splits = text_splitter.split_documents(documents)
```
- Initial retrieval failure rate was ~50%
- Likely needs fine-tuning for better retrieval accuracy

---

### **API**  
**Prompt:** _Tell me the waiting time at the CG station and the status of the red line, and also give me information about Formula 1 driver number 44!_  

| Metrics                | Agno          | LangGraph      | LlamaIndex     |
|------------------------|--------------|---------------|---------------|
| Response time - 100x   | 5.49 ± 1.40s  | 4.24 ± 1.35s  | 6.41 ± 2.47s  |
| Tokens - 100x         | 1849.2        | 1412.2        | 3913.4        |
| Misses - 100x         | 0 / 100       | 0 / 100       | 0 / 100       |

#### **Comments:**  
- **LlamaIndex:** The prompt was not fine-tuned so the number of tokens might be higher than expected, which leads to a higher response time.
  - This might need to be addressed by better customizing the prompt for the agent / LLM model.
---

## UI

To run the streamlit UI, run the following command:

```bash
streamlit run agent-ui.py
```

This will open a new tab in your browser with the UI where you can interact with the agents.
