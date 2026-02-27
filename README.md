## Multi‑Step RAG with LangGraph & Groq

A minimal, end‑to‑end example of a **multi‑step Retrieval‑Augmented Generation (RAG)** system built with **LangGraph**, **LangChain**, **Chroma**, **Hugging Face embeddings**, and **Groq LLMs**.

Instead of a single “ask a question → get an answer” call, this project demonstrates how to orchestrate a **multi‑node graph** that:

- **Rewrites questions** for better retrieval
- **Classifies questions** as on‑topic / off‑topic
- **Retrieves and grades documents**
- **Refines unclear questions**
- **Generates answers or gracefully declines** when it can’t help

The example domain is a fictional gym, **“Peak Performance Gym”**, with all knowledge stored as in‑memory documents.

---

### Features

- **Multi‑step RAG pipeline**
  - Question rewriting
  - On‑topic classification
  - Retrieval + relevance grading
  - Question refinement loop
  - Final answer generation
- **LangGraph orchestration**
  - Nodes for each reasoning / retrieval stage
  - Conditional routing between nodes
  - In‑memory checkpointing via `MemorySaver`
- **Modern RAG stack**
  - `Chroma` vector store for document retrieval
  - `HuggingFaceEmbeddings` (`sentence-transformers/all-miniLM-L6-v2`)
  - `ChatGroq` with `llama-3.1-8b-instant`
- **Small, easy‑to‑read codebase**
  - All logic lives in a single `main.py` file

---

### Tech Stack

- **Language**: Python (>= 3.13)
- **Orchestration**: `langgraph`
- **LLM / Chat Model**: `langchain-groq` (`ChatGroq`)
- **Vector Store**: `langchain-chroma` (`Chroma`)
- **Embeddings**: `langchain-huggingface` (`HuggingFaceEmbeddings`)
- **Core libraries**: `langchain`, `langchain-core`, `langchain-community`
- **Environment management**: `python-dotenv`

All direct dependencies are declared in `pyproject.toml` and `requirements.txt`.

---

### Project Structure

```text
Multi-Step-RAG/
├─ main.py           # Full multi-step RAG graph (LangGraph)
├─ pyproject.toml    # Project metadata & dependencies (for uv/pip)
├─ requirements.txt  # Minimal dependency list
├─ uv.lock           # uv lockfile (if you use uv)
└─ .python-version   # Python version pin
```

---

### How the Graph Works

The core logic is defined in `main.py` as a `StateGraph` over an `AgentState` `TypedDict`. At a high level:

1. **Question Rewriter (`question_rewriter`)**
   - Ensures the user’s latest question is captured.
   - If there is chat history, uses the LLM to **rephrase the latest question** into a standalone query optimized for retrieval.

2. **Question Classifier (`question_classifier`)**
   - Checks if the (rephrased) question is about one of the gym topics:
     - History & founder, hours, membership, classes, trainers, facilities, etc.
   - Produces an `on_topic` score of `"Yes"` or `"No"`.

3. **On‑Topic Router (`on_topic_router`)**
   - If `"Yes"` → routes to `retrieve`.
   - If `"No"` → routes to `off_topic_response`.

4. **Retriever (`retrieve`)**
   - Uses a `Chroma` vector store built over the in‑memory `docs` list.
   - Executes MMR retrieval with `k=4` on the rephrased question.

5. **Retrieval Grader (`retrieval_grader`)**
   - Uses the LLM with a structured `GradeDocument` schema.
   - For each retrieved document, decides if it is **relevant (`Yes`)** or **irrelevant (`No`)** to the question.
   - Controls a `proceed_to_generate` flag based on relevance.

6. **Proceed Router (`proceed_router`)**
   - If there are good documents → route to `generate_answer`.
   - If not, but refinements are still allowed → route to `refine_question`.
   - If the question cannot be clarified after several attempts → route to `cannot_answer`.

7. **Question Refiner (`refine_question`)**
   - Slightly adjusts the question to improve retrieval quality.
   - Increments a `rephrase_count` to avoid infinite loops.
   - Routes back into retrieval for another pass.

8. **Answer Generator (`generate_answer`)**
   - Builds a **history string** from prior messages.
   - Gathers document contents into a **context string**.
   - Invokes a RAG chain: `ChatPromptTemplate` → `ChatGroq`.
   - Returns a final natural‑language answer.

9. **Fallback Nodes**
   - `off_topic_response` – polite response if the question is outside gym topics.
   - `cannot_answer` – polite response if it can’t confidently answer from the docs.

All of these nodes are wired into a `StateGraph` with edges, conditional edges, and a `MemorySaver` checkpoint.

---

### Prerequisites

- **Python**: 3.13 or higher (see `.python-version`)
- **Groq API key**: You must have a Groq account and API key.
- (Optional) **uv**: Recommended for modern, fast Python dependency management.

---

### Setup

#### 1. Clone the repository

```bash
git clone https://github.com/Upnit-b/RAG-LangGraph-Multi-Step-Pipeline.git
cd Multi-Step-RAG
```

#### 2. Create and activate a virtual environment (if not using uv)

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# .venv\Scripts\activate   # on Windows (PowerShell / CMD)
```

#### 3. Install dependencies

- **Option A – using uv (recommended if you have it installed)**

```bash
uv sync
```

- **Option B – using pip**

```bash
pip install -r requirements.txt
```

#### 4. Configure environment variables

This project uses `python-dotenv` to load environment variables from a `.env` file.

- Create a `.env` file in the project root:

```bash
touch .env
```

- Add your Groq API key:

```bash
echo "GROQ_API_KEY=your_groq_api_key_here" >> .env
```

Make sure **not** to commit your real API keys to Git.

---

### Running the Demo

Once dependencies and environment variables are set:

```bash
python main.py
```

What this does:

- Builds a Chroma vector store from the in‑memory `docs`.
- Instantiates the LangGraph workflow with all nodes and routes.
- Runs a sample query:
  - `"Who founded Peak Performance Gym?"`
- Prints the final graph response to the console.

You can modify the example query at the bottom of `main.py` to test other questions (on‑topic and off‑topic) and observe how the graph routes.

---

### Customizing the RAG System

- **Change the knowledge base**
  - Update the `docs` list in `main.py` with your own domain‑specific documents.
  - Each document is a `langchain_core.documents.Document` with `page_content` and optional `metadata`.

- **Swap the model**
  - Adjust the `ChatGroq` model (e.g. different Groq‑hosted models) in `main.py`.

- **Tune retrieval**
  - Modify the retriever configuration in `main.py`, e.g. `search_type`, `k`, and other `search_kwargs`.

- **Extend the graph**
  - Add new nodes (e.g. routing by user intent, safety filters, tool‑calling nodes).
  - Add new conditional edges in the `StateGraph` to experiment with advanced flows.

This repository is intentionally small so you can easily copy and adapt the pattern to your own projects.

---

### Troubleshooting

- **No response / errors when calling the LLM**
  - Double‑check that `GROQ_API_KEY` is correctly set in `.env`.
  - Ensure your key has access to the `llama-3.1-8b-instant` (or whichever) model you’ve configured.

- **Import errors**
  - Verify dependencies are installed in the environment you’re using.
  - If using multiple Python versions, ensure the one matching `.python-version` / `pyproject.toml` is active.

- **Vector store issues**
  - This example uses an in‑memory Chroma instance created via `Chroma.from_documents`, so it should work out of the box as long as embeddings load correctly.

---

## 📬 Contact

Built by [@Upnit-b](https://github.com/Upnit-b) — feel free to reach out via GitHub Issues for any suggestions or bugs.
