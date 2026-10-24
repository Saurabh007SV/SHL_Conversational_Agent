# 🤖 SHL Conversational Assessment Agent

An AI-powered conversational agent that helps users discover and select the right [SHL](https://www.shl.com/) assessments through natural language dialogue. Built with **FastAPI**, **semantic search**, and **LLM-driven recommendations**.

---

## ✨ Features

- **Conversational Interface** — Multi-turn chat that clarifies requirements before recommending assessments.
- **Semantic Search** — Uses `BAAI/bge-small-en-v1.5` embeddings with cosine similarity to retrieve the most relevant assessments from the SHL product catalog.
- **LLM-Powered Recommendations** — Generates structured, grounded recommendations via Groq (Llama 3.3 70B) or OpenAI (GPT-4o-mini), ensuring every recommended URL comes directly from the catalog.
- **Structured JSON Responses** — Every response includes a conversational reply, a list of recommended assessments (name + URL), and a conversation-complete flag.
- **Deployment-Ready** — Pre-configured for one-click deployment on [Render](https://render.com).

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌────────────────────┐
│   FastAPI (app.py)  │  ← POST /chat
└────────┬───────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌────────┐ ┌────────────┐
│ search │ │   llm.py   │
│  .py   │ │ (LLMAgent) │
└───┬────┘ └─────┬──────┘
    │            │
    ▼            ▼
Embeddings   Groq / OpenAI
(bge-small)     API
```

1. **`app.py`** — FastAPI server with `/chat` (POST) and `/health` (GET) endpoints.
2. **`search.py`** — `SemanticSearch` class that loads the SHL catalog + pre-computed embeddings, embeds the user query with [fastembed](https://github.com/qdrant/fastembed), and returns the top-k most similar assessments.
3. **`llm.py`** — `LLMAgent` class that builds a system prompt with the retrieved catalog context and calls the LLM with JSON-mode output to generate structured recommendations.
4. **`models.py`** — Pydantic models (`ChatRequest`, `ChatResponse`, `Message`, `Recommendation`) defining the API contract.

---

## 📂 Project Structure

```
conversational_assessment_agent/
├── app.py                  # FastAPI application & endpoints
├── llm.py                  # LLM agent (Groq / OpenAI)
├── models.py               # Pydantic request/response schemas
├── search.py               # Semantic search over the SHL catalog
├── data/
│   ├── shl_product_catalog_new.json   # Full SHL assessment catalog
│   └── embeddings.npy                 # Pre-computed BGE embeddings
├── requirements.txt        # Python dependencies
├── Procfile                # Gunicorn start command (Heroku/Render)
├── render.yaml             # Render deployment blueprint
├── runtime.txt             # Python version specification (3.11.9)
├── pyrefly.toml            # Pyrefly type-checker config
├── .env                    # Local environment variables (git-ignored)
└── .gitignore
```

---

## 🚀 Getting Started


### 1. Clone the Repository

```bash
git clone https://github.com/Saurabh007SV/SHL_Conversational_Agent.git
cd SHL_Conversational_Agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Choose one LLM provider:
GROQ_API_KEY=your_groq_api_key
# — or —
OPENAI_API_KEY=your_openai_api_key

# Optional: override the default model
MODEL=llama-3.3-70b-versatile
```

### 5. Run the Server

```bash
python app.py
```

The API will be available at **`http://localhost:8000`**.

---

## 📡 API Reference

### `GET /health`

Health check endpoint.

**Response:**
```json
{ "status": "ok" }
```

### `POST /chat`

Send a conversation to the agent and receive assessment recommendations.

**Request Body:**
```json
{
  "messages": [
    { "role": "user", "content": "I need a cognitive assessment for entry-level candidates, under 30 minutes." }
  ]
}
```

**Response Body:**
```json
{
  "reply": "Based on your requirements, here are some assessments I'd recommend...",
  "recommendations": [
    {
      "name": "Verify Interactive - General Ability",
      "url": "https://www.shl.com/solutions/products/assessments/..."
    }
  ],
  "end_of_conversation": false
}
```

| Field                  | Type                | Description                                                                 |
|------------------------|---------------------|-----------------------------------------------------------------------------|
| `reply`                | `string`            | The agent's conversational response.                                        |
| `recommendations`      | `Recommendation[]`  | List of recommended assessments (each with `name` and `url`).               |
| `end_of_conversation`  | `boolean`           | `true` when the agent considers the task complete; `false` otherwise.        |

---

## ⚙️ Configuration

| Environment Variable | Required | Default                    | Description                                                  |
|----------------------|----------|----------------------------|--------------------------------------------------------------|
| `GROQ_API_KEY`       | One of   | —                          | Groq API key (preferred provider).                           |
| `OPENAI_API_KEY`     | these    | —                          | OpenAI API key (fallback provider).                          |
| `MODEL`              | No       | `llama-3.3-70b-versatile`  | LLM model name. Auto-selects based on provider if unset.     |
| `OPENAI_BASE_URL`    | No       | —                          | Custom base URL for OpenAI-compatible endpoints.             |

---

## ☁️ Deployment (Render)

This project includes a [`render.yaml`](render.yaml) blueprint for one-click deployment:

1. Push the repo to GitHub.
2. Connect the repo in the [Render Dashboard](https://dashboard.render.com/).
3. Set `GROQ_API_KEY` (or `OPENAI_API_KEY`) as an environment variable in the Render dashboard.
4. Deploy — Render will automatically run:
   ```
   pip install -r requirements.txt
   gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
   ```

---

## 🧠 How It Works

1. **User sends a message** via `POST /chat` with the full conversation history.
2. **Semantic search** embeds the user's query using `BAAI/bge-small-en-v1.5` and retrieves the top 30 most relevant assessments from the pre-computed embeddings.
3. **LLM generates a response** using the retrieved catalog as grounding context. It may:
   - Ask clarifying questions (job level, skills, duration, etc.)
   - Recommend 1–10 assessments with exact names and URLs from the catalog
   - Refine recommendations based on updated constraints
   - Compare assessments when asked
4. **Structured JSON** is returned with the reply, recommendations, and conversation state.

---

## 🛠️ Tech Stack

| Component          | Technology                                                   |
|--------------------|--------------------------------------------------------------|
| Web Framework      | [FastAPI](https://fastapi.tiangolo.com/)                     |
| Data Validation    | [Pydantic](https://docs.pydantic.dev/)                       |
| Embedding Model    | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) via [fastembed](https://github.com/qdrant/fastembed) |
| Similarity Search  | [scikit-learn](https://scikit-learn.org/) (cosine similarity) |
| LLM Provider       | [Groq](https://groq.com/) / [OpenAI](https://openai.com/)   |
| ASGI Server        | [Uvicorn](https://www.uvicorn.org/) + [Gunicorn](https://gunicorn.org/) |
| Deployment         | [Render](https://render.com/)                                |

---

## 📄 License

This project is for educational and demonstration purposes.
