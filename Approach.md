# Approach Document: Conversational Assessment Agent

## 1. Design Choices
The core design philosophy for this agent prioritizes **modularity, strict schema adherence, and speed**. To ensure the agent is easily deployable and testable against automated evaluators, the monolithic structure was heavily refactored into focused modules:
- **`app.py`**: The FastAPI router managing the `/health` and `/chat` endpoints.
- **`models.py`**: Uses Pydantic to enforce the non-negotiable output schema (`reply`, `recommendations`, `end_of_conversation`).
- **`search.py`**: Handles catalog data and semantic retrieval.
- **`llm.py`**: Manages the LLM interaction and prompt engineering.

To handle the "stateless" constraint of the `/chat` endpoint, the system is designed to parse the entire historical array of messages sent by the user, dynamically re-evaluating the full conversational context (e.g., mid-conversation refinement of constraints) during every API call. 

## 2. Retrieval Setup (RAG)
**Final Setup:** We implemented a Retrieval-Augmented Generation (RAG) pipeline using the pre-computed `data/embeddings.npy`. 
- When a user submits a query, their entire conversation history is concatenated and embedded using `SentenceTransformer("all-MiniLM-L6-v2")`.
- We compute the cosine similarity between the query vector and the 377 catalog embeddings entirely in-memory using `numpy` and `scikit-learn`.
- The top 30 most relevant assessments are retrieved and formatted as a JSON string context for the LLM. 

**What didn't work:** 
1. **ChromaDB Dependency:** Our initial implementation attempted to connect to a local ChromaDB instance (`recommender.py`). This failed due to a missing/corrupted database collection resulting in "index out of range" errors. Moving to raw `.npy` and `numpy` eliminated this fragile dependency.
2. **Context Window Overload:** We initially attempted to bypass RAG by injecting the entire minified SHL catalog (~220KB) directly into the LLM system prompt. While modern LLMs have large context windows, this immediately triggered a `413 Payload Too Large` error due to hitting the strict 12,000 Tokens Per Minute (TPM) limit on Groq's API. 

**Measured Improvement:** By filtering the context down to the Top 30 semantic matches, payload size was reduced from ~55,000 tokens to ~4,500 tokens. This completely resolved the rate limit errors and dropped API latency well below the 30-second evaluator threshold.

## 3. Prompt Design
The System Prompt is the critical "brain" of the agent, specifically engineered to map to the four required behaviors:
- **Clarification:** Instructed to output an empty `[]` recommendations array and ask for more details if the user prompt is vague (e.g., "I need a test").
- **Recommendation & Refinement:** Instructed to recommend between 1 and 10 items only once sufficient context is established, utilizing the stateless history array to update its shortlist if constraints change.
- **Grounding (Comparisons):** Instructed explicitly to compare items (e.g. "What is the difference between OPQ and GSA?") using *only* the retrieved catalog context, preventing hallucination based on the model's priors.
- **Guardrails:** Explicitly instructed to "ONLY discuss SHL assessments" and refuse general hiring/legal advice or prompt injections, again outputting an empty `[]` array for recommendations during refusals.

We utilized the LLM's `json_object` mode (via Groq/OpenAI client) to guarantee that the agent's textual output is perfectly mapped to the Pydantic schema expected by the evaluator.

## 4. Evaluation Approach
We evaluated the agent by manually stress-testing the API via the Swagger UI (`/docs`). 
- **Readiness:** Confirmed `/health` reliably returns `{"status": "ok"}` with a 200 HTTP code.
- **Behavioral Testing:** We sent multi-turn stateless JSON payloads. We verified that generic queries yielded an empty recommendations list and a follow-up question. 
- **Context Injection Testing:** By asking for a "coding assessment for software developers," we verified that the Semantic Search successfully retrieved "Automata" and "Smart Interview Live Coding" from the catalog and that the LLM appropriately packaged them into the strict 1-to-10 item limit schema.
