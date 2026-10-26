# AuroraStay QnA — RAG POC

Internal question-answering tool for AuroraStay Hotels staff. Answers are grounded only in the four AuroraStay policy documents and shown with their supporting source chunks.

## Architecture
Streamlit UI (port 8501) -> Flask REST API (port 7860, gunicorn) -> FAISS index + HarishMaths/Hotel-Policy-Embedding -> gpt-4o-mini

## Repository layout
- `backend/` — `app.py` (Flask API), `rag_utils.py` (retrieve / generate / rag), `faiss_index/` (serialized vectors), `requirements.txt`, `Dockerfile`
- `frontend/` — `app.py` (Streamlit UI), `requirements.txt`, `Dockerfile`

## Run in GitHub Codespaces
1. Add Codespaces secrets `OPENAI_API_KEY` and `OPENAI_API_BASE` for this repository.
2. `cd backend && docker build -t aurora-backend . && cd ..`
3. `cd frontend && docker build -t aurora-frontend . && cd ..`
4. `docker run --name backend --network host -e OPENAI_API_KEY=$OPENAI_API_KEY -e OPENAI_API_BASE=$OPENAI_API_BASE aurora-backend`
5. New terminal: `docker run --name frontend --network host aurora-frontend`
6. Make ports 7860 and 8501 public in the Ports tab and open the 8501 URL.

## API
| Method | Path | Body |
|---|---|---|
| GET | `/health` | — |
| POST | `/v1/relevant_chunks` | `query`, `k` |
| POST | `/v1/answer` | `query`, `k`, `model_name`, `temperature`, `top_p`, `max_tokens` |
| POST | `/v1/answer_with_relevant_chunks` | same as above |
| POST | `/query` | `question`, `k`, ... -> `question`, `answer`, `sources`, `retrieved_chunks`, `confidence` |

Delete the Codespace after testing to avoid storage charges.
