
import ast
import os
import re

import requests
import streamlit as st


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BACKEND_URL = "http://localhost:7860"
REQUEST_TIMEOUT = 180  # seconds; RAG answers can take up to a minute

# Friendly names for the four policy documents (shown as source badges)
SOURCE_LABELS = {
    "aurorastay_booking_terms_and_cancellation_policy.pdf": "Booking Terms & Cancellation Policy",
    "aurorastay_room_and_property_information.pdf": "Room & Property Information",
    "aurorastay_refunds_and_additional_charges_guide.pdf": "Refunds & Additional Charges Guide",
    "aurorastay_guest_conduct_accessibility_and_resolution_guidelines.pdf": "Guest Conduct, Accessibility & Resolution Guidelines",
}

SAMPLE_QUESTIONS = [
    "Is breakfast free at the Miami Brickell property?",
    "What is the pet charge at ASH-CHI Loop?",
    "Can a guest get a refund on an Advance Purchase rate?",
    "What is the fee for smoking in a non-smoking room?",
]


# ---------------------------------------------------------
# Page configuration and light branding
# ---------------------------------------------------------

st.set_page_config(
    page_title="AuroraStay Policy Assistant",
    page_icon="🏨",
    layout="wide"
)

st.markdown(
    """
    <style>
      .aurora-header {background: linear-gradient(90deg,#0B2545,#13315C); color:#fff;
                      padding:18px 24px; border-radius:10px; margin-bottom:12px;}
      .aurora-header h1 {color:#fff; margin:0; font-size:1.8rem;}
      .aurora-header p {color:#D6E4F0; margin:4px 0 0 0;}
      .source-badge {display:inline-block; background:#E8F1FA; color:#0B2545;
                     border-radius:12px; padding:3px 10px; margin:2px 6px 2px 0; font-size:0.85rem;}
    </style>
    <div class="aurora-header">
      <h1>🏨 AuroraStay Policy Assistant</h1>
      <p>Internal POC for front-desk and reservations staff · answers are grounded only in AuroraStay policy documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def chunk_source(chunk: str) -> str:
    """Extract the source PDF name from a chunk string formatted by rag_utils.retrieve()."""
    match = re.search(r"Metadata:\s*(\{.*\})", chunk, re.DOTALL)
    if match:
        try:
            meta = ast.literal_eval(match.group(1))
            return os.path.basename(str(meta.get("source", "unknown")))
        except (ValueError, SyntaxError):
            pass
    return "unknown"


def chunk_text(chunk: str) -> str:
    """Return only the policy text part of a chunk string."""
    text = chunk.split("Metadata:")[0]
    return text.replace("Content:", "", 1).strip()


def backend_status() -> str:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if r.status_code == 200:
            return f"🟢 Backend online · {r.json().get('vectors', '?')} vectors indexed"
        return "🟡 Backend reachable (health endpoint unavailable)"
    except requests.exceptions.RequestException:
        return "🔴 Backend offline"


# ---------------------------------------------------------
# Sidebar  -  Model parameters
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Parameters")

    k = st.slider(
        "k (chunks to retrieve)",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of relevant chunks fetched from the vector store."
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0 gives the most consistent, deterministic answers (recommended for policy questions)."
    )

    max_tokens = st.slider(
        "Max tokens",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
        help="Maximum number of tokens in the generated answer."
    )

    st.divider()
    st.caption(backend_status())
    st.caption("Model: gpt-4o-mini · Embeddings: HarishMaths/Hotel-Policy-Embedding · Vector DB: FAISS")


# ---------------------------------------------------------
# Main area
# ---------------------------------------------------------

st.write("Ask a question about AuroraStay booking, room, refund, charge, conduct or accessibility policies.")

with st.expander("💡 Example questions"):
    for q in SAMPLE_QUESTIONS:
        st.markdown(f"- {q}")


# ---------------------------------------------------------
# Response type selection (two options only)
# ---------------------------------------------------------

response_type = st.radio(
    "Response mode",
    ["Answer only", "Answer + Relevant Chunks"],
    horizontal=True
)


# ---------------------------------------------------------
# Query input
# ---------------------------------------------------------

query = st.text_input(
    "Your question",
    placeholder="e.g., What is the pet charge at ASH-CHI Loop?"
)


# ---------------------------------------------------------
# Submit
# ---------------------------------------------------------

if st.button("Submit", type="primary"):

    if not query.strip():
        st.warning("Please enter a question.")

    else:

        # Pick endpoint
        if response_type == "Answer only":
            endpoint = "/v1/answer"
        else:
            endpoint = "/v1/answer_with_relevant_chunks"

        # Build request payload with parameters
        payload = {
            "query": query,
            "k": k,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            with st.spinner("Searching AuroraStay policies and drafting an answer…"):
                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )

            if response.status_code == 200:

                result = response.json()

                st.subheader("Answer")
                st.success(result["answer"])

                # -- Answer + Relevant Chunks -------------------
                if response_type == "Answer + Relevant Chunks":

                    chunks = result.get("relevant_chunks", [])
                    sources = list(dict.fromkeys(chunk_source(c) for c in chunks))

                    st.subheader("Source documents")
                    st.markdown(
                        "".join(
                            f'<span class="source-badge">📄 {SOURCE_LABELS.get(s, s)}</span>'
                            for s in sources
                        ),
                        unsafe_allow_html=True,
                    )

                    st.subheader("Relevant Chunks")
                    for i, chunk in enumerate(chunks, start=1):
                        src = chunk_source(chunk)
                        with st.expander(f"Chunk {i} · {SOURCE_LABELS.get(src, src)}"):
                            st.markdown(chunk_text(chunk))

                st.caption("Verify against the source chunks before quoting fees or deadlines to a guest.")

            else:
                detail = ""
                try:
                    detail = response.json().get("error", "")
                except ValueError:
                    pass
                st.error(
                    f"Unable to get a response from the RAG API. "
                    f"Status code: {response.status_code}. {detail}"
                )

        except requests.exceptions.Timeout:
            st.error("The RAG API took too long to respond. Please try again.")

        except requests.exceptions.RequestException:
            st.error(
                "Unable to connect to the RAG API. "
                "Please make sure the backend service is running."
            )
