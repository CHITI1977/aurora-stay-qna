
from langchain_openai import ChatOpenAI
import os

# Standard refusal message used whenever the retrieved policy text cannot support an answer
REFUSAL_MESSAGE = "Insufficient information found in AuroraStay policy documents."


# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 3, vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]


# Retrieve the top-k chunks together with their source file and similarity score
# (used by the /query endpoint for source attribution and the confidence indicator)
def retrieve_with_scores(query: str, k: int = 3, vectorstore=None) -> list[dict]:
    results = vectorstore.similarity_search_with_score(query, k=k)

    chunks = []
    for rank, (doc, distance) in enumerate(results, start=1):
        source = os.path.basename(str(doc.metadata.get("source", "unknown")))
        # Embeddings are L2-normalised, so squared L2 distance d = 2 - 2*cos  ->  cos = 1 - d/2
        similarity = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
        chunks.append({
            "rank": rank,
            "source": source,
            "similarity": round(similarity, 4),
            "content": doc.page_content,
        })
    return chunks


# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks, model) -> str:

    prompt = f"""
    You are an internal assistant for AuroraStay Hotels staff. You answer questions about
    AuroraStay hotel policies: bookings, cancellations, rooms, check-in/check-out, amenities,
    parking, pets, service animals, accessibility, guest conduct, refunds and additional charges.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or typical hotel-industry practice.
    3. Include all relevant information needed to answer the query completely, including
       amounts, times, deadlines, limits and the property they apply to.
    4. Do not omit important conditions, exceptions, approvals or escalation steps.
    5. If the query names a property, use that property's specific values; if the policy
       gives a general rule and a property-specific rule, state the property-specific one.
       Never transfer a fee, time or rule from one property to another.
    6. Focus only on information directly relevant to the user's query and do not include
       unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly in the first sentence.
    8. Keep the answer concise, clear, and easy for front-desk staff to relay to a guest.
    9. Do not invent fees, times, room types, policies, conditions or exceptions.
    10. If the retrieved information is insufficient to answer (for example, the property
        is not listed), start with exactly: "{REFUSAL_MESSAGE}"
        Then, in one sentence, give only the next step the retrieved policy states
        (such as checking the live property profile or contacting the property team).
        If the retrieved information conflicts, clearly mention the conflict.
    11. When the policy says a value must be verified, approved or escalated before it is
        quoted or applied, include that step in one short sentence.

    Answer:
    """

    return model.invoke(prompt)


# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 3,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks


# RAG pipeline variant used by the /query endpoint: same retrieval and prompt as rag(),
# but also returns source documents, similarity scores and a retrieval confidence value
def rag_with_sources(
    query: str,
    k: int = 3,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    scored_chunks = retrieve_with_scores(query=query, k=k, vectorstore=vectorstore)

    # Format the chunks exactly as retrieve() does so the prompt is identical to rag()
    formatted_chunks = [
        f"Content:\n{c['content']}\n\nMetadata:\n{{'source': '{c['source']}'}}"
        for c in scored_chunks
    ]

    answer = generate(query=query, retrieved_chunks=formatted_chunks, model=model).content

    # Unique source documents, in rank order
    sources = list(dict.fromkeys(c["source"] for c in scored_chunks))

    # Retrieval confidence = similarity of the best-matching chunk (0 when the model refused)
    top_similarity = scored_chunks[0]["similarity"] if scored_chunks else 0.0
    refused = REFUSAL_MESSAGE.lower() in answer.lower()
    confidence = 0.0 if refused else top_similarity

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": scored_chunks,
        "confidence": round(confidence, 4),
        "refused": refused,
    }
