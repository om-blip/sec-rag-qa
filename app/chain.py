# chain.py
# This file takes a user question, retrieves relevant chunks,
# and sends them to Groq to generate a grounded answer.
# It also tracks conversation history so follow-up questions work.

import os
from groq import Groq
from dotenv import load_dotenv
from retriever import retrieve_chunks

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# We manually maintain conversation history as a list of messages.
# This is what gives the chatbot memory across turns.
conversation_history = []

SYSTEM_PROMPT = """You are a financial analyst assistant. You answer questions 
about company annual reports strictly based on the context provided.

Rules:
- Only use information from the provided context chunks
- Always mention which page number your answer comes from
- If the answer is not in the context, say "I could not find that in the document"
- Never make up numbers, percentages, or financial figures
- Be concise but complete
- Do not say 'According to Chunk' or reference chunk numbers"""


def ask(question: str, collection_name: str) -> dict:
    """
    Main function. Takes a question + collection ID.
    Returns a dict with 'answer' and 'sources' (list of page numbers).
    """
    # Step 1: retrieve the most relevant chunks for this question
    question = rewrite_query(question)
    chunks = retrieve_chunks(question, collection_name, top_k=5)

    # Step 2: format chunks into a context block for the prompt.
    # We include page numbers so the LLM can cite them in its answer.
    context = ""
    source_pages = []
    for i, chunk in enumerate(chunks):
        context += f"\n[Page {chunk['page']}]:\n{chunk['text']}\n"
        source_pages.append(chunk['page'])

    # Step 3: build the user message — question + context together.
    # We inject context here, not in the system prompt, so it changes
    # per question while the system prompt stays fixed.
    user_message = f"""Context from the annual report:
{context}

Question: {question}

Answer based only on the context above. Cite page numbers."""

    # Step 4: add to conversation history and call Groq.
    # We pass the full history each time — this is how LLMs get memory.
    # REPLACE the conversation_history.append + client.chat call with this:

    # Add a clean version to history (without bulky context)
    # so follow-up questions don't get confused by old chunk dumps
    conversation_history.append({
    "role": "user",
    "content": question  # just the question, not the chunks
    })

# But send context only for the CURRENT call, separately
    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history[:-1],  # history without current question
        {
            "role": "user",
            # current question gets the full context injected
            "content": user_message
        }
    ],
    temperature=0.1,
    max_tokens=1024
    )

    answer = response.choices[0].message.content

    # Step 5: save assistant response to history for follow-up questions
    conversation_history.append({
        "role": "assistant",
        "content": answer
    })

    return {
        "answer": answer,
        "sources": sorted(set(source_pages))  # deduplicated page numbers
    }


def reset_conversation():
    """Call this to start a fresh conversation (e.g. new document uploaded)."""
    global conversation_history
    conversation_history = []

def rewrite_query(question: str) -> str:
    """
    If the question is a follow-up (contains words like 'that', 'it', 'this',
    'compare', 'difference'), rewrite it as standalone using conversation history.
    """
    followup_signals = ["that", "it", "this", "compare", "difference", 
                        "more", "also", "what about", "how about"]
    
    is_followup = any(word in question.lower() for word in followup_signals)
    
    if not is_followup or not conversation_history:
        return question  # not a follow-up, use as-is
    
    # Ask the LLM to rewrite it as a standalone question
    last_exchange = conversation_history[-2:]  # last Q&A pair
    
    rewrite_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system", 
                "content": "Rewrite the follow-up question as a fully self-contained question. Return only the rewritten question, nothing else."
            },
            *last_exchange,
            {
                "role": "user",
                "content": f"Rewrite this as standalone: {question}"
            }
        ],
        temperature=0,
        max_tokens=100
    )
    
    rewritten = rewrite_response.choices[0].message.content.strip()
    print(f"  [Query rewritten: '{question}' → '{rewritten}']")
    return rewritten


# Test it directly: python app/chain.py <collection_id>
if __name__ == "__main__":
    import sys
    collection_id = sys.argv[1] if len(sys.argv) > 1 else "a6caf8154556"

    print("RAG Q&A ready. Type 'quit' to exit, 'reset' to clear history.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit":
            break
        if question.lower() == "reset":
            reset_conversation()
            print("Conversation reset.\n")
            continue
        if not question:
            continue

        result = ask(question, collection_id)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: Pages {result['sources']}\n")
        print("-" * 60 + "\n")