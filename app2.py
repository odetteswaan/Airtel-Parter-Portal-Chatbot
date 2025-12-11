import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from quardKeyword import allowed_topics_expanded
# ------------------------------
# CONFIG
# ------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY2")
INDEX_NAME = os.getenv("INDEX_NAME")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Embeddings
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# HuggingFace Model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.4,
    max_new_tokens=300,        # IMPORTANT FIX
)

chat_model = ChatHuggingFace(llm=llm)

# ------------------------------
# GUARDRAILS
# ------------------------------

ALLOWED_TOPICS = allowed_topics_expanded

BLOCK_KEYWORDS = [
    "joke", "poem", "story", "love", "funny", "song", "rap",
    "movie", "cricket", "football"
]

def passes_guardrails(query: str):

    q = query.lower()

    # Hard block irrelevant stuff
    if any(b in q for b in BLOCK_KEYWORDS):
        return False, "I can only answer Airtel Partner Portal related queries."

    # Allow if query contains ANY telecom-related keywords
    if any(a in q for a in ALLOWED_TOPICS):
        return True, ""

    # Final fallback: block
    return False, "I can only answer Airtel Partner Portal related queries."


# ------------------------------
# STREAMLIT
# ------------------------------
st.set_page_config(page_title="Airtel Partner Portal Chatbot")
st.title("🤖 Airtel Partner Portal Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ------------------------------
# RAG Context Fetch
# ------------------------------
def get_context(query):
    emb = model.encode(query).tolist()
    results = index.query(vector=emb, top_k=5, include_metadata=True)

    if not results.matches:
        return "No relevant data found."

    return "\n".join([m["metadata"].get("text", "") for m in results.matches])


# ------------------------------
# LLM Answer
# ------------------------------
def generate_answer(query, context):

    messages = [
        SystemMessage(
            content="""
You are the Airtel Partner Portal Chatbot.
Rules:
- Answer ONLY from context.
- If answer not found, reply "No".
- Keep replies concise and factual.
"""
        )
    ]

    messages.extend(st.session_state.chat_history)

    messages.append(
        HumanMessage(
            content=f"Question: {query}\n\nContext:\n{context}"
        )
    )

    try:
        resp = chat_model.invoke(messages)
        return resp.content.strip()
    except Exception as e:
        return f"Error generating answer: {e}"


# ------------------------------
# Chatbox UI
# ------------------------------
user_query = st.text_input("Ask something about Airtel Partner Portal:")

if user_query:

    allowed, guard_msg = passes_guardrails(user_query)

    if not allowed:
        reply = guard_msg
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=reply))
    else:
        context = get_context(user_query)
        answer = generate_answer(user_query, context)

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=answer))


# ------------------------------
# Show Chat History
# ------------------------------
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"🧑‍💼 **You:** {msg.content}")
    else:
        st.markdown(f"🤖 **Bot:** {msg.content}")
