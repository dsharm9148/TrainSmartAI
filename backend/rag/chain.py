"""
LangChain RAG chain for answering health questions.

Retrieves the most relevant health documents from Chroma, injects them into
a coaching-style prompt, and streams the answer from GPT-4o-mini.

Public API:
  build_rag_chain(embeddings=None, llm=None) -> Runnable
  ask(question, embeddings=None, llm=None) -> str
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.config import settings
from backend.rag.indexer import get_vectorstore

_SYSTEM = """\
You are a personal health coach with access to the user's Apple Health data.
Use the health data excerpts below to answer their question concisely and helpfully.
If the data does not contain enough information to answer clearly, say so — do not invent numbers.
Cite specific dates or metrics from the context when relevant.
Be encouraging and evidence-based.

Health data context:
{context}"""

_HUMAN = "{question}"


def build_rag_chain(embeddings=None, llm=None):
    """
    Build a retrieval-augmented generation chain with chat history support.

    Returns a LangChain Runnable that accepts
        {"question": str, "history": list[BaseMessage]}
    and returns a plain-text answer. History may be an empty list.
    """
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    if embeddings is None:
        embeddings = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
    if llm is None:
        llm = ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
        )

    retriever = get_vectorstore(embeddings).as_retriever(search_kwargs={"k": 8})
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        MessagesPlaceholder("history"),
        ("human", _HUMAN),
    ])

    def _join_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    return (
        {
            "context": (lambda x: x["question"]) | retriever | _join_docs,
            "question": lambda x: x["question"],
            "history": lambda x: x.get("history", []),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def ask(
    question: str,
    history: list | None = None,
    embeddings=None,
    llm=None,
) -> str:
    """Single-turn or multi-turn wrapper. `history` is a list of BaseMessage."""
    chain = build_rag_chain(embeddings=embeddings, llm=llm)
    return chain.invoke({"question": question, "history": history or []})


def history_from_rows(rows) -> list:
    """Convert ChatMessage ORM rows (oldest-first) into LangChain messages."""
    out = []
    for r in rows:
        if r.role == "user":
            out.append(HumanMessage(content=r.content))
        else:
            out.append(AIMessage(content=r.content))
    return out
