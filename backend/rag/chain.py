"""
LangChain RAG chain for answering health questions.

Retrieves the most relevant health documents from Chroma, injects them into
a coaching-style prompt, and streams the answer from GPT-4o-mini.

Public API:
  build_rag_chain(embeddings=None, llm=None) -> Runnable
  ask(question, embeddings=None, llm=None) -> str
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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
    Build a retrieval-augmented generation chain.

    Returns a LangChain Runnable that accepts a question string and returns
    a plain-text answer. Pass custom `embeddings` or `llm` to substitute
    fakes in tests without touching environment variables.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    if embeddings is None:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )
    if llm is None:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=settings.openai_api_key,
        )

    retriever = get_vectorstore(embeddings).as_retriever(search_kwargs={"k": 8})
    prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])

    def _join_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    return (
        {"context": retriever | _join_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def ask(question: str, embeddings=None, llm=None) -> str:
    """Convenience wrapper: build the chain and run a single question."""
    return build_rag_chain(embeddings=embeddings, llm=llm).invoke(question)
