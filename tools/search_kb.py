from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from logger import section, log_in, log_out
import os

@tool
def search_kb(query: str) -> str:
    """Use this tool to fetch the vector knowledge base. Provide the 'query' to use to search in the knowledge base"""
    section("search_kb.py :: search_kb", "bold cyan")
    log_in("query", query)

    vectorstore = Chroma(
    persist_directory=os.path.abspath(os.path.join(os.path.dirname(__file__), "../kb/chroma_db")),
    embedding_function=OpenAIEmbeddings()
)

    retriever = vectorstore.as_retriever(search_kwargs={"k":10})
    log_in("retriever k", 10)

    retrieved_results = retriever.invoke(query)
    log_out("retrieved_results count", len(retrieved_results))
    for i, d in enumerate(retrieved_results):
        log_out(f"  doc[{i}] source", f"{d.metadata.get('source','?')} | content[:60]={str(d.page_content)[:60]!r}")

    return "\n\n".join(d.page_content for d in retrieved_results)



