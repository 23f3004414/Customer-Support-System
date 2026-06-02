import os
from pathlib import Path
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing import List
from langchain.schema import Document
from vector_store import VectorStore
from langchain_core.output_parsers import StrOutputParser


def _format_chat_history(history: List[BaseMessage]) -> str:
    lines = []
    for msg in history:
        if isinstance(msg, SystemMessage):
            continue
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines) if lines else "None"


class LLMRAGHandler:
    """
    A class to handle LLM-based RAG (Retrieval-Augmented Generation) tasks.
    
    Attributes:
        llm (ChatOllama): The language model used for generating responses.
        vector_store (VectorStore): The vector store used for document retrieval.
        system_prompt (str): The system prompt given to the model.
        history (List[BaseMessage]): The conversation history.
        rag_prompt (PromptTemplate): The prompt template for q&a with RAG.
        llm_chain (Chain): The chain for RAG.
        rag_chain (Chain): The retrieval chain.
    
    Methods:
        __init__(self, model=None): Initializes the LLMRAGHandler with the specified model (or OLLAMA_MODEL env).
        generate_response(self, human_message) -> AIMessage: Generates and appends a response from the LLM.
        reset(self) -> None: Resets the conversation history.
        get_history(self) -> List[BaseMessage]: Returns the conversation history.
        retrieve(self, question: str, k:int = 4) -> List[Document]: Retrieves the most relevant documents for a given question.
        add_pdf_to_context(self, filePath: Path): Adds a PDF file to the context for retrieval.

    """
    def __init__(self, model=None):
        """        
        Initializes the LLMRAGHandler with the specified model.

        Args:
            model (str | None): Ollama chat model. Defaults to OLLAMA_MODEL / OLLAMA_CHAT_MODEL
                (llama3.2:1b). Embeddings use OLLAMA_EMBED_MODEL (nomic-embed-text).
        """
        chat_model = model or os.environ.get("OLLAMA_MODEL") or os.environ.get(
            "OLLAMA_CHAT_MODEL", "llama3.2:1b"
        )
        embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.llm = ChatOllama(model=chat_model)
        self.vector_store = VectorStore(embedding_model=embed_model)
        
        # System prompt - These are the instructions for the model
        self.system_prompt = (
            "You are a helpful assistant that answers questions about uploaded documents. "
            "Use ONLY the provided Context to answer. If the answer appears in the Context, "
            "state it clearly. Do not claim information is missing when it is present in the Context."
        )
        
        # keep track of the conversation history
        self.history = []
        self.history.append(SystemMessage(content=self.system_prompt))

        # prompt template for q&a with rag
        self.rag_prompt = PromptTemplate.from_template(
            "Context:\n{context}\n\n"
            "Previous conversation:\n{chat_history}\n\n"
            "Question: {input}\n\n"
            "Answer using the Context above:"
        )

        self.llm_chain = self.rag_prompt | self.llm | StrOutputParser()

    
    def generate_response(self, human_message) -> AIMessage:
        """
        Generates and appends a response from the LLM.

        Args:
            human_message (str): The user's message.

        Returns:
            AIMessage: The AI's response.
        """
        docs = self.vector_store.retrieve_documents(human_message, k=8)
        if not docs:
            answer = (
                "No document content is indexed yet. Upload a PDF in the sidebar "
                "and wait until it shows as indexed, then ask again."
            )
        else:
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)
            answer = self.llm_chain.invoke({
                "input": human_message,
                "context": context,
                "chat_history": _format_chat_history(self.history),
            })

        self.history.append(HumanMessage(content=human_message))
        self.history.append(AIMessage(content=answer))
        return answer

    def reset(self) -> None:
        """
        Resets the conversation history.
        """
        self.history = []
        self.history.append(SystemMessage(content=self.system_prompt))

    def get_history(self) -> List[BaseMessage]:
        """
        Returns the conversation history.

        Returns:
            List[BaseMessage]: The conversation history.
        """       
        return self.history
    
    def retrieve(self, question: str, k:int = 4) -> List[Document]:
        """
        Retrieves the most relevant documents for a given question.

        Args:
            question (str): The question to retrieve documents for.
            k (int): The number of documents to retrieve. Default is 4.

        Returns:
            List[Document]: The retrieved documents.
        """
        return self.vector_store.retrieve_documents(question, k=k)

    
    def add_pdf_to_context(self, filePath: Path) -> List[Document]:
        """
        Adds a PDF file to the context for retrieval.

        Args:
            filePath (Path): The path to the PDF file.
        Returns:
            List[Document]: The documents added to the vector store.
        """
        self.vector_store.add_document(filePath)
    
if __name__ == '__main__':
    print(ChatOllama.list_models())