"""
Vector Store Module
Handles document storage and retrieval using ChromaDB.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional
import hashlib
import os


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "web_documents",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _generate_id(self, text: str, url: str) -> str:
        """Generate a unique ID for a document chunk."""
        content = f"{url}:{text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def add_document(self, content: str, metadata: dict) -> int:
        """
        Add a document to the vector store.

        Args:
            content: The text content to store
            metadata: Metadata associated with the document

        Returns:
            Number of chunks added
        """
        # Split content into chunks
        chunks = self.text_splitter.split_text(content)

        if not chunks:
            print("No content to add")
            return 0

        # Generate IDs and prepare data
        ids = []
        metadatas = []
        documents = []

        url = metadata.get('url', 'unknown')

        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_id(chunk, f"{url}_{i}")
            ids.append(chunk_id)

            chunk_metadata = {
                **metadata,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            metadatas.append(chunk_metadata)
            documents.append(chunk)

        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self._get_embeddings(documents)

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

        print(f"Added {len(chunks)} chunks from {metadata.get('title', url)}")
        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None
    ) -> List[dict]:
        """
        Search for relevant documents.

        Args:
            query: The search query
            n_results: Number of results to return
            where: Optional filter conditions

        Returns:
            List of relevant document chunks with metadata
        """
        # Generate query embedding
        query_embedding = self._get_embeddings([query])[0]

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })

        return formatted_results

    def get_all_sources(self) -> List[str]:
        """Get all unique source URLs/paths in the collection."""
        all_data = self.collection.get(include=["metadatas"])
        sources = set()
        if all_data['metadatas']:
            for metadata in all_data['metadatas']:
                # Check for 'url' (web sources) or 'source' (uploaded files)
                if 'url' in metadata:
                    sources.add(metadata['url'])
                elif 'source' in metadata:
                    sources.add(metadata['source'])
        return list(sources)

    def delete_by_url(self, url: str) -> int:
        """
        Delete all chunks from a specific URL or source.

        Args:
            url: The URL or source path to delete

        Returns:
            Number of chunks deleted
        """
        # Try to get documents with 'url' field first (web sources)
        results = self.collection.get(
            where={"url": url},
            include=["metadatas"]
        )

        # If no results, try 'source' field (uploaded files)
        if not results['ids']:
            results = self.collection.get(
                where={"source": url},
                include=["metadatas"]
            )

        if results['ids']:
            self.collection.delete(ids=results['ids'])
            return len(results['ids'])
        return 0

    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        count = self.collection.count()
        sources = self.get_all_sources()
        return {
            'total_chunks': count,
            'total_sources': len(sources),
            'sources': sources
        }

    def clear(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector store cleared")

