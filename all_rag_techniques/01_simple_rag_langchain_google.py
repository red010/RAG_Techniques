"""
Simple RAG (Retrieval-Augmented Generation) System with Gemini

Overview:
This script implements a basic Retrieval-Augmented Generation (RAG) system for processing and querying PDF documents.
The system encodes the document content into a vector store using Gemini embeddings, which can then be queried to retrieve relevant information.

Key Components:
1. PDF processing and text extraction
2. Text chunking for manageable processing
3. Vector store creation using Chroma and Gemini embeddings
4. Retriever setup for querying the processed documents
5. Evaluation of the RAG system

Method Details:
- Document Preprocessing: PDF loading and text splitting into chunks
- Text Cleaning: Custom function to clean extracted text
- Vector Store Creation: Gemini embeddings with Chroma for efficient similarity search
- Retriever Setup: Configuration to fetch top-k most relevant chunks
- Encoding Function: Modular function encapsulating the entire PDF-to-vector-store process

Key Features:
- Modular Design: Encoding process encapsulated in reusable functions
- Configurable Chunking: Adjustable chunk size and overlap parameters
- Efficient Retrieval: Chroma for fast similarity search in high-dimensional spaces
- Gemini Integration: Uses Google's Gemini embeddings for text representation
- Evaluation: Built-in performance assessment capabilities

Benefits:
- Scalability: Handles large documents by processing them in chunks
- Flexibility: Easy parameter adjustment for different use cases
- Efficiency: Optimized retrieval with Chroma and Gemini embeddings
- Google Integration: Ready for Google AI ecosystem integration

Usage:
python 01_simple_rag_langchain_google.py --path data/document.pdf --query "What is the main topic?"
"""

import os
import sys
import argparse
import time
from dotenv import load_dotenv

# Add the parent directory to the path since we work with notebooks
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Load environment variables from a .env file
load_dotenv()
# Set the Google API key environment variable (using GEMINI_API_KEY from .env)
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY', '')

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from helper_functions import (EmbeddingProvider,
                              retrieve_context_per_question,
                              replace_t_with_space,
                              get_langchain_embedding_provider,
                              show_context)

from evaluation.evalute_rag import evaluate_rag

from langchain_community.vectorstores import Chroma


class SimpleRAGGemini:
    """
    A class to handle the Simple RAG process for document chunking and query retrieval using Gemini embeddings.

    This class encapsulates the entire RAG pipeline from document preprocessing to query retrieval,
    providing a clean interface for encoding PDF documents and performing similarity-based queries with Gemini.
    """

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2):
        """
        Initializes the SimpleRAGGemini by encoding the PDF document and creating the retriever.

        This method performs the core document preprocessing pipeline:
        1. PDF loading and text extraction using PyPDFLoader
        2. Text splitting into manageable chunks with specified overlap
        3. Text cleaning to handle formatting issues
        4. Vector embedding creation using Gemini embeddings
        5. FAISS vector store creation for efficient similarity search
        6. Retriever configuration for query processing

        Args:
            path (str): Path to the PDF file to encode.
            chunk_size (int): Size of each text chunk in characters (default: 1000).
                         Larger chunks preserve more context but may reduce retrieval precision.
            chunk_overlap (int): Number of characters to overlap between consecutive chunks (default: 200).
                         Helps maintain context continuity across chunk boundaries.
            n_retrieved (int): Number of most relevant chunks to retrieve for each query (default: 2).
                         More chunks provide richer context but increase processing time.
        """
        print("\n--- Initializing Simple RAG Retriever with Gemini ---")

        # Document Preprocessing: Encode PDF into vector store using Gemini embeddings
        # This involves loading, chunking, cleaning, and embedding the document content
        start_time = time.time()
        self.vector_store = encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"Chunking Time: {self.time_records['Chunking']:.2f} seconds")

        # Retriever Setup: Create retriever configured to fetch top-k most relevant chunks
        # The retriever uses Chroma for efficient similarity search in the vector space
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        Executes the retrieval phase of RAG for a given query using Gemini embeddings.

        This method performs the core retrieval operation:
        1. Embeds the input query using Gemini embedding model
        2. Performs similarity search in the FAISS vector space
        3. Retrieves the top-k most relevant document chunks
        4. Displays the retrieved context for user inspection

        Args:
            query (str): The user's question or query to retrieve relevant context for.
                        The query will be embedded and matched against document chunks.

        Returns:
            None: Results are displayed directly to console for immediate user feedback.
        """
        # Retrieval Phase: Perform similarity search to find relevant document chunks
        # The query is embedded using Gemini and compared against all document chunks in vector space
        start_time = time.time()
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")

        # Display retrieved context: Show the most relevant chunks found by the retriever
        # This allows users to inspect the quality and relevance of retrieved information
        show_context(context)


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using Gemini embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A Chroma vector store containing the encoded book content.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create vector store with Chroma
    vectorstore = Chroma.from_documents(cleaned_texts, embeddings)

    return vectorstore


def validate_args(args):
    """
    Validates command-line arguments to ensure they meet the requirements for RAG processing.

    This function performs input validation to prevent runtime errors and ensure
    that the chunking and retrieval parameters are within acceptable ranges.

    Args:
        args: Parsed command-line arguments from argparse.

    Returns:
        args: Validated arguments (returned unchanged if validation passes).

    Raises:
        ValueError: If any argument fails validation criteria.
    """
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved must be a positive integer.")
    return args


def parse_args():
    """
    Parses and validates command-line arguments for the Simple RAG Gemini system.

    This function sets up the argument parser with all configurable parameters
    for the RAG pipeline, providing sensible defaults while allowing full customization.

    Returns:
        args: Validated command-line arguments ready for use.
    """
    parser = argparse.ArgumentParser(
        description="Encode a PDF document and test a simple RAG system with Gemini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_simple_rag_langchain_google.py --path data/document.pdf --query "What is machine learning?"
  python 01_simple_rag_langchain_google.py --chunk_size 500 --chunk_overlap 100 --n_retrieved 3 --evaluate
        """
    )

    parser.add_argument("--path", type=str, default="data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to encode and process.")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of each text chunk in characters (default: 1000). "
                             "Larger values preserve more context but may reduce precision.")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap between consecutive chunks in characters (default: 200). "
                             "Helps maintain context continuity across chunk boundaries.")
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="Number of most relevant chunks to retrieve per query (default: 2). "
                             "More chunks provide richer context but increase processing time.")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="Query to test the retriever with (default: climate change question).")
    parser.add_argument("--evaluate", action="store_true",
                        help="Enable performance evaluation of the RAG system (default: False). "
                             "When enabled, runs comprehensive evaluation metrics.")

    # Parse and validate arguments
    return validate_args(parser.parse_args())


def main(args):
    """
    Main execution function that orchestrates the complete RAG pipeline with Gemini.

    This function serves as the entry point that:
    1. Initializes the SimpleRAGGemini system with user-specified parameters
    2. Executes the retrieval query
    3. Optionally evaluates system performance

    Args:
        args: Parsed and validated command-line arguments.
    """
    # Initialize the RAG system with document encoding and retriever setup
    simple_rag = SimpleRAGGemini(
        path=args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved
    )

    # Execute the retrieval query and display results
    simple_rag.run(args.query)

    # Optional evaluation phase: Assess retrieval quality and performance metrics
    if args.evaluate:
        print("\n--- Evaluating RAG System Performance ---")
        evaluate_rag(simple_rag.chunks_query_retriever)


if __name__ == '__main__':
    # Call the main function with parsed arguments
    main(parse_args())


