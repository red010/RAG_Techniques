# %% [markdown]
# <!-- ![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=adaptive-retrieval) -->
# 
# 
# 
# # Adaptive Retrieval-Augmented Generation (RAG) System
# 
# ## Overview
# 
# This system implements an advanced Retrieval-Augmented Generation (RAG) approach that adapts its retrieval strategy based on the type of query. By leveraging Language Models (LLMs) at various stages, it aims to provide more accurate, relevant, and context-aware responses to user queries.
# 
# ## Motivation
# 
# Traditional RAG systems often use a one-size-fits-all approach to retrieval, which can be suboptimal for different types of queries. Our adaptive system is motivated by the understanding that different types of questions require different retrieval strategies. For example, a factual query might benefit from precise, focused retrieval, while an analytical query might require a broader, more diverse set of information.
# 
# ## Key Components
# 
# 1. **Query Classifier**: Determines the type of query (Factual, Analytical, Opinion, or Contextual).
# 
# 2. **Adaptive Retrieval Strategies**: Four distinct strategies tailored to different query types:
#    - Factual Strategy
#    - Analytical Strategy
#    - Opinion Strategy
#    - Contextual Strategy
# 
# 3. **LLM Integration**: LLMs are used throughout the process to enhance retrieval and ranking.
# 
# 4. **OpenAI GPT Model**: Generates the final response using the retrieved documents as context.
# 
# ## Method Details
# 
# ### 1. Query Classification
# 
# The system begins by classifying the user's query into one of four categories:
# - Factual: Queries seeking specific, verifiable information.
# - Analytical: Queries requiring comprehensive analysis or explanation.
# - Opinion: Queries about subjective matters or seeking diverse viewpoints.
# - Contextual: Queries that depend on user-specific context.
# 
# ### 2. Adaptive Retrieval Strategies
# 
# Each query type triggers a specific retrieval strategy:
# 
# #### Factual Strategy
# - Enhances the original query using an LLM for better precision.
# - Retrieves documents based on the enhanced query.
# - Uses an LLM to rank documents by relevance.
# 
# #### Analytical Strategy
# - Generates multiple sub-queries using an LLM to cover different aspects of the main query.
# - Retrieves documents for each sub-query.
# - Ensures diversity in the final document selection using an LLM.
# 
# #### Opinion Strategy
# - Identifies different viewpoints on the topic using an LLM.
# - Retrieves documents representing each viewpoint.
# - Uses an LLM to select a diverse range of opinions from the retrieved documents.
# 
# #### Contextual Strategy
# - Incorporates user-specific context into the query using an LLM.
# - Performs retrieval based on the contextualized query.
# - Ranks documents considering both relevance and user context.
# 
# ### 3. LLM-Enhanced Ranking
# 
# After retrieval, each strategy uses an LLM to perform a final ranking of the documents. This step ensures that the most relevant and appropriate documents are selected for the next stage.
# 
# ### 4. Response Generation
# 
# The final set of retrieved documents is passed to an OpenAI GPT model, which generates a response based on the query and the provided context.
# 
# ## Benefits of This Approach
# 
# 1. **Improved Accuracy**: By tailoring the retrieval strategy to the query type, the system can provide more accurate and relevant information.
# 
# 2. **Flexibility**: The system adapts to different types of queries, handling a wide range of user needs.
# 
# 3. **Context-Awareness**: Especially for contextual queries, the system can incorporate user-specific information for more personalized responses.
# 
# 4. **Diverse Perspectives**: For opinion-based queries, the system actively seeks out and presents multiple viewpoints.
# 
# 5. **Comprehensive Analysis**: The analytical strategy ensures a thorough exploration of complex topics.
# 
# ## Conclusion
# 
# This adaptive RAG system represents a significant advancement over traditional RAG approaches. By dynamically adjusting its retrieval strategy and leveraging LLMs throughout the process, it aims to provide more accurate, relevant, and nuanced responses to a wide variety of user queries.

# %% [markdown]
# <div style="text-align: center;">
# 
# <img src="../images/adaptive_retrieval.svg" alt="adaptive retrieval" style="width:100%; height:auto;">
# </div>

# %% [markdown]
# # Package Installation and Imports
# 
# The cell below installs all necessary packages required to run this notebook.
# 

# %%
# Install required packages
!pip install faiss-cpu langchain langchain-openai python-dotenv

# %%
# Clone the repository to access helper functions and evaluation modules
!git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
# If you need to run with the latest data
# !cp -r RAG_TECHNIQUES/data .

# %%
import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

from langchain_core.retrievers import BaseRetriever
from typing import Dict, Any
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field


# Original path append replaced for Colab compatibility
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# %% [markdown]
# ### Define the query classifer class

# %%
class categories_options(BaseModel):
        category: str = Field(description="The category of the query, the options are: Factual, Analytical, Opinion, or Contextual", example="Factual")


class QueryClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="Classify the following query into one of these categories: Factual, Analytical, Opinion, or Contextual.\nQuery: {query}\nCategory:"
        )
        self.chain = self.prompt | self.llm.with_structured_output(categories_options)


    def classify(self, query):
        print("clasiffying query")
        return self.chain.invoke(query).category

# %% [markdown]
# ### Define the Base Retriever class, such that the complex ones will inherit from it

# %%
class BaseRetrievalStrategy:
    def __init__(self, texts):
        self.embeddings = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
        self.documents = text_splitter.create_documents(texts)
        self.db = FAISS.from_documents(self.documents, self.embeddings)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)


    def retrieve(self, query, k=4):
        return self.db.similarity_search(query, k=k)

# %% [markdown]
# ### Define Factual retriever strategy

# %%
class relevant_score(BaseModel):
        score: float = Field(description="The relevance score of the document to the query", example=8.0)

class FactualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4):
        print("retrieving factual")
        # Use LLM to enhance the query
        enhanced_query_prompt = PromptTemplate(
            input_variables=["query"],
            template="Enhance this factual query for better information retrieval: {query}"
        )
        query_chain = enhanced_query_prompt | self.llm
        enhanced_query = query_chain.invoke(query).content
        print(f'enhande query: {enhanced_query}')

        # Retrieve documents using the enhanced query
        docs = self.db.similarity_search(enhanced_query, k=k*2)

        # Use LLM to rank the relevance of retrieved documents
        ranking_prompt = PromptTemplate(
            input_variables=["query", "doc"],
            template="On a scale of 1-10, how relevant is this document to the query: '{query}'?\nDocument: {doc}\nRelevance score:"
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(relevant_score)

        ranked_docs = []
        print("ranking docs")
        for doc in docs:
            input_data = {"query": enhanced_query, "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))

        # Sort by relevance score and return top k
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:k]]

# %% [markdown]
# ### Define Analytical reriever strategy

# %%
class SelectedIndices(BaseModel):
    indices: List[int] = Field(description="Indices of selected documents", example=[0, 1, 2, 3])

class SubQueries(BaseModel):
    sub_queries: List[str] = Field(description="List of sub-queries for comprehensive analysis", example=["What is the population of New York?", "What is the GDP of New York?"])

class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4):
        print("retrieving analytical")
        # Use LLM to generate sub-queries for comprehensive analysis
        sub_queries_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Generate {k} sub-questions for: {query}"
        )

        llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        sub_queries_chain = sub_queries_prompt | llm.with_structured_output(SubQueries)

        input_data = {"query": query, "k": k}
        sub_queries = sub_queries_chain.invoke(input_data).sub_queries
        print(f'sub queries for comprehensive analysis: {sub_queries}')

        all_docs = []
        for sub_query in sub_queries:
            all_docs.extend(self.db.similarity_search(sub_query, k=2))

        # Use LLM to ensure diversity and relevance
        diversity_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="""Select the most diverse and relevant set of {k} documents for the query: '{query}'\nDocuments: {docs}\n
            Return only the indices of selected documents as a list of integers."""
        )
        diversity_chain = diversity_prompt | self.llm.with_structured_output(SelectedIndices)
        docs_text = "\n".join([f"{i}: {doc.page_content[:50]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices_result = diversity_chain.invoke(input_data).indices
        print(f'selected diverse and relevant documents')
        
        return [all_docs[i] for i in selected_indices_result if i < len(all_docs)]

# %% [markdown]
# ### Define Opinion retriever strategy

# %%
class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=3):
        print("retrieving opinion")
        # Use LLM to identify potential viewpoints
        viewpoints_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Identify {k} distinct viewpoints or perspectives on the topic: {query}"
        )
        viewpoints_chain = viewpoints_prompt | self.llm
        input_data = {"query": query, "k": k}
        viewpoints = viewpoints_chain.invoke(input_data).content.split('\n')
        print(f'viewpoints: {viewpoints}')

        all_docs = []
        for viewpoint in viewpoints:
            all_docs.extend(self.db.similarity_search(f"{query} {viewpoint}", k=2))

        # Use LLM to classify and select diverse opinions
        opinion_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="Classify these documents into distinct opinions on '{query}' and select the {k} most representative and diverse viewpoints:\nDocuments: {docs}\nSelected indices:"
        )
        opinion_chain = opinion_prompt | self.llm.with_structured_output(SelectedIndices)
        
        docs_text = "\n".join([f"{i}: {doc.page_content[:100]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices = opinion_chain.invoke(input_data).indices
        print(f'selected diverse and relevant documents')
        
        return [all_docs[int(i)] for i in selected_indices.split() if i.isdigit() and int(i) < len(all_docs)]

# %% [markdown]
# ### Define Contextual retriever strategy

# %%
class ContextualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4, user_context=None):
        print("retrieving contextual")
        # Use LLM to incorporate user context into the query
        context_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Given the user context: {context}\nReformulate the query to best address the user's needs: {query}"
        )
        context_chain = context_prompt | self.llm
        input_data = {"query": query, "context": user_context or "No specific context provided"}
        contextualized_query = context_chain.invoke(input_data).content
        print(f'contextualized query: {contextualized_query}')

        # Retrieve documents using the contextualized query
        docs = self.db.similarity_search(contextualized_query, k=k*2)

        # Use LLM to rank the relevance of retrieved documents considering the user context
        ranking_prompt = PromptTemplate(
            input_variables=["query", "context", "doc"],
            template="Given the query: '{query}' and user context: '{context}', rate the relevance of this document on a scale of 1-10:\nDocument: {doc}\nRelevance score:"
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(relevant_score)
        print("ranking docs")

        ranked_docs = []
        for doc in docs:
            input_data = {"query": contextualized_query, "context": user_context or "No specific context provided", "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))


        # Sort by relevance score and return top k
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_docs[:k]]

# %% [markdown]
# ### Define the Adapive retriever class

# %%
class AdaptiveRetriever:
    def __init__(self, texts: List[str]):
        self.classifier = QueryClassifier()
        self.strategies = {
            "Factual": FactualRetrievalStrategy(texts),
            "Analytical": AnalyticalRetrievalStrategy(texts),
            "Opinion": OpinionRetrievalStrategy(texts),
            "Contextual": ContextualRetrievalStrategy(texts)
        }

    def get_relevant_documents(self, query: str) -> List[Document]:
        category = self.classifier.classify(query)
        strategy = self.strategies[category]
        return strategy.retrieve(query)

# %% [markdown]
# ### Define aditional retriever that inherits from langchain BaseRetriever 

# %%
class PydanticAdaptiveRetriever(BaseRetriever):
    adaptive_retriever: AdaptiveRetriever = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.adaptive_retriever.get_relevant_documents(query)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

# %% [markdown]
# ### Define the Adaptive RAG class

# %%
class AdaptiveRAG:
    def __init__(self, texts: List[str]):
        adaptive_retriever = AdaptiveRetriever(texts)
        self.retriever = PydanticAdaptiveRetriever(adaptive_retriever=adaptive_retriever)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        
        # Create a custom prompt
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Create the LLM chain
        self.llm_chain = prompt | self.llm
        
      

    def answer(self, query: str) -> str:
        docs = self.retriever.get_relevant_documents(query)
        input_data = {"context": "\n".join([doc.page_content for doc in docs]), "question": query}
        return self.llm_chain.invoke(input_data)

# %% [markdown]
# ### Demonstrate use of this model

# %%
# Usage
texts = [
    "The Earth is the third planet from the Sun and the only astronomical object known to harbor life."
    ]
rag_system = AdaptiveRAG(texts)

# %% [markdown]
# ### Showcase the four different types of queries

# %%
factual_result = rag_system.answer("What is the distance between the Earth and the Sun?").content
print(f"Answer: {factual_result}")

analytical_result = rag_system.answer("How does the Earth's distance from the Sun affect its climate?").content
print(f"Answer: {analytical_result}")

opinion_result = rag_system.answer("What are the different theories about the origin of life on Earth?").content
print(f"Answer: {opinion_result}")

contextual_result = rag_system.answer("How does the Earth's position in the Solar System influence its habitability?").content
print(f"Answer: {contextual_result}")

# %% [markdown]
# ![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=all-rag-techniques--adaptive-retrieval)


