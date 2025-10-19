"""
RAG Evaluation Script

This script evaluates the performance of a Retrieval-Augmented Generation (RAG) system
using various metrics from the deepeval library.

Dependencies:
- deepeval
- langchain_openai
- json

Custom modules:
- helper_functions (for RAG-specific operations)
"""

import json
from typing import List, Tuple, Dict, Any

from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 09/15/24 kimmeyh Added path where helper functions is located to the path
# Add the parent directory to the path since we work with notebooks
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from helper_functions import (
    create_question_answer_from_context_chain,
    answer_question_from_context,
    retrieve_context_per_question
)

def create_deep_eval_test_cases(
    questions: List[str],
    gt_answers: List[str],
    generated_answers: List[str],
    retrieved_documents: List[str]
) -> List[LLMTestCase]:
    """
    Create a list of LLMTestCase objects for evaluation.

    Args:
        questions (List[str]): List of input questions.
        gt_answers (List[str]): List of ground truth answers.
        generated_answers (List[str]): List of generated answers.
        retrieved_documents (List[str]): List of retrieved documents.

    Returns:
        List[LLMTestCase]: List of LLMTestCase objects.
    """
    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer,
            retrieval_context=retrieved_document
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]

# Define evaluation metrics with custom LLM support
def create_evaluation_metrics(llm_model="gpt-4-turbo"):
    correctness_metric = GEval(
        name="Correctness",
        model=llm_model,
        evaluation_params=[
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        evaluation_steps=[
            "Determine whether the actual output is factually correct based on the expected output."
        ],
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model=llm_model,
        include_reason=False
    )

    relevance_metric = ContextualRelevancyMetric(
        threshold=1,
        model=llm_model,
        include_reason=True
    )

    return correctness_metric, faithfulness_metric, relevance_metric

# Default metrics using GPT-4
correctness_metric, faithfulness_metric, relevance_metric = create_evaluation_metrics()

def evaluate_rag(retriever, llm=None, num_questions: int = 5) -> Dict[str, Any]:
    """
    Evaluates a RAG system using predefined test questions and metrics.

    Args:
        retriever: The retriever component to evaluate
        llm: Language model to use for evaluation (optional, defaults to GPT-4)
        num_questions: Number of test questions to generate

    Returns:
        Dict containing evaluation metrics
    """

    # Initialize LLM and metrics based on provider
    if llm is None:
        llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo-preview")
        # Use default GPT-4 metrics
        corr_metric, faith_metric, rel_metric = correctness_metric, faithfulness_metric, relevance_metric
    else:
        # Check if it's a Google model
        if hasattr(llm, 'model_name') and 'gemini' in llm.model_name.lower():
            # For Google models, create simplified evaluation without complex metrics
            print("🔄 Usando valutazione semplificata per modello Google Gemini")
            return evaluate_rag_simple(retriever, llm, num_questions)
        else:
            # For other models, try to use them with DeepEval
            try:
                corr_metric, faith_metric, rel_metric = create_evaluation_metrics(llm.model_name)
            except:
                # Fallback to GPT-4 metrics
                corr_metric, faith_metric, rel_metric = correctness_metric, faithfulness_metric, relevance_metric
    
    # Create evaluation prompt
    eval_prompt = PromptTemplate.from_template("""
    Evaluate the following retrieval results for the question.
    
    Question: {question}
    Retrieved Context: {context}
    
    Rate on a scale of 1-5 (5 being best) for:
    1. Relevance: How relevant is the retrieved information to the question?
    2. Completeness: Does the context contain all necessary information?
    3. Conciseness: Is the retrieved context focused and free of irrelevant information?
    
    Provide ratings in JSON format:
    """)
    
    # Create evaluation chain
    eval_chain = (
        eval_prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Generate test questions
    question_gen_prompt = PromptTemplate.from_template(
        "Generate {num_questions} diverse test questions about climate change:"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()
    
    questions = question_chain.invoke({"num_questions": num_questions}).split("\n")
    
    # Evaluate each question
    results = []
    for question in questions:
        # Get retrieval results
        context = retriever.invoke(question)
        context_text = "\n".join([doc.page_content for doc in context])
        
        # Evaluate results
        eval_result = eval_chain.invoke({
            "question": question,
            "context": context_text
        })
        results.append(eval_result)
    
    return {
        "questions": questions,
        "results": results,
        "average_scores": calculate_average_scores(results)
    }

def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    """Calculate average scores across all evaluation results."""
    # Implementation depends on the exact format of your results
    pass


def evaluate_rag_simple(retriever, llm, num_questions: int = 3) -> Dict[str, Any]:
    """
    Simplified RAG evaluation using the provided LLM directly.

    Args:
        retriever: The retriever component to evaluate
        llm: Language model to use for evaluation
        num_questions: Number of test questions to generate

    Returns:
        Dict containing basic evaluation metrics
    """

    # Generate simple test questions using the provided LLM
    question_prompt = f"Generate {num_questions} simple questions about climate change:"
    try:
        questions_response = llm.invoke(question_prompt)
        questions = str(questions_response.content).split('\n')[:num_questions]
        questions = [q.strip() for q in questions if q.strip() and not q.startswith(('1.', '2.', '3.', '-', '*'))]
    except:
        # Fallback questions if generation fails
        questions = [
            "What is climate change?",
            "What causes climate change?",
            "How can we reduce climate change?"
        ][:num_questions]

    results = []
    for question in questions:
        try:
            # Get retrieval results
            context_docs = retriever.invoke(question)
            context_text = "\n".join([doc.page_content for doc in context_docs])

            # Simple evaluation using the LLM
            eval_prompt = f"""
            Evaluate this RAG response for the question: "{question}"

            Retrieved context: {context_text[:1000]}...

            Rate on a scale of 1-5:
            1. Relevance (how well does the context answer the question?)
            2. Completeness (does it contain enough information?)

            Return only the ratings as numbers.
            """

            eval_response = llm.invoke(eval_prompt)
            results.append({
                "question": question,
                "context_length": len(context_text),
                "evaluation": str(eval_response.content)
            })

        except Exception as e:
            results.append({
                "question": question,
                "error": str(e)
            })

    return {
        "evaluation_type": "simple_llm_based",
        "llm_model": getattr(llm, 'model_name', 'unknown'),
        "questions_evaluated": len(results),
        "results": results,
        "summary": f"Evaluated {len(results)} questions using {getattr(llm, 'model_name', 'unknown')} model"
    }


if __name__ == "__main__":
    # Add any necessary setup or configuration here
    # Example: evaluate_rag(your_chunks_query_retriever_function)
    pass
