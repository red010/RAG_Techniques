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
    # Configure DeepEval to use custom model
    from deepeval.models import DeepEvalBaseLLM

    class GeminiModel(DeepEvalBaseLLM):
        def __init__(self, model_name="gemini-2.0-flash"):
            self.model_name = model_name  # Use gemini-2.0-flash as requested

        def load_model(self):
            return self.model_name

        def generate(self, prompt: str) -> str:
            # Import here to avoid circular imports
            from helper_functions import ModelProvider, get_langchain_model_provider
            llm = get_langchain_model_provider(ModelProvider.GOOGLE, model_id=self.model_name, temperature=0)
            response = llm.invoke(prompt)
            return response.content

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

        def get_model_name(self):
            return self.model_name

    # Create Gemini model instance
    gemini_llm = GeminiModel(model_name=llm_model)

    correctness_metric = GEval(
        name="Correctness",
        model=gemini_llm,
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
        model=gemini_llm,
        include_reason=False
    )

    relevance_metric = ContextualRelevancyMetric(
        threshold=1,
        model=gemini_llm,
        include_reason=True
    )

    return correctness_metric, faithfulness_metric, relevance_metric

# Default metrics using GPT-4
correctness_metric, faithfulness_metric, relevance_metric = create_evaluation_metrics()

def evaluate_rag(retriever, llm=None, num_questions: int = 5) -> Dict[str, Any]:
    """
    Evaluates a RAG system using DeepEval metrics with the specified LLM model.

    Args:
        retriever: The retriever component to evaluate
        llm: Language model to use for evaluation (defaults to gemini-2.0-flash)
        num_questions: Number of test questions to generate

    Returns:
        Dict containing evaluation metrics with numerical scores
    """

    # Determine model name for DeepEval
    if llm is None:
        model_name = "gemini-2.0-flash"  # Default to Gemini 2.0 Flash
    else:
        # Extract model name from LangChain LLM object
        if hasattr(llm, 'model_name') and llm.model_name:
            model_name = llm.model_name
        elif hasattr(llm, 'model') and llm.model:
            model_name = llm.model
        else:
            model_name = "gemini-2.0-flash"  # Fallback to Gemini 2.0 Flash as requested

    print(f"🔬 Usando DeepEval con modello: {model_name}")

    # Create metrics with the specified model
    try:
        corr_metric, faith_metric, rel_metric = create_evaluation_metrics(model_name)
    except Exception as e:
        print(f"❌ Errore creazione metriche DeepEval: {e}")
        print("🔄 Fallback a gemini-2.0-flash")
        corr_metric, faith_metric, rel_metric = create_evaluation_metrics("gemini-2.0-flash")

    # Generate test questions using the evaluation LLM
    from deepeval.test_case import LLMTestCase

    # Create test questions using the LLM
    question_prompt = f"""Generate exactly {num_questions} diverse questions about climate change.
Return each question on a separate line, numbered 1 to {num_questions}.
Do not include any other text or explanations.

Example format:
1. What is climate change?
2. What are the main causes of climate change?
3. How can we reduce climate change?
"""
    try:
        # Use the same LLM instance for question generation
        from helper_functions import ModelProvider, get_langchain_model_provider
        if llm is None:
            gen_llm = get_langchain_model_provider(ModelProvider.GOOGLE, model_id=model_name, temperature=0.7)
        else:
            gen_llm = llm

        questions_response = gen_llm.invoke(question_prompt)
        response_text = str(questions_response.content).strip()

        # Extract questions that start with numbers
        lines = response_text.split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, num_questions + 1)):
                # Remove the number prefix
                question = line.split('.', 1)[1].strip()
                if question:
                    questions.append(question)

        # If we didn't get enough questions, use fallback
        if len(questions) < num_questions:
            questions = [
                "What is climate change?",
                "What are the main causes of climate change?",
                "How can we reduce climate change?",
                "What are the effects of climate change?",
                "What is being done to combat climate change?"
            ][:num_questions]

    except Exception as e:
        print(f"⚠️ Errore generazione domande: {e}, uso fallback")
        questions = [
            "What is climate change?",
            "What are the main causes of climate change?",
            "How can we reduce climate change?",
            "What are the effects of climate change?",
            "What is being done to combat climate change?"
        ][:num_questions]

    print(f"📝 Generate {len(questions)} domande di test")

    # Evaluate each question using DeepEval metrics
    results = []
    total_scores = {"correctness": 0, "faithfulness": 0, "relevance": 0}

    for i, question in enumerate(questions, 1):
        print(f"🔍 Valutando domanda {i}/{len(questions)}: {question[:50]}...")

        try:
            # Get retrieval results
            context_docs = retriever.invoke(question)
            context_text = "\n".join([doc.page_content for doc in context_docs])

            # Create test case for DeepEval
            # Note: For this demo, we'll use the context as both input and expected output
            # In a real scenario, you'd have ground truth answers
            test_case = LLMTestCase(
                input=question,
                actual_output=context_text,
                expected_output=context_text,  # Using context as expected for demo
                retrieval_context=[doc.page_content for doc in context_docs]
            )

            # Evaluate using DeepEval metrics
            correctness_result = corr_metric.measure(test_case)
            faithfulness_result = faith_metric.measure(test_case)
            relevance_result = rel_metric.measure(test_case)

            # Handle both object and float results
            def get_score_and_reason(result):
                if hasattr(result, 'score'):
                    return result.score, getattr(result, 'reason', 'N/A')
                else:
                    # Assume it's already a float score
                    return float(result), 'N/A'

            corr_score, corr_reason = get_score_and_reason(correctness_result)
            faith_score, faith_reason = get_score_and_reason(faithfulness_result)
            rel_score, rel_reason = get_score_and_reason(relevance_result)

            # Accumulate scores for averages
            total_scores["correctness"] += corr_score
            total_scores["faithfulness"] += faith_score
            total_scores["relevance"] += rel_score

            result = {
                "question": question,
                "context_length": len(context_text),
                "scores": {
                    "correctness": {
                        "score": corr_score,
                        "reason": corr_reason
                    },
                    "faithfulness": {
                        "score": faith_score,
                        "reason": faith_reason
                    },
                    "relevance": {
                        "score": rel_score,
                        "reason": rel_reason
                    }
                }
            }
            results.append(result)

        except Exception as e:
            print(f"❌ Errore valutazione domanda {i}: {e}")
            results.append({
                "question": question,
                "error": str(e),
                "scores": None
            })

    # Calculate averages
    num_evaluated = len([r for r in results if r.get("scores") is not None])
    if num_evaluated > 0:
        average_scores = {
            "correctness": total_scores["correctness"] / num_evaluated,
            "faithfulness": total_scores["faithfulness"] / num_evaluated,
            "relevance": total_scores["relevance"] / num_evaluated
        }
    else:
        average_scores = {"correctness": 0, "faithfulness": 0, "relevance": 0}

    return {
        "evaluation_type": "deepeval_with_custom_model",
        "model_used": model_name,
        "questions_evaluated": len(results),
        "results": results,
        "average_scores": average_scores,
        "summary": f"Evaluated {len(results)} questions using DeepEval with {model_name}. Average scores: Correctness={average_scores['correctness']:.3f}, Faithfulness={average_scores['faithfulness']:.3f}, Relevance={average_scores['relevance']:.3f}"
    }

def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    """Calculate average scores across all evaluation results."""
    # Implementation depends on the exact format of your results
    pass


if __name__ == "__main__":
    # Add any necessary setup or configuration here
    # Example: evaluate_rag(your_chunks_query_retriever_function)
    pass
