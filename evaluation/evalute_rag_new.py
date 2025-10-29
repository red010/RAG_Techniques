"""
RAG Evaluation Script

This script evaluates the performance of a Retrieval-Augmented Generation (RAG) system
using LangSmith evaluators with gemini-2.5-flash.

Dependencies:
- langsmith
- langchain_core
- os, dotenv

Custom modules:
- helper_functions (for RAG-specific operations)
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set LangSmith API key
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

# Import LangSmith evaluators
from langsmith.evaluation import evaluate, LangSmithEvaluator
from langsmith.schemas import Example, Run

# 09/15/24 kimmeyh Added path where helper functions is located to the path
# Add the parent directory to the path since we work with notebooks
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Note: ModelProvider and get_langchain_model_provider are imported locally where needed


def create_langsmith_evaluators():
    """
    Create LangSmith evaluators for RAG evaluation using gemini-2.5-flash.

    Returns:
        Dict of evaluators for correctness, faithfulness, and relevance
    """
    evaluators = {}

    # Correctness evaluator
    evaluators["correctness"] = LangSmithEvaluator(
        evaluator_type="cot_qa",
        model="gemini-2.5-flash",
        prompt="""
        Evaluate how correct and accurate the answer is based on the context provided.
        Rate on a scale of 0-1, where 1 is perfectly correct and 0 is completely incorrect.

        Context: {context}
        Question: {question}
        Answer: {answer}

        Provide a score between 0 and 1, and a brief justification.
        """,
        feedback_key="correctness_score"
    )

    # Faithfulness/Groundedness evaluator
    evaluators["faithfulness"] = LangSmithEvaluator(
        evaluator_type="cot_qa",
        model="gemini-2.5-flash",
        prompt="""
        Evaluate how well the answer is grounded in and supported by the provided context.
        Does the answer contain information that can be directly supported by the context?
        Rate on a scale of 0-1, where 1 is perfectly grounded and 0 contains unsupported claims.

        Context: {context}
        Question: {question}
        Answer: {answer}

        Provide a score between 0 and 1, and a brief justification.
        """,
        feedback_key="faithfulness_score"
    )

    # Relevance evaluator
    evaluators["relevance"] = LangSmithEvaluator(
        evaluator_type="cot_qa",
        model="gemini-2.5-flash",
        prompt="""
        Evaluate how relevant the retrieved context is to the question asked.
        Does the context contain information that helps answer the question?
        Rate on a scale of 0-1, where 1 is highly relevant and 0 is completely irrelevant.

        Context: {context}
        Question: {question}

        Provide a score between 0 and 1, and a brief justification.
        """,
        feedback_key="relevance_score"
    )

    return evaluators


def evaluate_rag(retriever, llm=None, num_questions: int = 3) -> Dict[str, Any]:
    """
    Evaluates a RAG system using LangSmith evaluators with gemini-2.5-flash.

    Args:
        retriever: The retriever component to evaluate
        llm: Language model to use for evaluation (defaults to gemini-2.5-flash)
        num_questions: Number of test questions to generate

    Returns:
        Dict containing evaluation metrics with numerical scores
    """

    print(f"🔬 Usando LangSmith evaluators con gemini-2.5-flash")

    # Create evaluators
    evaluators = create_langsmith_evaluators()

    # Generate test questions using the LLM
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
            gen_llm = get_langchain_model_provider(ModelProvider.GOOGLE, model_id="gemini-2.5-flash", temperature=0.7)
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
            "How can we reduce climate change?"
        ][:num_questions]

    print(f"📝 Generate {len(questions)} domande di test")

    # Evaluate each question using LangSmith evaluators
    results = []
    total_scores = {"correctness": 0, "faithfulness": 0, "relevance": 0}

    for i, question in enumerate(questions, 1):
        print(f"🔍 Valutando domanda {i}/{len(questions)}: {question[:50]}...")

        try:
            # Get retrieval results
            context_docs = retriever.invoke(question)
            context_text = "\n".join([doc.page_content for doc in context_docs])

            # Create example for LangSmith evaluation
            example = Example(
                inputs={"question": question, "context": context_text},
                outputs={"answer": context_text}  # Using context as answer for demo
            )

            # Evaluate using LangSmith evaluators
            eval_results = {}

            # Correctness evaluation
            try:
                correctness_result = evaluators["correctness"].evaluate_run(
                    run=Run(inputs={"question": question, "context": context_text, "answer": context_text}),
                    example=example
                )
                eval_results["correctness"] = {
                    "score": float(correctness_result.get("correctness_score", {}).get("score", 0)),
                    "reason": correctness_result.get("correctness_score", {}).get("reasoning", "N/A")
                }
            except Exception as e:
                print(f"⚠️ Errore valutazione correttezza: {e}")
                eval_results["correctness"] = {"score": 0, "reason": f"Error: {e}"}

            # Faithfulness evaluation
            try:
                faithfulness_result = evaluators["faithfulness"].evaluate_run(
                    run=Run(inputs={"question": question, "context": context_text, "answer": context_text}),
                    example=example
                )
                eval_results["faithfulness"] = {
                    "score": float(faithfulness_result.get("faithfulness_score", {}).get("score", 0)),
                    "reason": faithfulness_result.get("faithfulness_score", {}).get("reasoning", "N/A")
                }
            except Exception as e:
                print(f"⚠️ Errore valutazione fedeltà: {e}")
                eval_results["faithfulness"] = {"score": 0, "reason": f"Error: {e}"}

            # Relevance evaluation (using context relevance to question)
            try:
                relevance_result = evaluators["relevance"].evaluate_run(
                    run=Run(inputs={"question": question, "context": context_text}),
                    example=example
                )
                eval_results["relevance"] = {
                    "score": float(relevance_result.get("relevance_score", {}).get("score", 0)),
                    "reason": relevance_result.get("relevance_score", {}).get("reasoning", "N/A")
                }
            except Exception as e:
                print(f"⚠️ Errore valutazione rilevanza: {e}")
                eval_results["relevance"] = {"score": 0, "reason": f"Error: {e}"}

            # Accumulate scores for averages
            total_scores["correctness"] += eval_results["correctness"]["score"]
            total_scores["faithfulness"] += eval_results["faithfulness"]["score"]
            total_scores["relevance"] += eval_results["relevance"]["score"]

            result = {
                "question": question,
                "context_length": len(context_text),
                "scores": eval_results
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
        "evaluation_type": "langsmith_with_gemini_2_5_flash",
        "model_used": "gemini-2.5-flash",
        "questions_evaluated": len(results),
        "results": results,
        "average_scores": average_scores,
        "summary": f"Evaluated {len(results)} questions using LangSmith with gemini-2.5-flash. Average scores: Correctness={average_scores['correctness']:.3f}, Faithfulness={average_scores['faithfulness']:.3f}, Relevance={average_scores['relevance']:.3f}"
    }


if __name__ == "__main__":
    # Add any necessary setup or configuration here
    # Example: evaluate_rag(your_chunks_query_retriever_function)
    pass
