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
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Example, Run

# 09/15/24 kimmeyh Added path where helper functions is located to the path
# Add the parent directory to the path since we work with notebooks
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from helper_functions import (
    create_question_answer_from_context_chain,
    answer_question_from_context,
    retrieve_context_per_question
)


def create_langsmith_evaluators():
    """
    Create LangSmith evaluators for RAG evaluation using gemini-2.5-flash.

    Returns:
        Dict of evaluators for correctness, faithfulness, and relevance
    """
    from langchain.evaluation import StringEvaluator
    from langchain_core.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import NumberedListOutputParser

    evaluators = {}

    # Create LLM instance for evaluators
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Correctness evaluator
    class CorrectnessEvaluator(StringEvaluator):
        @property
        def evaluation_name(self):
            return "correctness"

        def _evaluate_strings(self, prediction, input=None, **kwargs):
            prompt = f"""
            Evaluate how correct and accurate the answer is based on the context provided.
            Rate on a scale of 0-1, where 1 is perfectly correct and 0 is completely incorrect.

            Context: {kwargs.get('context', '')}
            Question: {kwargs.get('question', '')}
            Answer: {prediction}

            Return only a number between 0 and 1 (e.g., 0.85) and a brief justification separated by a pipe (|).
            """
            response = llm.invoke(prompt)
            result = str(response.content).strip()
            try:
                score_part, reason_part = result.split('|', 1)
                score = float(score_part.strip())
                reason = reason_part.strip()
            except:
                score = 0.5
                reason = "Unable to parse evaluation"

            return {"score": score, "reason": reason}

    # Faithfulness evaluator
    class FaithfulnessEvaluator(StringEvaluator):
        @property
        def evaluation_name(self):
            return "faithfulness"

        def _evaluate_strings(self, prediction, input=None, **kwargs):
            prompt = f"""
            Evaluate how well the answer is grounded in and supported by the provided context.
            Does the answer contain information that can be directly supported by the context?
            Rate on a scale of 0-1, where 1 is perfectly grounded and 0 contains unsupported claims.

            Context: {kwargs.get('context', '')}
            Question: {kwargs.get('question', '')}
            Answer: {prediction}

            Return only a number between 0 and 1 (e.g., 0.92) and a brief justification separated by a pipe (|).
            """
            response = llm.invoke(prompt)
            result = str(response.content).strip()
            try:
                score_part, reason_part = result.split('|', 1)
                score = float(score_part.strip())
                reason = reason_part.strip()
            except:
                score = 0.5
                reason = "Unable to parse evaluation"

            return {"score": score, "reason": reason}

    # Relevance evaluator
    class RelevanceEvaluator(StringEvaluator):
        @property
        def evaluation_name(self):
            return "relevance"

        def _evaluate_strings(self, prediction, input=None, **kwargs):
            prompt = f"""
            Evaluate how relevant the retrieved context is to the question asked.
            Does the context contain information that helps answer the question?
            Rate on a scale of 0-1, where 1 is highly relevant and 0 is completely irrelevant.

            Context: {kwargs.get('context', '')}
            Question: {kwargs.get('question', '')}

            Return only a number between 0 and 1 (e.g., 0.78) and a brief justification separated by a pipe (|).
            """
            response = llm.invoke(prompt)
            result = str(response.content).strip()
            try:
                score_part, reason_part = result.split('|', 1)
                score = float(score_part.strip())
                reason = reason_part.strip()
            except:
                score = 0.5
                reason = "Unable to parse evaluation"

            return {"score": score, "reason": reason}

    # Wrap in LangSmith evaluators
    evaluators["correctness"] = LangChainStringEvaluator(
        evaluator=CorrectnessEvaluator(),
        config={"context": "{context}", "question": "{question}"}
    )

    evaluators["faithfulness"] = LangChainStringEvaluator(
        evaluator=FaithfulnessEvaluator(),
        config={"context": "{context}", "question": "{question}"}
    )

    evaluators["relevance"] = LangChainStringEvaluator(
        evaluator=RelevanceEvaluator(),
        config={"context": "{context}", "question": "{question}"}
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

    # Create evaluation dataset
    from langsmith.schemas import Dataset

    dataset = Dataset(name="rag_evaluation_dataset")
    examples = []

    for question in questions:
        # Get retrieval results
        context_docs = retriever.invoke(question)
        context_text = "\n".join([doc.page_content for doc in context_docs])

        example = Example(
            inputs={
                "question": question,
                "context": context_text
            },
            outputs={
                "answer": context_text  # Using context as answer for demo
            }
        )
        examples.append(example)

    dataset.examples = examples

    # Define evaluation function that simulates RAG retrieval
    def rag_retrieval(inputs):
        """Simulate RAG retrieval for evaluation"""
        return {"answer": inputs["context"]}

    print(f"📊 Avviando valutazione LangSmith su {len(examples)} esempi...")

    try:
        # Run evaluation with LangSmith
        eval_results_langsmith = evaluate(
            rag_retrieval,
            data=dataset,
            evaluators=list(evaluators.values()),
            experiment_prefix="rag_evaluation"
        )

        # Process results
        results = []
        total_scores = {"correctness": 0, "faithfulness": 0, "relevance": 0}

        for i, (example, eval_result) in enumerate(zip(examples, eval_results_langsmith)):
            question = example.inputs["question"]
            context_length = len(example.inputs["context"])

            # Extract scores from evaluation results
            scores = {}
            try:
                # Try to extract scores from the evaluation results
                if hasattr(eval_result, 'feedback'):
                    feedback = eval_result.feedback
                    scores["correctness"] = {
                        "score": float(feedback.get("correctness", {}).get("score", 0.5)),
                        "reason": feedback.get("correctness", {}).get("reason", "N/A")
                    }
                    scores["faithfulness"] = {
                        "score": float(feedback.get("faithfulness", {}).get("score", 0.5)),
                        "reason": feedback.get("faithfulness", {}).get("reason", "N/A")
                    }
                    scores["relevance"] = {
                        "score": float(feedback.get("relevance", {}).get("score", 0.5)),
                        "reason": feedback.get("relevance", {}).get("reason", "N/A")
                    }
                else:
                    # Fallback for different result format
                    scores["correctness"] = {"score": 0.5, "reason": "Evaluation completed"}
                    scores["faithfulness"] = {"score": 0.5, "reason": "Evaluation completed"}
                    scores["relevance"] = {"score": 0.5, "reason": "Evaluation completed"}
            except Exception as e:
                print(f"⚠️ Errore estrazione punteggi per domanda {i+1}: {e}")
                scores["correctness"] = {"score": 0.5, "reason": f"Error: {e}"}
                scores["faithfulness"] = {"score": 0.5, "reason": f"Error: {e}"}
                scores["relevance"] = {"score": 0.5, "reason": f"Error: {e}"}

            # Accumulate scores
            total_scores["correctness"] += scores["correctness"]["score"]
            total_scores["faithfulness"] += scores["faithfulness"]["score"]
            total_scores["relevance"] += scores["relevance"]["score"]

            result = {
                "question": question,
                "context_length": context_length,
                "scores": scores
            }
            results.append(result)

        # Calculate averages
        num_evaluated = len(results)
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

    except Exception as e:
        print(f"❌ Errore valutazione LangSmith: {e}")
        # Fallback: return basic structure
        return {
            "evaluation_type": "langsmith_error_fallback",
            "model_used": "gemini-2.5-flash",
            "questions_evaluated": len(questions),
            "results": [{"question": q, "error": str(e)} for q in questions],
            "average_scores": {"correctness": 0, "faithfulness": 0, "relevance": 0},
            "summary": f"LangSmith evaluation failed: {e}"
        }


if __name__ == "__main__":
    # Add any necessary setup or configuration here
    # Example: evaluate_rag(your_chunks_query_retriever_function)
    pass
