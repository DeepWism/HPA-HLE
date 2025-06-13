"""
Expert Reasoning System

A high-performance reasoning system that evaluates complex questions using
multiple expert LLM agents with different strengths and specializations.

This system routes questions to appropriate processing pipelines, generates
responses from multiple expert models, evaluates the responses, and selects
the best answer based on logical reasoning and accuracy.
"""

import os
import sys
import asyncio
import logging
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from agent.agent import GPTAgent, ClaudeAgent, GeminiAgent, TongyiAgent
from agent.router import QuestionRouter
from agent.evaluator import ResponseEvaluator

from tools.search import WebSearchTool
from tools.code_executor import CodeExecutor

from memory.storage import ResultStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('expert_reasoning.log')
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global semaphore for concurrency control
MAX_CONCURRENCY = int(os.getenv('MAX_CONCURRENCY', '5'))
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


async def process_single_question(index: int, question_data: Dict[str, Any]) -> List[Any]:
    """
    Process a single question using the expert reasoning system.

    Args:
        index: Index of the question in the dataset
        question_data: Dictionary containing question information

    Returns:
        List containing processing results
    """
    async with semaphore:  # Limit concurrency
        question_id = question_data["ID"]
        question = question_data["Question"]
        correct_answer = str(question_data["Answer"])
        answer_type = str(question_data["answer_type"])
        category = str(question_data["category"])

        logger.info(f"Processing question {index} (ID: {question_id})")

        # Initialize agents and tools
        router = QuestionRouter()
        evaluator = ResponseEvaluator()
        gpt_agent = GPTAgent()
        claude_agent = ClaudeAgent()
        gemini_agent = GeminiAgent()
        search_tool = WebSearchTool()
        code_executor = CodeExecutor()

        # Route the question
        strategy = await router.determine_strategy(question)
        needs_search = strategy["classification"]["needs_search"]
        needs_code = "no"
        needs_code = strategy["classification"]["needs_code"]

        # Store full results for detailed analysis
        detailed_result = {
            "question_id": question_id,
            "question": question,
            "correct_answer": correct_answer,
            "answer_type": answer_type,
            "category": category,
            "classification": strategy["classification"],
            "processing_strategy": strategy,
            "responses": {}
        }

        if True:
            # Standard reasoning approach (no search needed)
            prompt = f"""
                <User Question>
                {question}
                </User Question>
                
                Please answer the above user question based on the following principles:

                - Thoroughly comprehend the question and fully understand all its constraints.
                - Perform a detailed, step-by-step analysis and systematically derive the answer.
                - During the reasoning process, be mindful of any implicit logic that the question may not explicitly state, and avoid falling into any traps deliberately embedded in the problem.
                - Rigorously review and verify your answer.
                - Finally, present your definitive, confirmed conclusion.
            
            """

            # Prepare messages for LLM calls
            messages = [{"role": "user", "content": prompt}]

            # Generate code in parallel if needed
            code_result = ""
            exe_result = ""
            if strategy["classification"]["needs_code"]:
                code_execution = await code_executor.solve_with_code(question, str(question_id))
                code_result = code_execution["code"]
                exe_result = code_execution["execution_output"]
                detailed_result["code_execution"] = code_execution

            # Call reasoning experts in parallel
            res_1, res_2, res_3, res_4, res_5 = await asyncio.gather(
                gpt_agent.generate_reasoning_response(
                    messages, model_name="o3-mini", temperature=0.6),
                gpt_agent.generate_reasoning_response(
                    messages, model_name="o3-mini", temperature=0.8),
                gpt_agent.generate_reasoning_response(
                    messages, model_name="o4-mini", temperature=0.6),
                gpt_agent.generate_reasoning_response(
                    messages, model_name="o4-mini", temperature=0.8),
                # claude_agent.generate_reasoning_response(
                #     messages, model_name="claude-3-7-sonnet-20250219", temperature=1),
                gemini_agent.search_and_generate(
                    prompt, "gemini-2.5-pro-preview-03-25", temperature=0.6)
            )

            # Store responses in detailed result
            detailed_result["responses"] = {
                "expert_1": res_1,
                "expert_2": res_2,
                "expert_3": res_3,
                "expert_4": res_4,
                "expert_5": res_5,
                "code": code_result,
                "code_execution": exe_result
            }

            # Verify responses against correct answer
            verification = await evaluator.verify_against_answer(
                [res_1, res_2, res_3, res_4, res_5],
                correct_answer,
                code_result,
                exe_result,
                question
            )

            detailed_result["verification"] = verification

            # Select best response
            selection = await evaluator.select_best_response(
                [res_1, res_2, res_3, res_4, res_5],
                question,
                code_result,
                exe_result
            )

            detailed_result["selection"] = selection

            # Save detailed result
            result_storage = ResultStorage()
            result_storage.save_detailed_result(detailed_result, question_id)

            # Return summarized result
            return [
                question_id,
                question,
                correct_answer,
                answer_type,
                category,
                res_1,
                res_2,
                res_3,
                res_4,
                res_5,
                code_result,
                exe_result,
                verification.get("is_correct", False),
                verification.get("correct_agents", "no"),
                "no",
                needs_code,
                selection.get("best_response", "")
            ]
        else:
            # Search-based approach
            search_prompt = f"""
                <User Question>
                {question}
                </User Question>

                Please answer the question above based on the following principles:
                - Carefully read and analyze the information retrieved. Search in English and answer in English.
                - Ensure that the search keywords are closely aligned with the user's question.
                - Think thoroughly step by step, and derive the answer step by step.
                - Be cautious during the reasoning process to avoid any traps or misleading setups in the question.
                - Review and verify the answer once more.
                - Finally, output the confirmed and final answer conclusion.
            """

            # Perform search
            res_1 = await search_tool.search(search_prompt, temperature=1)

            # Empty placeholders for other experts not used in search mode
            res_2 = ""
            res_3 = ""
            res_4 = ""
            res_5 = ""
            code_result = ""
            exe_result = ""

            # Store response in detailed result
            detailed_result["responses"] = {
                "search_expert": res_1
            }

            # Verify response against correct answer
            verification = await evaluator.verify_against_answer(
                [res_1],
                correct_answer,
                "",
                "",
                question
            )

            detailed_result["verification"] = verification

            # Save detailed result
            result_storage = ResultStorage()
            result_storage.save_detailed_result(detailed_result, question_id)

            # Return summarized result
            return [
                question_id,
                question,
                correct_answer,
                answer_type,
                category,
                res_1,
                res_2,
                res_3,
                res_4,
                res_5,
                code_result,
                exe_result,
                verification.get("is_correct", False),
                verification.get("correct_agents", "no"),
                "yes",
                needs_code,
                "agent_1"
            ]


async def process_questions(file_path: str, sample_size: int = 200) -> None:
    """
    Process multiple questions from a dataset file.

    Args:
        file_path: Path to the Excel file containing questions
        sample_size: Number of questions to sample from the dataset
    """
    try:
        # Load questions from Excel file
        df = pd.read_excel(file_path)
        logger.info(f"Loaded {len(df)} questions from {file_path}")

        # Sample questions if requested
        if sample_size > 0 and sample_size < len(df):
            df_sample = df.sample(
                n=sample_size, random_state=None).reset_index(drop=True)
            logger.info(f"Sampled {sample_size} questions for processing")
        else:
            df_sample = df.reset_index(drop=True)
            logger.info(f"Processing all {len(df)} questions")

        # Create tasks for processing each question
        tasks = [
            process_single_question(index, row.to_dict())
            for index, row in df_sample.iterrows()
        ]

        # Process questions in parallel
        logger.info(
            f"Processing {len(tasks)} questions with max concurrency {MAX_CONCURRENCY}")
        results = await asyncio.gather(*tasks)

        # Save results to Excel
        result_storage = ResultStorage()
        output_file = result_storage.save_results(
            results,
            columns=[
                "ID", "Question", "Answer", "answer_type", "category",
                "agent_1", "agent_2", "agent_3", "agent_4", "agent_5",
                "program_agent", "program_result", "is_correct", "correct_agents", "need_search", "need_code", "best_response"
            ]
        )

        # Load results for analysis
        results_df = result_storage.load_results(os.path.basename(output_file))

        # Analyze results
        analysis = result_storage.analyze_results(results_df)

        # Export analysis
        analysis_file = result_storage.export_analysis(analysis)

        logger.info(f"All questions processed successfully.")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Analysis saved to: {analysis_file}")

        # Print summary statistics
        correct_count = analysis.get("correct_count", 0)
        total_count = analysis.get("total_count", 0)
        accuracy = analysis.get("accuracy", 0)

    except Exception as e:
        logger.error(f"Error processing questions: {str(e)}", exc_info=True)
        raise


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Expert Reasoning System")
    parser.add_argument("--file", type=str, default=r"test_hle.xlsx",
                        help="Path to the Excel file containing questions")
    parser.add_argument("--sample", type=int, default=200,
                        help="Number of questions to sample (0 for all)")
    args = parser.parse_args()

    logger.info("Starting Expert Reasoning System")
    logger.info(f"Processing questions from: {args.file}")

    try:
        asyncio.run(process_questions(args.file, args.sample))
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    # sys.exit(main())
    asyncio.run(process_questions("test_hle.xlsx", 50))
