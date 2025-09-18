#!/usr/bin/env python3
"""
Crowd Pruning and Knee Point Analysis Script

This script implements the crowd_pruning.ipynb notebook functionality as a standalone script.
It performs the following operations:
1. Load initial prompts and create a priority queue
2. Perform crowding using LLM to group similar prompts
3. Find knee point for optimal prompt selection
4. Evaluate final performance on all test centers

Usage:
    python evaluate_by_crowding_and_knee.py [--few_shot N] [--provider PROVIDER]
"""

import argparse
import ast
import re
import os
from typing import List
import numpy as np
import torch

import util


# --- Configuration ---
CROWDING_INTERVAL = 10         # perform crowding every X iterations
CROWDING_ITERATIONS = 3        # number of crowding passes
NUMBER_OF_PROMPTS_TO_GROUP = 30
MAX_RETRIES = 5


class CrowdingManager:
    """
    Encapsulates all crowding-related logic: grouping duplicate prompts via LLM
    and pruning the priority queue accordingly.
    """

    def __init__(self,
                 client,
                 interval: int = CROWDING_INTERVAL,
                 iterations: int = CROWDING_ITERATIONS,
                 group_size: int = NUMBER_OF_PROMPTS_TO_GROUP,
                 max_retries: int = MAX_RETRIES):
        self.client = client
        self.interval = interval
        self.iterations = iterations
        self.group_size = group_size
        self.max_retries = max_retries

        self.group_prompt = """The task is to group textual description pairs of visual discriminative features for tumor detection in histopathology. 
Current Prompt Pairs: Format: <Index. Prompt Pair>
{prompt_pairs_str}
Each pair corresponds to a feature of the same medical concept. Group the prompt pairs that has exactly same observation but differ only in language variations. Give the indexes of the grouped pairs in the output.
Provide the output as follows: list[list[index:int]]. Make sure to include all pairs in the output, even if they are not grouped with others.
Let's think step by step.
"""
        self.retry_prompt = """The task is to group textual description pairs of visual discriminative features for tumor detection in histopathology. 
Current Prompt Pairs:
{prompt_pairs_str}

You've already grouped some pairs, but there are still ungrouped pairs remaining.
Current Grouped indexes:
{current_grouped_indexes}

Remaining Prompt Pairs:
{prompt_pairs_str_remaining}

Provide the output as follows: list[list[index:int]]. Make sure to include all pairs in the output, even if they are not grouped with others.
Let's think step by step."""

    def _parse_grouped_indexes(self, text: str):
        return ast.literal_eval(text)

    def _get_unique_indexes(self, grouped_indexes: list[list[int]]) -> list[int]:
        unique_indexes: list[int] = []
        for group in grouped_indexes:
            # if the group is not a list and just an integer, let's just append it.
            if isinstance(group, int):
                unique_indexes.append(group)
            else:
                # Append the first index of each group
                unique_indexes.append(group[0])

        return unique_indexes

    def _flatten_grouped_list(self, grouped_indexes: list[list[int]]) -> list[int]:
        # sometimes the list from LLM is in the format of [[1, 2], [2, 3], 4, 5, 6]
        # have to accommodate for that - flatten both nested lists and individual items
        flat_list = []
        for item in grouped_indexes:
            if isinstance(item, list):
                flat_list.extend(item)
            else:
                flat_list.append(item)

        return flat_list

    def _get_remaining_indexes(self, grouped_indexes: list[list[int]], total_count: int) -> list[int]:
        flat_list = self._flatten_grouped_list(grouped_indexes)
        missing_indexes = set(range(1, total_count + 1)) - set(flat_list)
        return list(missing_indexes)

    def _get_grouped_indexes_from_llm(self, llm_prompt: str) -> list[list[int]]:
        print("Sending Prompt: ", llm_prompt)
        for attempt in range(self.max_retries):
            try:
                response = self.client.get_llm_response(prompt=llm_prompt)
                print(response)
                # Try to extract code block with or without 'python'
                m = re.search(r'```(?:python)?\s*([\s\S]*?)\s*```', response)
                if not m:
                    raise ValueError(
                        "No code block found between triple backticks")
                list_str = m.group(1)
                grouped_indexes = ast.literal_eval(list_str)
                return grouped_indexes
            except Exception as e:
                print(f"Error in LLM response: {e}")
                if attempt == self.max_retries - 1:
                    raise ValueError(
                        "Failed to get a valid response from the LLM after multiple retries.")

    def perform_crowding(self, pq: util.PriorityQueue) -> util.PriorityQueue:
        deleted_num = 0
        for i in range(self.iterations):
            print(f"=== Iteration {i+1} of {self.iterations} ===")
            # retrieve the best prompt pairs from the priority queue
            prompt_pairs = pq.get_best_n(n=self.group_size)
            prompt_pairs_str = "\n".join(
                [f"{i+1}. ('{pair[0]}' , '{pair[1]}')" for i,
                 (pair, score) in enumerate(prompt_pairs)]
            )

            grouped_indexes = self._get_grouped_indexes_from_llm(
                llm_prompt=self.group_prompt.format(
                    prompt_pairs_str=prompt_pairs_str, num_of_prompts=self.group_size),
            )

            # Dealing with remaining indexes that were not grouped
            if self.group_size > len(prompt_pairs):
                # we have pruned the queue to less than group size
                remaining_indexes = self._get_remaining_indexes(
                    grouped_indexes,  len(prompt_pairs))
            else:
                # usual scenario when the pq size is greater than the grouping size
                remaining_indexes = self._get_remaining_indexes(
                    grouped_indexes, self.group_size)

            if len(remaining_indexes) > 0:
                retry_prompt_str = self.retry_prompt.format(
                    prompt_pairs_str=prompt_pairs_str,
                    current_grouped_indexes=str(grouped_indexes),
                    prompt_pairs_str_remaining="\n".join(
                        [f"{original_idx}. {prompt_pairs[original_idx-1][0]}" for original_idx in remaining_indexes]
                    )
                )
                grouped_indexes = self._get_grouped_indexes_from_llm(
                    llm_prompt=retry_prompt_str,
                )

            unique_indexes = self._get_unique_indexes(grouped_indexes)
            print(f"Unique indexes: {unique_indexes}")

            # select the best prompts based on the unique indexes
            print("Debug: Length of prompt pairs before selecting best:",
                  len(prompt_pairs))

            # if any index is out of range, skip this iteration
            if any(i < 1 or i > len(prompt_pairs) for i in unique_indexes):
                print("Warning: Some indexes are out of range. Skipping this iteration.")
                continue

            best_prompt_pairs_with_scores = [
                prompt_pairs[i-1] for i in unique_indexes]

            # delete the top n prompts from the priority queue
            pq.delete_top_n(self.group_size)
            # add the best prompts back to the priority queue
            for prompt_pair, score in best_prompt_pairs_with_scores:
                pq.insert(prompt_pair, score)

            # print the number of deleted prompts
            deleted_num += (self.group_size - len(unique_indexes))
            print(
                f"Iteration {i+1} completed. Deleted {deleted_num} duplicate prompts so far.")

        pq.delete_top_n(pq.max_capacity)  # Clear the queue at the end
        for prompt_pair, score in best_prompt_pairs_with_scores:
            # Reinsert the best prompts into the queue
            pq.insert(prompt_pair, score)

        return pq


def save_results_to_file(prompt_pairs_with_scores, filename):
    """Save prompt pairs and scores to a file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for prompt_pair, score in prompt_pairs_with_scores:
            f.write(f"{prompt_pair}, Score: {score}\n")
    print(f"Results saved to {filename}")


def save_evaluation_results(results, center_id, filename):
    """Save evaluation results to a file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a') as f:
        f.write(f"\n--- Center {center_id} Evaluation Results ---\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"AUC: {results['auc']:.4f}\n")
        f.write(f"Confusion Matrix:\n{results['cm']}\n")
        f.write(f"Classification Report:\n{results['report']}\n")
        f.write("-" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Crowd Pruning and Knee Point Analysis")
    parser.add_argument("--few_shot", type=int, default=1,
                        help="Number of few-shot examples (default: 1)")
    parser.add_argument("--provider", type=str, default="gemini",
                        choices=["gemini", "openai", "anthropic"],
                        help="LLM provider for crowding (default: gemini)")
    parser.add_argument("--max_capacity", type=int, default=1000,
                        help="Maximum capacity of priority queue (default: 1000)")
    parser.add_argument("--filter_threshold", type=float, default=0.6,
                        help="Filter threshold for priority queue (default: 0.6)")
    parser.add_argument("--crowding_iterations", type=int, default=3,
                        help="Number of crowding iterations (default: 3)")
    parser.add_argument("--group_size", type=int, default=30,
                        help="Number of prompts to group in each iteration (default: 30)")

    args = parser.parse_args()

    print(
        f"Starting crowd pruning and knee point analysis with {args.few_shot}-shot configuration...")
    print(f"Using {args.provider} as LLM provider")

    # Step 1: Load initial prompts and create priority queue
    print("\n=== Step 1: Loading Initial Prompts ===")
    initial_prompts = util.load_last_iteration_prompts(
        f"final_results/Experiment-70-strategy-inv-bce-gemma3-{args.few_shot}shot_opt_pairs.txt"
    )
    pq = util.PriorityQueue(
        max_capacity=args.max_capacity,
        filter_threshold=args.filter_threshold,
        initial=initial_prompts
    )

    print(
        f"Initial prompts loaded. Priority queue size: {len(pq.get_best_n(n=1000))}")

    # Step 2: Perform crowding
    print("\n=== Step 2: Performing Crowding ===")
    llm_client = util.LLMClient(provider=args.provider)
    crowding_manager = CrowdingManager(
        client=llm_client,
        iterations=args.crowding_iterations,
        group_size=args.group_size
    )

    pq = crowding_manager.perform_crowding(pq=pq)

    # Save crowded results
    crowded_results = pq.get_best_n(n=100)
    crowded_filename = f"final_results/crowded/{args.few_shot}-shot.txt"
    save_results_to_file(crowded_results, crowded_filename)

    print(f"Crowding completed. Reduced to {len(crowded_results)} prompts.")

    # Step 3: Find knee point
    print("\n=== Step 3: Finding Knee Point ===")
    scores = [score for (_, score) in pq.get_best_n(n=1000)]
    knee_point_analyzer = util.KneePointAnalysis(scores)
    knee_point = knee_point_analyzer.find_knee_point()

    print(f"Knee point found at: {knee_point}")

    # Save knee point results
    knee_results = pq.get_best_n(n=knee_point)
    knee_filename = f"final_results/knee/{args.few_shot}-shot.txt"
    save_results_to_file(knee_results, knee_filename)

    # Step 4: Final performance evaluation
    print("\n=== Step 4: Final Performance Evaluation ===")

    # Load model, process, and tokenizer
    print("Loading CLIP model...")
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # Load dataset
    print("Extracting center embeddings...")
    centers_features: List[np.ndarray]
    centers_labels: List[np.ndarray]
    centers_features, centers_labels = util.extract_center_embeddings(
        model=model,
        preprocess=preprocess,
        num_centers=5,  # evaluating on all centers
        isTrain=False,  # Evaluating on test centers only
    )
    print("Center embeddings extracted successfully.")

    # Convert to torch tensors for each center
    centers_features = [torch.from_numpy(feat) for feat in centers_features]
    centers_labels = [torch.from_numpy(label) for label in centers_labels]

    # Evaluate each center
    evaluation_filename = f"final_results/evaluation/{args.few_shot}-shot-results.txt"

    # Clear previous results file
    os.makedirs(os.path.dirname(evaluation_filename), exist_ok=True)
    with open(evaluation_filename, 'w') as f:
        f.write(f"=== {args.few_shot}-Shot Evaluation Results ===\n")
        f.write(f"Knee point: {knee_point}\n")
        f.write(f"Number of prompts used: {len(knee_results)}\n")
        f.write("=" * 50 + "\n")

    for i, _ in enumerate(centers_features):
        print(f"Evaluating center {i}...")
        results = util.evaluate_prompt_list(
            pq.get_best_n(n=knee_point),
            centers_features[i],
            centers_labels[i],
            model,
            tokenizer,
            unweighted=False
        )

        # Print results
        print(f"\n--- Center {i} Ensemble Evaluation Results ---")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"AUC: {results['auc']:.4f}")
        print("Confusion Matrix:\n", results['cm'])
        print("Classification Report:\n", results['report'])

        # Save results to file
        save_evaluation_results(results, i, evaluation_filename)

    print(f"\n=== Evaluation Complete ===")
    print(f"Results saved to:")
    print(f"  - Crowded prompts: {crowded_filename}")
    print(f"  - Knee point prompts: {knee_filename}")
    print(f"  - Evaluation results: {evaluation_filename}")


if __name__ == "__main__":
    main()
