#!/usr/bin/env python3
"""
Crowd Pruning Script for WBC Classification

This script performs crowding on prompt pairs to remove duplicates and similar prompts.
It performs the following operations:
1. Load initial prompts and create a priority queue
2. Perform crowding using LLM to group similar prompts
3. Save the crowded results

Usage:
    python crowding.py --few_shot N --class_label CLASS [--provider PROVIDER]
    
Example:
    python crowding.py --few_shot 1 --class_label Basophil
    python crowding.py --few_shot 2 --class_label Neutrophil --provider openai
"""

import argparse
import ast
import re
import os

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

        self.group_prompt = """The task is to group textual description pairs of visual discriminative features for white blood cell classification. 
Current Prompt Pairs: Format: <Index. Prompt Pair>
{prompt_pairs_str}
Each pair corresponds to a feature of the same medical concept. Group the prompt pairs that has exactly same observation but differ only in language variations. Give the indexes of the grouped pairs in the output.
Provide the output as follows: list[list[index:int]]. Make sure to include all pairs in the output, even if they are not grouped with others.
Let's think step by step.
"""
        self.retry_prompt = """The task is to group textual description pairs of visual discriminative features for white blood cell classification. 
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


def main():
    parser = argparse.ArgumentParser(
        description="Crowd Pruning for WBC Classification")
    parser.add_argument("--few_shot", type=int, default=1,
                        help="Number of few-shot examples (default: 1)")
    parser.add_argument("--class_label", type=str, required=True,
                        choices=["Basophil", "Eosinophil",
                                 "Lymphocyte", "Monocyte", "Neutrophil"],
                        help="WBC class label to process")
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
        f"Starting crowd pruning with {args.few_shot}-shot configuration for {args.class_label}...")
    print(f"Using {args.provider} as LLM provider")

    # Step 1: Load initial prompts and create priority queue
    print("\n=== Step 1: Loading Initial Prompts ===")
    initial_prompts = util.load_last_iteration_prompts(
        f"final_results/Wbcatt_Experiment1_inverted_weighted_ce-FEWSHOT{args.few_shot}-{args.class_label}_opt_pairs.txt"
    )
    pq = util.PriorityQueue(
        max_capacity=args.max_capacity,
        filter_threshold=args.filter_threshold,
        initial=initial_prompts
    )

    print(
        f"Initial prompts loaded for {args.class_label}. Priority queue size: {len(pq.get_best_n(n=1000))}")

    # Step 2: Perform crowding
    print("\n=== Step 2: Performing Crowding ===")
    llm_client = util.LLMClient(provider=args.provider)
    crowding_manager = CrowdingManager(
        client=llm_client,
        iterations=args.crowding_iterations,
        group_size=args.group_size
    )

    pq = crowding_manager.perform_crowding(pq=pq)

    # Step 3: Save crowded results
    print("\n=== Step 3: Saving Results ===")
    crowded_results = pq.get_best_n(n=100)
    crowded_filename = f"final_results/crowded/{args.few_shot}-shot-{args.class_label}.txt"
    save_results_to_file(crowded_results, crowded_filename)

    print(f"\n=== Crowding Complete for {args.class_label} ===")
    print(
        f"Reduced from initial prompts to {len(crowded_results)} crowded prompts.")
    print(f"Results saved to: {crowded_filename}")


if __name__ == "__main__":
    main()
