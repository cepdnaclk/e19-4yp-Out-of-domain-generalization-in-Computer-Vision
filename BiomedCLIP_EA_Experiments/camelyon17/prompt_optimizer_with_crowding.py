import re
import ast
from typing import List, Tuple
import util
import torch
import numpy as np
import os

# Crowding parameters
CROWDING_INTERVAL = 20  # Perform crowding burst after this many main iterations
# Number of consecutive crowding steps in a burst
NUM_CROWDING_ITERATIONS_PER_BURST = 5
# Number of top prompts to consider for grouping by LLM
NUMBER_OF_PROMPTS_TO_GROUP = 50
# Max retries for LLM calls during crowding
MAX_RETRIES_CROWDING = 5


# --- CrowdingManager Class ---
class CrowdingManager:
    """
    Manages the crowding-based selection and duplicate removal process
    for prompt pairs using an LLM.
    """

    def __init__(self, llm_client, prompt_grouping_size: int, max_retries: int):
        self._llm_client = llm_client
        self._prompt_grouping_size = prompt_grouping_size
        self._max_retries = max_retries

        self._base_llm_prompt_template = """The task is to group textual description pairs of visual discriminative features for tumor detection in histopathology. 
Current Prompt Pairs:
{prompt_pairs_str}
Group the prompt pairs that has exactly same observation but differ only in language variations. Give the indexes of the grouped pairs in the output.
Provide the output as follows: list[list[index:int]]. Make sure to include all pairs in the output, even if they are not grouped with others.
Let's think step by step. Count from 1-{num_of_prompts} to verify each item is in the list.
"""

        self._retry_llm_prompt_template = """The task is to group textual description pairs of visual discriminative features for tumor detection in histopathology. 
Current Prompt Pairs:
{prompt_pairs_str}

You've already grouped some pairs, but there are still ungrouped pairs remaining.
Current Grouped indexes:
{current_grouped_indexes}

Remaining Prompt Pairs:
{prompt_pairs_str_remaining}

Provide the output as follows: list[list[index:int]]. Make sure to include all pairs in the output, even if they are not grouped with others.
Let's think step by step."""

    @staticmethod
    def _parse_grouped_indexes(text: str) -> List[List[int]]:
        """Parses the LLM output string to a list of grouped indexes."""
        # Check if the text is already a list, if so, return directly
        if isinstance(text, list):
            return text

        # Try to find a python code block
        m = re.search(r'```python\s*([\s\S]*?)\s*```', text)
        if m:
            list_str = m.group(1)
        else:
            # If no code block, assume the entire text is the list string
            list_str = text.strip()

        # Attempt to parse, handle cases where it might just be the list string directly
        try:
            return ast.literal_eval(list_str)
        except (ValueError, SyntaxError) as e:
            # Fallback for malformed output: try to extract obvious list structures
            print(
                f"Warning: Failed to parse LLM output using ast.literal_eval: {e}. Attempting regex fallback.")
            # Simple regex to find list of lists of integers
            matches = re.findall(r'\[(\s*\d+\s*(?:,\s*\d+\s*)*)\]', list_str)
            result = []
            for match in matches:
                try:
                    result.append([int(idx.strip())
                                  for idx in match.split(',') if idx.strip()])
                except ValueError:
                    continue  # Skip malformed inner lists
            if result:
                return result
            raise ValueError(
                f"Could not parse grouped indexes from LLM response: {list_str}") from e

    @staticmethod
    def _get_unique_indexes(grouped_indexes: List[List[int]]) -> List[int]:
        """Returns the first index from each group, representing unique concepts."""
        unique_indexes: List[int] = []
        # Ensure that all indexes are accounted for, even if they are in a group of one
        flat_list_set = set(
            item for sublist in grouped_indexes for item in sublist)

        # First, add the primary representative of each group
        for group in grouped_indexes:
            if group:  # Ensure group is not empty
                unique_indexes.append(group[0])

        # Then, add any singletons that were not part of any group (i.e., [idx] where idx is not in another group)
        # This handles cases where LLM might just return [1], [2], [3]
        for group in grouped_indexes:
            if len(group) == 1 and group[0] not in unique_indexes:
                unique_indexes.append(group[0])

        # Ensure uniqueness and sort for consistency
        return sorted(list(set(unique_indexes)))

    @staticmethod
    def _get_remaining_indexes(grouped_indexes: List[List[int]], total_count: int) -> List[int]:
        """Identifies indexes that were not included in any group."""
        flat_list = [item for sublist in grouped_indexes for item in sublist]
        missing_indexes = set(range(1, total_count + 1)) - set(flat_list)
        return sorted(list(missing_indexes))

    def _get_grouped_indexes_from_llm(self, llm_prompt: str) -> List[List[int]]:
        """Sends a prompt to the LLM and parses the grouped indexes from its response."""
        print("Sending Prompt to LLM for grouping...")
        for attempt in range(self._max_retries):
            try:
                response = self._llm_client.get_llm_response(
                    prompt=llm_prompt)
                # print(f"LLM Response (attempt {attempt+1}):\n{response}") # For debugging LLM raw response
                grouped_indexes = self._parse_grouped_indexes(response)
                # Basic validation: ensure all sublists contain integers and overall structure is list[list[int]]
                if not all(isinstance(group, list) and all(isinstance(idx, int) for idx in group) for group in grouped_indexes):
                    raise ValueError(
                        "Parsed output is not in the expected list[list[int]] format.")
                return grouped_indexes
            except Exception as e:
                print(
                    f"Error in LLM response during grouping (Attempt {attempt + 1}/{self._max_retries}): {e}")
                if attempt == self._max_retries - 1:
                    raise ValueError(
                        "Failed to get a valid response from the LLM after multiple retries for grouping.")

    def perform_crowding(self, pq: util.PriorityQueue) -> int:
        """
        Performs a single crowding operation on the top prompts in the priority queue.
        Removes duplicates and re-inserts unique representative prompts.

        Args:
            pq: The PriorityQueue instance containing prompt pairs and their scores.

        Returns:
            The number of duplicate prompts removed.
        """
        # print(f"\nPerforming crowding on top {self._prompt_grouping_size} prompts...") # This print moved to the main loop

        # 1. Retrieve the best prompt pairs from the priority queue for grouping
        # Get actual (pair, score) tuples, 0-indexed for internal use
        prompt_pairs_with_scores = pq.get_best_n(n=self._prompt_grouping_size)

        # Create a 1-indexed string representation for the LLM
        crowding_prompt_pairs_str = "\n".join(
            [f"{idx+1}. ('{pair[0]}' , '{pair[1]}')" for idx,
             (pair, score) in enumerate(prompt_pairs_with_scores)]
        )

        # 2. Call LLM for initial grouping
        llm_prompt_for_crowding = self._base_llm_prompt_template.format(
            prompt_pairs_str=crowding_prompt_pairs_str,
            num_of_prompts=self._prompt_grouping_size
        )
        grouped_indexes = self._get_grouped_indexes_from_llm(
            llm_prompt=llm_prompt_for_crowding)

        # 3. Check for remaining ungrouped indexes and retry if necessary
        remaining_indexes = self._get_remaining_indexes(
            grouped_indexes, self._prompt_grouping_size)
        if len(remaining_indexes) > 0:
            print(
                f"Remaining indexes after first grouping attempt: {remaining_indexes}. Retrying...")

            # Prepare remaining prompts string (1-indexed for LLM)
            # Need to get the actual prompts corresponding to remaining_indexes
            remaining_prompts_for_retry_str = "\n".join(
                [f"{i}. ('{prompt_pairs_with_scores[i-1][0]}' , '{prompt_pairs_with_scores[i-1][1]}')"
                 for i in remaining_indexes]
            )

            retry_prompt_str = self._retry_llm_prompt_template.format(
                prompt_pairs_str=crowding_prompt_pairs_str,  # Full list for context
                current_grouped_indexes=str(grouped_indexes),
                prompt_pairs_str_remaining=remaining_prompts_for_retry_str
            )
            grouped_indexes = self._get_grouped_indexes_from_llm(
                llm_prompt=retry_prompt_str)

        # 4. Get unique indexes (representatives) after all grouping attempts
        unique_indexes_1_based = self._get_unique_indexes(grouped_indexes)
        # print(f"Unique indexes identified after crowding: {unique_indexes_1_based}") # This print moved to main loop

        # 5. Select the best prompts based on the unique indexes (only keep one from each grouped set)
        # Convert 1-based unique_indexes back to 0-based for list access
        best_prompt_pairs_after_crowding = [
            prompt_pairs_with_scores[i-1] for i in unique_indexes_1_based
        ]

        # 6. Update the priority queue: delete old and insert new unique
        # This is effectively "removing" the duplicates that were grouped
        # Delete the original set that was processed
        pq.delete_top_n(self._prompt_grouping_size)

        # Add only the unique/representative prompts back to the priority queue
        for prompt_pair, score in best_prompt_pairs_after_crowding:
            pq.insert(prompt_pair, score)

        # 7. Calculate and return the number of duplicates removed
        deleted_duplicates_count = self._prompt_grouping_size - \
            len(unique_indexes_1_based)
        # print(f"Crowding completed. Removed {deleted_duplicates_count} duplicate prompts from the top {self._prompt_grouping_size}.") # This print moved to main loop
        return deleted_duplicates_count

# --- End of CrowdingManager Class ---


def get_prompt_template(iteration_num: int, prompt_content: str, generate_n: int = 10) -> str:
    # (Your existing get_prompt_template function, unchanged)
    """
    Returns the appropriate instruction based on the iteration number range.
    Args:
        iteration_num: Current iteration number (1-indexed)
    Returns:
        String containing the iteration-specific instruction
    """
    # define a dictionary to map iteration ranges to instructions
    instruction_map = {
        "medical_concepts": f"Write {generate_n} new prompt pairs that are different from the old ones and has a score as high as possible.",
        "similar": f"Write {generate_n} new prompt pairs that are more similar to the high scoring prompts.",
        "combined_medical_concepts": f"Write {generate_n} new prompt pairs by combining multiple medical concepts only from the above prompts to make the score as high as possible.",
        "language_styles": f"Write {generate_n} new prompt pairs by paraphrasing each of the above. Each pair should have distinct language style.",
        "slight_changes": f"Write {generate_n} new prompt pairs similar to the above pairs only making slight changes to the language style to make the score as high as possible.",
        "summarize_and_mutate": f"Please follow the instruction step-by-step to generate a better prompt pair with a score greater than 90.\nStep 1: Write one prompt pair that combines all the knowledge from the above prompts.\nStep 2: Mutate the generated prompt pair in {generate_n} different ways so that each description cohesive.",
        "explainability": "For each prompt pair, rewrite them by including a brief rationale for why each discriminative feature predicts tumor vs. non-tumor.",
        "quantitative": f"Write {generate_n} new prompt pairs that adds quantitative cues to the qualitative prompts given above. Score as high as possible.",
        "borderline": f"Write {generate_n} new prompt pairs appending rare or borderline patterns which are easily misclassified to score as high as possible.",
        "expert": f"Write {generate_n} new prompt pairs expanding each prompt by appending expert biomedical knowledge to score as high as possible.",
        "strategy": f"Write {generate_n} new prompt pairs that are different to from the old ones and has a score as high as possible. Formulate a strategy",
    }

    # Base meta prompt template
    base_meta_prompt_template = """The task is to generate distinct textual descriptions pairs of visual discriminative features to identify whether the central region of a histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section.
    Here are the best performing pairs in ascending order. High scores indicate higher quality visual discriminative features.
    {content}
    {iteration_specific_instruction}
    Only provide the output as Python code in the following format: prompts = list[tuple[negative: str, positive: str]]. Let's think step-by-step
    """

    if 1 <= iteration_num <= 2000:
        return base_meta_prompt_template.format(
            content=prompt_content,
            iteration_specific_instruction=instruction_map["strategy"]
        )
    elif 2001 <= iteration_num <= 3000:
        return base_meta_prompt_template.format(
            content=prompt_content,
            iteration_specific_instruction=instruction_map["combined_medical_concepts"]
        )
    elif 3001 <= iteration_num <= 4000:
        return base_meta_prompt_template.format(
            content=prompt_content,
            iteration_specific_instruction=instruction_map["similar"]
        )
    elif iteration_num > 4001:
        return base_meta_prompt_template.format(
            content=prompt_content,
            iteration_specific_instruction=instruction_map["slight_changes"]
        )
    else:
        raise IndexError("Error occurred when getting prompt template")


def main():
    # Updated experiment name
    experiment_name = "Experiment-51-strategy-bce_inverted-gemma3_crowding_burst"
    print(f"Running {experiment_name}...")

    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    results_filename = os.path.join(
        results_dir, f"{experiment_name}_opt_pairs.txt")

    # 1. Load model, preprocess, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. Load dataset
    centers_features: List[np.ndarray]
    centers_labels: List[np.ndarray]
    centers_features, centers_labels = util.extract_center_embeddings(
        model=model, preprocess=preprocess, num_centers=3
    )
    all_feats: torch.Tensor = torch.from_numpy(
        np.concatenate(centers_features, axis=0)).float()
    all_labels: torch.Tensor = torch.from_numpy(
        np.concatenate(centers_labels, axis=0)).long()
    print("Center embeddings extracted successfully.")

    # 3. Initialize the LLM client
    client = util.LLMClient(provider='gemini')

    # 4. Initialize CrowdingManager
    crowding_manager = CrowdingManager(
        llm_client=client,
        prompt_grouping_size=NUMBER_OF_PROMPTS_TO_GROUP,
        max_retries=MAX_RETRIES_CROWDING
    )

    # 5. Configure the initial meta prompt
    meta_init_prompt = """Give 50 distinct textual descriptions of pairs of visual discriminative features to identify whether the central region of a histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section. Only provide the output as Python code in the following format: prompts = list[tuple[negative: str, positive: str]]. Let's think step-by-step"""

    # 6. Initialize the Priority Queue
    pq = util.PriorityQueue(max_capacity=1000, filter_threshold=0.4)
    prompt_content = ""

    # Optimization loop
    for j in range(1, 1001):  # Iteration loop from 1 to 1000 for clarity
        print(f"\n--- Main Optimization Iteration {j} ---")

        # Step 6a: Generate new prompts
        if j == 1:  # First iteration, use initial prompt
            prompts = util.get_prompt_pairs(meta_init_prompt, client)
        else:  # Subsequent iterations, use adapted meta-prompt
            meta_prompt = get_prompt_template(
                iteration_num=j, prompt_content=prompt_content, generate_n=10
            )
            prompts = util.get_prompt_pairs(meta_prompt, client)

        # Step 6b: Evaluate and insert new prompts into PQ
        for i, prompt_pair in enumerate(prompts):
            if not isinstance(prompt_pair, tuple) or len(prompt_pair) != 2:
                print(f"Invalid prompt pair format: {prompt_pair}. Skipping.")
                continue
            negative_prompt, positive_prompt = prompt_pair
            results = util.evaluate_prompt_pair(
                negative_prompt, positive_prompt, all_feats, all_labels, model, tokenizer
            )
            pq.insert((negative_prompt, positive_prompt),
                      results['inverted_bce'])

        # Step 6c: Display current top prompts
        n_display = 10
        print(
            f"\nCurrent Top {n_display} prompt pairs (before crowding this iteration):")
        selected_prompts = pq.get_roulette_wheel_selection(
            n_display, isNormalizedInts=True)
        selected_prompts = sorted(
            selected_prompts, key=lambda x: x[1], reverse=False)  # Ascending order

        prompt_content = f"Current Top {n_display} prompt pairs:\n"
        for i, (prompt_pair, score) in enumerate(selected_prompts):
            # Display score as float for more precision during iterations
            print(f"{i+1}. {prompt_pair}, Score: {score:.4f}")
            # Keep prompt_content score as float for LLM context, it's better not to lose precision
            prompt_content += f"{prompt_pair}, Score: {score:.2f}\n"

        # Step 6d: Perform Crowding Burst - NEW LOGIC
        if j % CROWDING_INTERVAL == 0:
            print(
                f"\n--- Initiating {NUM_CROWDING_ITERATIONS_PER_BURST} crowding burst iterations after main iteration {j} ---")
            for crowding_iter_idx in range(1, NUM_CROWDING_ITERATIONS_PER_BURST + 1):
                print(
                    f"--- Crowding Burst Iteration {crowding_iter_idx} of {NUM_CROWDING_ITERATIONS_PER_BURST} ---")
                deleted_count = crowding_manager.perform_crowding(pq)
                print(
                    f"Crowding Burst Iteration {crowding_iter_idx} completed. Removed {deleted_count} duplicates.")
                # You might want to re-display top prompts here if you want to see the effect after each crowding step
                # For brevity, I've left it to display after the entire burst.

        # Step 6e: Save results and print stats
        # Changed (j % 1 == 0) to (j % 10 == 0) for typical saving frequency,
        # but kept j == 1 to save initial state. Adjust as needed.
        if j % 10 == 0 or j == 1:
            top_prompts = pq.get_best_n(pq.max_capacity)
            with open(results_filename, "w") as f:
                f.write(f"Iteration {j}:\n")
                for prompt_pair, score in top_prompts:
                    f.write(f"{prompt_pair}, Score: {score:.4f}\n")
                f.write("\n")

        print(
            f"Iteration {j}: mean inverted BCE of top 10: {pq.get_average_score(10):.4f}.\n")


if __name__ == "__main__":
    main()
