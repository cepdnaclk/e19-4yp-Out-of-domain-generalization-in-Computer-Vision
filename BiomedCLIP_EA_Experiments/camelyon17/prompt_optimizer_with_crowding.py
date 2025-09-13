import util
import torch
import numpy as np
import os
import re
import ast
from typing import List

# --- Configuration ---
CROWDING_INTERVAL = 10         # perform crowding every X iterations
CROWDING_ITERATIONS = 5      # number of crowding passes
NUMBER_OF_PROMPTS_TO_GROUP = 30
MAX_RETRIES = 5


FITNESS_METRIC = 'inverted_bce'  # 'accuracy', 'auc', 'f1_macro', 'inverted_bce'


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
Current Prompt Pairs:
{prompt_pairs_str}
Group the prompt pairs that has exactly same observation but differ only in language variations. Give the indexes of the grouped pairs in the output.
Provide the output as follows: list[list[index:int]]. Make sure to include all pairs in the output, even if they are not grouped with others.
Let's think step by step. Count from 1-{num_of_prompts} to verify each item is in the list.
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
            # Append the first index of each group
            unique_indexes.append(group[0])
        return unique_indexes

    def _get_remaining_indexes(self, grouped_indexes: list[list[int]], total_count: int) -> list[int]:
        # sometimes the list from LLM is in the format of [[1, 2], [2, 3], 4, 5, 6]
        # have to accommodate for that - flatten both nested lists and individual items
        flat_list = []
        for item in grouped_indexes:
            if isinstance(item, list):
                flat_list.extend(item)
            else:
                flat_list.append(item)

        missing_indexes = set(range(1, total_count + 1)) - set(flat_list)
        return list(missing_indexes)

    def _get_grouped_indexes_from_llm(self, llm_prompt: str) -> list[list[int]]:
        print("Sending Prompt: ", llm_prompt)
        for attempt in range(self.max_retries):
            try:
                response = self.client.get_llm_response(prompt=llm_prompt)
                print(response)
                m = re.search(r'```python\s*([\s\S]*?)\s*```', response)
                if not m:
                    raise ValueError("No ```python ... ``` block found")
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

            remaining_indexes = self._get_remaining_indexes(
                grouped_indexes, self.group_size)
            if len(remaining_indexes) > 0:
                print(f"Remaining indexes: {remaining_indexes}")
                retry_prompt_str = self.retry_prompt.format(
                    prompt_pairs_str=prompt_pairs_str,
                    current_grouped_indexes=str(grouped_indexes),
                    prompt_pairs_str_remaining="\n".join(
                        [f"{original_idx}. ('{prompt_pairs[original_idx-1][0]}' , '{prompt_pairs[original_idx-1][1]}')" for original_idx in remaining_indexes]
                    )
                )
                grouped_indexes = self._get_grouped_indexes_from_llm(
                    llm_prompt=retry_prompt_str,
                )

            unique_indexes = self._get_unique_indexes(grouped_indexes)
            print(f"Unique indexes: {unique_indexes}")

            # select the best prompts based on the unique indexes
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


def get_prompt_template(iteration_num: int, prompt_content: str, generate_n: int = 10) -> str:
    """
    Returns the appropriate instruction based on the iteration number range.

    Args:
        iteration_num: Current iteration number (1-indexed)

    Returns:
        String containing the iteration-specific instruction

    """

    # First iteration: meta initialize prompt
    if iteration_num == 0:
        return """Give 50 distinct textual descriptions of pairs of visual discriminative features to identify whether the central region of a histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section. Only provide the output as Python code in the following format: prompts = list[tuple[negative: str, positive: str]]. Let's think step-by-step"""

    # Base meta prompt template
    base_meta_prompt_template = """The task is to generate distinct textual descriptions pairs of visual discriminative features to identify whether the central region of a histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section.
    Here are the best performing pairs in ascending order. High scores indicate higher quality visual discriminative features.
    {content}
    Write {generate_n} new prompt pairs that are different to from the old ones and has a score as high as possible. Formulate a strategy.
    Only provide the output as Python code in the following format: prompts = list[tuple[negative: str, positive: str]]. Let's think step-by-step
    """

    return base_meta_prompt_template.format(
        content=prompt_content,
        generate_n=generate_n,
    )


def main():
    # Name the experiment we are currently running
    experiment_name = "Experiment-1-bce_inverted-gemma3"
    print(f"Running {experiment_name}...")

    # Create experiment results directory
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)

    # Create filename with experiment name
    results_filename = os.path.join(
        results_dir, f"{experiment_name}_opt_pairs.txt")

   # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. load dataset
    # 1) Unpack—annotate what extract_center_embeddings returns
    centers_features: List[np.ndarray]
    centers_labels:   List[np.ndarray]
    centers_features, centers_labels = util.extract_center_embeddings(
        model=model,
        preprocess=preprocess,
        num_centers=3,  # trained only on center 0
    )

    # 2) Concatenate and convert—annotate the resulting tensors
    all_feats: torch.Tensor = torch.from_numpy(
        np.concatenate(centers_features, axis=0)
    ).float()   # shape: (N_total, D), dtype=torch.float32

    all_labels: torch.Tensor = torch.from_numpy(
        np.concatenate(centers_labels, axis=0)
    ).long()    # shape: (N_total,), dtype=torch.int64

    print("Center embeddings extracted successfully.")

    # 3. load initial prompts (optional)
    # initial_prompts = util.load_initial_prompts()

    # 4. Initialize the LLM client
    # 'gemini', 'ollama', or 'azure_openai'
    client = util.LLMClient(provider='gemini')

    # Optimization loop
    # initial_prompts = util.load_initial_prompts(
    #     "experiment_results/medical_concepts.txt")
    pq = util.PriorityQueue(max_capacity=1000, filter_threshold=0.4)
    prompt_content = ""
    crowd_manager = CrowdingManager(client)

    for j in range(1000):

        meta_prompt = get_prompt_template(
            iteration_num=j, prompt_content=prompt_content, generate_n=10)

        prompts = util.get_prompt_pairs(meta_prompt, client)

        for i, prompt_pair in enumerate(prompts):

            # validate the prompt pair
            if len(prompt_pair) != 2:
                print(f"Invalid prompt pair: {prompt_pair}")
                continue

            negative_prompt, positive_prompt = prompt_pair
            results = util.evaluate_prompt_pair(
                negative_prompt, positive_prompt, all_feats, all_labels, model, tokenizer)
            # print(
            #     f"Inverted BCE for prompt pair {i+1}: {results['inverted_bce']:.4f} {results['accuracy']}")
            pq.insert((negative_prompt, positive_prompt),
                      results[FITNESS_METRIC])

        n = 10
        print(f"\nCurrent Top {n} prompt pairs:")

        selected_prompts = pq.get_roulette_wheel_selection(
            n, isNormalizedInts=True)
        # reverse the order to set it to acsending order: Recency Bias
        selected_prompts = sorted(
            selected_prompts, key=lambda x: x[1], reverse=False)

        # Prepare the content for the meta prompt
        prompt_content = f"Current Top {n} prompt pairs:\n"
        for i, (prompt_pair, score) in enumerate(selected_prompts):
            print(f"{i+1}. {prompt_pair}, Score: {int(score)}")
            prompt_content += f"{prompt_pair}, Score: {int(score)}\n"

        # Perform crowding every CROWDING_INTERVAL iterations
        if (j + 1) % CROWDING_INTERVAL == 0:
            print(f"Performing crowding at iteration {j+1}...")
            pq = crowd_manager.perform_crowding(pq)
            print("Crowding completed.")

        # Save the best prompt pairs to a file, every 10 iterations
        if (j + 1) % 1 == 0 or j == 0:
            top_prompts = pq.get_best_n(1000)
            with open(results_filename, "a") as f:
                f.write(f"Iteration {j+1}:\n")
                for prompt_pair, score in top_prompts:
                    f.write(f"{prompt_pair}, Score: {score:.4f}\n")
                f.write("\n")

        # print the average score of the top n prompts
        print(
            f"Iteration {j+1}: mean {FITNESS_METRIC} of top 10: {pq.get_average_score(10)}.\n")


if __name__ == "__main__":
    main()
