import util
import torch
import numpy as np
import os
import re
import ast
from typing import List

# --- Configuration ---
CROWDING_INTERVAL = 2        # perform crowding every X iterations
CROWDING_ITERATIONS = 5       # number of crowding passes
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
        flat_list = [item for sublist in grouped_indexes for item in sublist]
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
                        [f"{remaining_indexes[i]}. ('{pair[0]}' , '{pair[1]}')" for i, (pair, score) in enumerate(
                            [prompt_pairs[i-1] for i in remaining_indexes])]
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
    # define a dictionary to map iteration ranges to instructions
    instruction_map = {
        "medical_concepts": f"Write {generate_n} new prompt pairs that are different from the old ones and has a score as high as possible.",
        "similar": f"Write {generate_n} new prompt pairs that are more similar to the high scoring prompts.",
        "combined_medical_concepts": f"Write {generate_n} new prompt pairs by combining multiple medical concepts only from the above prompts to make the score as high as possible.",
        "language_styles": f"Write {generate_n} new prompt pairs by paraphrasing each of the above. Each pair should have distinct language style.",
        "slight_changes": f"Write {generate_n} new prompt pairs similar to the above pairs only making slight changes to the language style to make the score as high as possible.",
        "summarize_and_mutate": f"Please follow the instruction step-by-step to generate a better prompt pair with a score greater than 90.\nStep 1: Write one prompt pair that combines all the knowledge from the above prompts.\nStep 2:  Mutate the generated prompt pair in {generate_n} different ways so that each description cohesive.",
        "explainability": "For each prompt pair, rewrite them by including a brief rationale for why each discriminative feature predicts tumor vs. non-tumor.",
        "quantitative": f"Write {generate_n} new prompt pairs that adds quantitative cues to the qualitative prompts given above. Score as high as possible.",
        "borderline": f"Write {generate_n} new prompt pairs appending rare or borderline patterns which are easily misclassified to score as high as possible.",
        "expert": f"Write {generate_n} new prompt pairs expanding each prompt by appending expert biomedical knowledge to score as high as possible.",
        "strategy": f"Write {generate_n} new prompt pairs that are different to from the old ones and has a score as high as possible. Formulate a strategy",
    }

    # Base meta prompt template
    base_meta_prompt_template = """The task is to generate distinct textual descriptions pairs of visual discriminative features to identify whether the central region of a histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section.
    Here are the best performing pairs in ascending order. High scores indicate higher quality visual discriminative features. Each prompt should contain about 10 words.
    {content}
    {iteration_specific_instruction}
    Only provide the output as Python code in the following format: prompts = list[tuple[negative: str, positive: str]]. Let's think step-by-step
    """

    if 1 <= iteration_num <= 2000:
        # Iterations 1-50: Basic exploration
        return base_meta_prompt_template.format(
            content=prompt_content,
            iteration_specific_instruction=instruction_map["strategy"]
        )
    elif 2001 <= iteration_num <= 3000:
        # Iterations 51-100: Concept combination
        return base_meta_prompt_template.format(
            content=prompt_content,
            iteration_specific_instruction=instruction_map["combined_medical_concepts"]
        )
    elif 3001 <= iteration_num <= 4000:
        # Iterations 101-200: Language style variation
        return base_meta_prompt_template.format(
            content=prompt_content,
            iteration_specific_instruction=instruction_map["similar"]
        )
    elif iteration_num > 4001:
        # Iterations 201+: Fine-tuning with slight modifications
        return base_meta_prompt_template.format(
            content=prompt_content,
            iteration_specific_instruction=instruction_map["slight_changes"]
        )
    else:
        # Fallback (shouldn't happen with normal iteration numbering)
        raise IndexError("Error occured when getting prompt template")


def main():
    # Name the experiment we are currently running
    experiment_name = "Experiment-43-strategy-regularized_bce_inverted-gemma3"
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

    # Configure the prompt templates
    meta_init_prompt = """Give 50 distinct textual descriptions of pairs of visual discriminative features to identify whether the central region of a histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section. Each prompt should contain about 10 words. Only provide the output as Python code in the following format: prompts = list[tuple[negative: str, positive: str]]. Let's think step-by-step"""

    # Optimization loop
    # initial_prompts = util.load_initial_prompts(
    #     "experiment_results/medical_concepts.txt")
    pq = util.PriorityQueue(max_capacity=1000, filter_threshold=0.4)
    prompt_content = ""
    crowd_manager = CrowdingManager(client)

    for j in range(1000):
        if j == 0:
            prompts = util.get_prompt_pairs(meta_init_prompt, client)
            # prompts = INITIAL_CHATGPT_PROMPTS
        else:
            meta_prompt = get_prompt_template(
                iteration_num=j, prompt_content=prompt_content, generate_n=10)

            prompts = util.get_prompt_pairs(meta_prompt, client)

        for i, prompt_pair in enumerate(prompts):
            if len(prompt_pair) != 2:
                print(f"Invalid prompt pair: {prompt_pair}")
                continue
            negative_prompt, positive_prompt = prompt_pair
            results = util.evaluate_prompt_pair(
                negative_prompt, positive_prompt, all_feats, all_labels, model, tokenizer)
            # print(
            #     f"Inverted BCE for prompt pair {i+1}: {results['inverted_bce']:.4f} {results['accuracy']}")
            pq.insert((negative_prompt, positive_prompt),
                      results['inverted_bce'])

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
            f"Iteration {j+1}: mean accuracy of top 10: {pq.get_average_score(10)}.\n")


if __name__ == "__main__":
    main()
