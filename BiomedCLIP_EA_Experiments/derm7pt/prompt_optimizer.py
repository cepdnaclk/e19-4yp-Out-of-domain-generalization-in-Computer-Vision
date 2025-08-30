"""
Optimization Only - No Evolutionary Algorithm (EA) - Prompt Optimization Script
"""
import sys
from typing import List
import util
import torch
import numpy as np
import os
# from chatgpt_initial import INITIAL_CHATGPT_PROMPTS


def get_prompt_template(prompt_content: str, generate_n: int = 10) -> str:
    """
    Returns the appropriate instruction based on the iteration number range.

    Args:
        iteration_num: Current iteration number (1-indexed)

    Returns:
        String containing the iteration-specific instruction

    """
    # define a dictionary to map iteration ranges to instructions

    # Base meta prompt template
    base_meta_prompt_template = """The task is to generate distinct textual descriptions pairs of visual discriminative features to identify whether a dermoscopic image shows melanoma or not. 
    Here are the best performing pairs in ascending order. High scores indicate higher quality visual discriminative features.
    {content}
    Write {generate_n} new prompt pairs that are different from the old ones and has a score as high as possible, formulate a strategy.
    Only provide the output as Python code in the following format: prompts = list[tuple[negative: str, positive: str]]. Let's think step-by-step
    """

    return base_meta_prompt_template.format(
        content=prompt_content,
        generate_n=generate_n,
    )


def main():

    label_type = "melanoma"

    # Name the experiment we are currently running
    experiment_name = "Derm7pt_Expertiment5_WeightedinvertedBCE_" + label_type
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

    # 2. load dataset - MODIFIED FOR CHEXPERT
    features, labels = util.extract_embeddings(
        model=model,
        preprocess=preprocess,
        split="train",
        label_type=label_type,
    )

    # Convert to tensors - MODIFIED FOR MULTI-OBSERVATION SUPPORT
    all_feats = torch.from_numpy(features).float()
    all_labels = torch.from_numpy(labels).long()

    # print(f"shape of all_feats: {all_feats.shape} and all_labels: {all_labels.shape}")

    print(f"Loaded {len(all_feats)} Derm7pt embeddings")

    # 3. load initial prompts (optional)
    # initial_prompts = util.load_initial_prompts()

    # 4. Initialize the LLM client
    # Set use_local_ollama to True if you want to use a local Ollama server
    client = util.LLMClient(
        use_local_ollama=False, ollama_model="hf.co/unsloth/medgemma-27b-text-it-GGUF:Q8_0")

    # Configure the prompt templates
    meta_init_prompt = """Give 50 distinct textual descriptions of pairs of visual discriminative features to identify whether a dermoscopic image shows melanoma or not. Only provide the output as Python code in the following format: prompts = list[tuple[negative: str, positive: str]]. Let's think step-by-step"""

    # meta_prompt_template = """The task is to generate 50 textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section.
    # Here are the best performing pairs. You should aim to get higher scores. Each description should be about 5-20 words.
    # {content}
    # 1-10: Generate the first 10 pairs exploring variations of the top 1 (best) given. Remove certain words, add words, change order and generate variations
    # 11-20: Generate 10 pairs using the top 10, explore additional knowledge and expand on it.
    # 21-30: The next 10 pairs should maintain similar content as middle pairs but use different language style and sentence structures.
    # 31-40: The next 10 pairs should combine knowledge of top pairs and bottom pairs.
    # 41-50: The remaining 10 pairs should be randomly generated.
    # Only give the output as python code in the format - prompts: list[tuple[negative: str, positive: str]]
    # """

    # Optimization loop
    # initial_prompts = util.load_initial_prompts(
    #     "experiment_results/medical_concepts.txt")
    pq = util.PriorityQueue(max_capacity=1000)
    prompt_content = ""

    for j in range(500):
        if j == 0:
            prompts = util.get_prompt_pairs(meta_init_prompt, client)
            # prompts = INITIAL_CHATGPT_PROMPTS
        else:
            meta_prompt = get_prompt_template(
                prompt_content=prompt_content, generate_n=10)

            prompts = util.get_prompt_pairs(meta_prompt, client)

        for i, prompt_pair in enumerate(prompts):
            if len(prompt_pair) != 2:
                print(f"Invalid prompt pair: {prompt_pair}")
                continue
            negative_prompt, positive_prompt = prompt_pair
            results = util.evaluate_prompt_pair(
                negative_prompt, positive_prompt, all_feats, all_labels, model, tokenizer)
            # print(f"Weighted Inverted BCE for prompt pair {i+1}: {results['weighted_inverted_bce']:.4f} {results['accuracy']} F1: {results['f1']:.4f}")
            pq.insert((negative_prompt, positive_prompt),
                      results['weighted_inverted_bce'])

        n = 10
        print(f"\nCurrent Top {n} prompt pairs:")

        selected_prompts = pq.get_roulette_wheel_selection(
            n, isNormalizedInts=True)

        selected_prompts = sorted(
            selected_prompts, key=lambda x: x[1], reverse=False)

        # Prepare the content for the meta prompt
        prompt_content = f"Current Top {n} prompt pairs:\n"
        for i, (prompt_pair, score) in enumerate(selected_prompts):
            print(f"{i+1}. {prompt_pair}, Weighted Inverted BCE: {int(score)}")
            prompt_content += f"{prompt_pair}, Weighted Inverted BCE: {int(score)}\n"

        # Save the best prompt pairs to a file, every 10 iterations
        if (j + 1) % 10 == 0 or j == 0:
            top_prompts = pq.get_best_n(1000)
            with open(results_filename, "a") as f:
                f.write(f"Iteration {j+1}:\n")
                for prompt_pair, score in top_prompts:
                    f.write(f"{prompt_pair}, Score: {score:.4f}\n")
                f.write("\n")

        # print the average Weighted Inverted BCE of the top n prompts
        print(
            f"Iteration {j+1}: mean Weighted Inverted BCE of top 10: {pq.get_average_score(10)}.\n")


if __name__ == "__main__":
    main()
