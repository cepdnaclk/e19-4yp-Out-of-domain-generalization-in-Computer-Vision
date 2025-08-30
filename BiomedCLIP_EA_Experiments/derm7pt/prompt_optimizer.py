"""
Optimization Only - No Evolutionary Algorithm (EA) - Prompt Optimization Script
"""
from typing import List
import util
import torch
import numpy as np
import os


def get_prompt_template(iteration: int, prompt_content: str, label_type: str, generate_n: int = 10) -> str:
    """
    Returns the appropriate instruction based on the iteration number range.

    Args:
        iteration: Current iteration number (0-indexed)
        prompt_content: String containing the best performing prompt pairs so far
        label_type: The dermoscopic feature being targeted
        generate_n: Number of new prompt pairs to generate

    Returns:
        String containing the iteration-specific instruction
    """
    task_specific_description_map: dict[str] = {
        "melanoma":  "shows melanoma or not",
        "pigment_network": "shows absent, typical or atypical pigment network. Negative prompts should describe both absent and typical pigment networks, while positive prompts should describe atypical pigment networks.",
        "blue_whitish_veil": "shows absence or presence of blue-whitish veil. Negative prompts should describe absence, while positive prompts should describe presence of blue-whitish veil.",
        "streaks": "shows absence, regular or irregular streaks. Negative prompts should describe both absence and regular streaks, while positive prompts should describe irregular streaks.",
        "vascular_structures": "shows absent, regular or irregular vascular structures. Negative prompts should describe all features corresponding to absent, arborizing, comma, hairpin, within regression and wreath vascular structures. Positive prompt should describe features related to both dotted and linear irregular vascular strcutures.",
        "pigmentation": "shows absent, regualr or irregular pigmentation. Negative prompts should describe all features corresponding to absent, diffuse regular and localized regular pigmentation. Positive prompts should describe diffuse and localized irregular pigmentations.",
        "dots_and_globules": "shows absent, regular or irregular dots and globules. Negative prompts should describe both absent and regular dots and globules, while positive prompts should describe irregular dots and globules.",
        "regression_structures": "shows absence or presence of regression structures. Negative prompts should describe features corresponding to absent regression structures. Positive prompts should describe features corresponding to all blue areas, white areas and combinations of regression structures.",
    }

    # Initial meta prompt for the first iteration
    meta_init_prompt = """Give 50 distinct textual descriptions of pairs of visual discriminative features to identify whether a dermoscopic image {task_specific_description}. Only provide the output as Python code in the following format: prompts = list[tuple[negative: str, positive: str]]. Let's think step-by-step"""

    # Meta prompt template for subsequent iterations
    base_meta_prompt_template = """The task is to generate distinct textual descriptions pairs of visual discriminative features to identify whether a dermoscopic image {task_specific_description}. 
    Here are the best performing pairs in ascending order. High scores indicate higher quality visual discriminative features.
    {content}
    Write {generate_n} new prompt pairs that are different from the old ones and has a score as high as possible, formulate a strategy.
    Only provide the output as Python code in the following format: prompts = list[tuple[negative: str, positive: str]]. Let's think step-by-step
    """

    # Use the initial prompt for the first iteration
    if iteration == 0:
        return meta_init_prompt.format(
            task_specific_description=task_specific_description_map[label_type],
        )

    # Use the iterative prompt for subsequent iterations
    return base_meta_prompt_template.format(
        task_specific_description=task_specific_description_map[label_type],
        content=prompt_content,
        generate_n=generate_n,
    )


def main():
    # Set the dermoscopic feature to optimize prompts for
    label_type = "pigment_network"

    # Name the experiment we are currently running
    experiment_name = "Derm7pt_Experiment8_F1_" + label_type
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

    print(f"Loaded {len(all_feats)} Derm7pt embeddings")

    # 3. Optionally load initial prompts (currently commented out)
    # initial_prompts = util.load_initial_prompts()

    # 4. Initialize the LLM client for prompt generation
    # Set use_local_ollama to True to use a local Ollama server
    client = util.LLMClient(
        use_local_ollama=False, ollama_model="hf.co/unsloth/medgemma-27b-text-it-GGUF:Q8_0")

    # Optimization loop
    pq = util.PriorityQueue(max_capacity=1000, filter_threshold=0.35)
    prompt_content = ""

    # 6. Optimization loop: generate, evaluate, and select prompts for 500 iterations
    for j in range(500):
        # Generate the meta prompt for the LLM
        meta_prompt = get_prompt_template(iteration=j,
                                          prompt_content=prompt_content, label_type=label_type, generate_n=10)

        # Generate new prompt pairs using the LLM client
        prompts = util.get_prompt_pairs(meta_prompt, client)

        # Evaluate each prompt pair and insert into the priority queue
        for i, prompt_pair in enumerate(prompts):
            if len(prompt_pair) != 2:
                print(f"Invalid prompt pair: {prompt_pair}")
                continue
            negative_prompt, positive_prompt = prompt_pair
            results = util.evaluate_prompt_pair(
                negative_prompt, positive_prompt, all_feats, all_labels, model, tokenizer)
            # Insert prompt pair and its score into the priority queue
            pq.insert((negative_prompt, positive_prompt),
                      results['f1'])

        n = 10
        print(f"\nCurrent Top {n} prompt pairs:")

        # Select top prompt pairs using roulette wheel selection
        selected_prompts = pq.get_roulette_wheel_selection(
            n, isNormalizedInts=True)

        # Sort selected prompts by score (ascending)
        selected_prompts = sorted(
            selected_prompts, key=lambda x: x[1], reverse=False)

        # Prepare the content for the next meta prompt
        prompt_content = f"Current Top {n} prompt pairs:\n"
        for i, (prompt_pair, score) in enumerate(selected_prompts):
            print(f"{i+1}. {prompt_pair}, score: {int(score)}")
            prompt_content += f"{prompt_pair}, score: {int(score)}\n"

        # Save the best prompt pairs to a file every 10 iterations (and on the first iteration)
        if (j + 1) % 10 == 0 or j == 0:
            top_prompts = pq.get_best_n(1000)
            with open(results_filename, "a") as f:
                f.write(f"Iteration {j+1}:\n")
                for prompt_pair, score in top_prompts:
                    f.write(f"{prompt_pair}, Score: {score:.4f}\n")
                f.write("\n")

        # Print the average score of the top n prompts
        print(
            f"Iteration {j+1}: mean score of top 10: {pq.get_average_score(10)}.\n")


# Entry point for the script
if __name__ == "__main__":
    main()
