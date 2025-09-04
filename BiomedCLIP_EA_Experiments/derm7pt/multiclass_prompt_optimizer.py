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
        prompt_content: String containing the best performing prompt sets so far
        label_type: The dermoscopic feature being targeted
        generate_n: Number of new prompt sets to generate

    Returns:
        String containing the iteration-specific instruction
    """

    # Initial meta prompt for the first iteration
    meta_init_prompt = """Give 20 distinct textual description sets to identify the pigment network of a dermoscopic image. Pigment Networks are labelled as absent, typical or atypical. 
Each description set must contain three distinct, contrasting features: one for an absent, one for a typical, and one for an atypical pigment network. These features should be direct, discriminating characteristics, not independent descriptions. You should NOT include any observations related to Blue Whitish Veil, Vascular Structures, Pigmentation, Streaks, Dots and Globules, Regression Structures. Only focus on the Pigment Network.
Only provide the output as Python code in the following format: prompts = list[tuple[str, str, str]]. Let's think step-by-step"""

    # Use the initial prompt for the first iteration
    if iteration == 0:
        return meta_init_prompt

    meta_optimizer_prompt = """The task is to generate distinct textual description sets to identify the pigment network of a dermoscopic image. Pigment Networks are labelled as absent, typical or atypical. 
Each description set must contain three distinct, contrasting features: one for an absent, one for a typical, and one for an atypical pigment network. These features should be direct, discriminating characteristics, not independent descriptions. You should only describe the pigment network aspect of the dermoscopic image.
Here are the best performing sets in ascending order. High scores indicate higher quality visual discriminative features.
{content}
{iteration_specific_instruction}
Only provide the output as Python code in the following format: prompts = list[tuple[str, str, str]]. Think step-by-step.
"""

    iteration_specific_instruction = """Write {generate_n} new prompt sets that are different to from the old ones and has a score as high as possible. Formulate a strategy. Let's think step-by-step."""

    prompt = meta_optimizer_prompt.format(
        content=prompt_content,
        iteration_specific_instruction=iteration_specific_instruction.format(
            generate_n=10)
    )
    # Use the iterative prompt for subsequent iterations
    return prompt


def main():
    # Set the dermoscopic feature to optimize prompts for
    label_type = "pigment_network"

    # Name the experiment we are currently running
    experiment_name = "Derm7pt_Experiment10_Multiclass_" + label_type
    print(f"Running {experiment_name}...")

    # Create experiment results directory
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)

    # Create filename with experiment name
    results_filename = os.path.join(
        results_dir, f"{experiment_name}_opt_sets.txt")

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
    pq = util.PriorityQueue(max_capacity=1000, filter_threshold=0.5)
    prompt_content = ""

    # 6. Optimization loop: generate, evaluate, and select prompt sets for 500 iterations
    for j in range(500):
        # Generate the meta prompt for the LLM
        meta_prompt = get_prompt_template(iteration=j,
                                          prompt_content=prompt_content, label_type=label_type, generate_n=10)

        # Generate new prompt sets using the LLM client
        prompt_sets = util.get_prompts_from_llm(meta_prompt, client)

        # Evaluate each prompt set and insert into the priority queue
        for i, prompt_set in enumerate(prompt_sets):
            if len(prompt_set) != 3:
                print(f"Invalid prompt set: {prompt_set}")
                continue
            results = util.evaluate_prompt_set(
                prompt_set, all_feats, all_labels, model, tokenizer)
            # Insert prompt set and its score into the priority queue
            # Use accuracy as the score
            pq.insert(prompt_set, results['accuracy'])

        n = 10
        print(f"\nCurrent Top {n} prompt sets:")

        # Select top prompt sets using roulette wheel selection
        selected_prompts = pq.get_roulette_wheel_selection(
            n, isNormalizedInts=True)

        # Sort selected prompts by score (ascending)
        selected_prompts = sorted(
            selected_prompts, key=lambda x: x[1], reverse=False)

        # Prepare the content for the next meta prompt
        prompt_content = f"Current Top {n} prompt sets:\n"
        for i, (prompt_set, score) in enumerate(selected_prompts):
            print(f"{i+1}. {prompt_set}, score: {int(score)}")
            prompt_content += f"{prompt_set}, score: {int(score)}\n"

        # Save the best prompt sets to a file every 10 iterations (and on the first iteration)
        if (j + 1) % 10 == 0 or j == 0:
            top_prompts = pq.get_best_n(1000)
            with open(results_filename, "a") as f:
                f.write(f"Iteration {j+1}:\n")
                for prompt_set, score in top_prompts:
                    f.write(f"{prompt_set}, Score: {score:.4f}\n")
                f.write("\n")

        # Print the average score of the top n prompt sets
        print(
            f"Iteration {j+1}: mean score of top 10: {pq.get_average_score(10)}.\n")


# Entry point for the script
if __name__ == "__main__":
    main()
