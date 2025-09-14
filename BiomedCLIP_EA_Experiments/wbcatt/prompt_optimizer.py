"""
Optimization Only - No Evolutionary Algorithm (EA) - Prompt Optimization Script


# medical concepts
These are the following features an expert would look for: Cell Size, Cell Shape, Nucleus Shape, Nuclear-Cytoplasmic Ratio, Chromatin-Density, Cytoplasm-Vacuole, Cytoplasm-Texture, Cytoplasm-Color, Granule-Type, Granule-Color, Granularity
Basophils: Nucleus=Segmented, NC Ratio=Low, Granularity=Yes, Color=Blue/Black (dense), Size=-
Eosinophils: Nucleus=Segmented, NC Ratio=Low, Granularity=Yes, Color=Red, Size=-
Lymphocytes: Nucleus=Unsegmented, NC Ratio=High, Granularity=No, Color=-, Size=Small
Monocytes: Nucleus=Unsegmented, NC Ratio=Low, Granularity=No, Color=-, Size=-
Neutrophils: Nucleus=Segmented, NC Ratio=Low, Granularity=Yes, Color=Blue, Size=-
Each description set must contain five discriminating meaningful descriptions to identify each of the five cell types.

"""
from typing import List
import util
import torch
import numpy as np
import os

# 'accuracy', 'auc', 'f1_macro', 'inverted_weighted_ce'
FITNESS_METRIC = 'f1_macro'


def get_prompt_template(iteration: int, prompt_content: str, generate_n: int = 8) -> str:
    """
    Returns the appropriate instruction based on the iteration number range.

    Args:
        iteration: Current iteration number (0-indexed)
        prompt_content: String containing the best performing prompt pairs so far
        generate_n: Number of new prompt pairs to generate

    Returns:
        String containing the iteration-specific meta_prompt
    """

    # Initial meta prompt for the first iteration
    meta_init_prompt = """Give 50 distinct textual description sets of visual discriminative features to identify {task_specific_description}.
Only provide the output as Python code in the following format: prompts = list[tuple[str, ...]]. Let's think step-by-step"""

    # Meta prompt template for subsequent iterations
    base_meta_prompt_template = """The task is to generate distinct textual descriptions pairs of visual discriminative features to identify {task_specific_description}. 
Here are the best performing sets in ascending order. High scores indicate higher quality visual discriminative features.
{content}
Write {generate_n} new descriptions sets that are different from the old ones and has a score as high as possible, formulate a strategy.
Only provide the output as Python code in the following format: prompts = list[tuple[str, ...]. Let's think step-by-step
"""

    task_specific_description = """Basophil, Eosinophil, Lymphocyte, Monocyte, and Neutrophil peripheral blood cells.
Format: <Features describing Basophil>, <Features describing Eosinophil>, <Features describing Lymphocyte>, <Features describing Monocyte>, <Features describing Neutrophil> 
    """
    # Use the initial prompt for the first iteration
    if iteration == 0:
        return meta_init_prompt.format(
            task_specific_description=task_specific_description,
        )

    # Use the iterative prompt for subsequent iterations
    return base_meta_prompt_template.format(
        task_specific_description=task_specific_description,
        content=prompt_content,
        generate_n=generate_n,
    )


def main():

    # Name the experiment we are currently running
    experiment_name = f"Wbcatt_Experiment6_{FITNESS_METRIC}-Gen8-AllFeatures"
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
    )

    # Convert to tensors - MODIFIED FOR MULTI-OBSERVATION SUPPORT
    all_feats = torch.from_numpy(features).float()
    all_labels = torch.from_numpy(labels).long()

    print(f"Loaded {len(all_feats)} wbcatt embeddings")

    # 3. Optionally load initial prompts (currently commented out)
    # initial_prompts = util.load_initial_prompts()

    # 4. Initialize the LLM client for prompt generation
    # Set use_local_ollama to True to use a local Ollama server
    client = util.LLMClient(provider="Gemini")

    # Optimization loop
    pq = util.PriorityQueue(max_capacity=1000, filter_threshold=0.1)
    prompt_content = ""

    # 6. Optimization loop: generate, evaluate, and select prompts for 500 iterations
    for j in range(500):
        # Generate the meta prompt for the LLM
        meta_prompt = get_prompt_template(iteration=j,
                                          prompt_content=prompt_content, generate_n=8)

        # Generate new prompt sets using the LLM client
        prompt_sets = util.get_prompts_from_llm(meta_prompt, client)

        # Evaluate each prompt set and insert into the priority queue
        for i, prompt_set in enumerate(prompt_sets):
            if len(prompt_set) != 5:
                print(f"Invalid prompt set: {prompt_set}")
                continue
            results = util.evaluate_prompt_set(
                prompt_set, all_feats, all_labels, model, tokenizer)
            # Insert prompt set and its score into the priority queue
            # Use accuracy as the score
            pq.insert(prompt_set, results[FITNESS_METRIC])

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
            f"Iteration {j+1}: mean {FITNESS_METRIC} of top 10: {pq.get_average_score(10)}.\n")


# Entry point for the script
if __name__ == "__main__":
    main()
