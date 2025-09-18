"""
Optimization Only - No Evolutionary Algorithm (EA) - Prompt Optimization Script
"""
import sys
from typing import List
import util
import torch
import numpy as np
import os

FITNESS_METRIC = "inverted_bce"
FEW_SHOT = 8  # Default value, will be overridden by command line
CLASS = "Pneumonia"  # Default value, will be overridden by command line


def get_prompt_template(iteration_num: int, prompt_content: str, generate_n: int = 10) -> str:
    """
    Returns the appropriate instruction based on the iteration number range.

    Args:
        iteration_num: Current iteration number (1-indexed)

    Returns:
        String containing the iteration-specific instruction

    """
    # Initial meta prompt for the first iteration
    meta_init_prompt = f"""Generate 50 distinct pairs of textual descriptions of visual discriminative features to identify whether a chest X-ray image shows {CLASS} or not.
    Negative examples may display signs of other diseases such as Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia, but should not indicate {CLASS}.
    Positive examples should clearly indicate {CLASS}.
    Only provide the output as Python code in the following format: prompts = list[tuple[negative: str, positive: str]].
    Let's think step-by-step."""

    if iteration_num == 0:
        return meta_init_prompt

    # Base meta prompt template
    base_meta_prompt_template = """The task is to generate distinct textual descriptions pairs of visual discriminative features to identify whether a chest X-ray image shows {disease} or not. 
    Negative examples may display signs of other diseases such as Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia, but should not indicate {CLASS}.
    Positive examples should clearly indicate {CLASS}.    
    Here are the best performing pairs in ascending order. High scores indicate higher quality visual discriminative features.
    {content}
    Write {generate_n} new prompt pairs that are different from the old ones and has a score as high as possible, formulate a strategy
    Only provide the output as Python code in the following format: prompts = list[tuple[negative: str, positive: str]]. Let's think step-by-step
    """

    # Iterations 1-50: Basic exploration
    return base_meta_prompt_template.format(
        content=prompt_content,
        generate_n=generate_n,
        disease=CLASS,
    )


def main():
    # Name the experiment we are currently running
    experiment_name = f"NIH_Train_Experiments-n{FEW_SHOT}_{CLASS}"
    print(f"Running {experiment_name}...")

    # Create experiment results directory
    results_dir = "final_results"
    os.makedirs(results_dir, exist_ok=True)

    # Create filename with experiment name
    results_filename = os.path.join(
        results_dir, f"{experiment_name}.txt")

   # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. load dataset - MODIFIED FOR CHEXPERT
    features, labels = util.extract_embeddings(
        model=model,
        preprocess=preprocess,
        train_or_test="train",
        target_label=CLASS,
    )

    # Convert to tensors - MODIFIED FOR MULTI-OBSERVATION SUPPORT
    all_feats = torch.from_numpy(features).float()
    all_labels = torch.from_numpy(labels).long()

    if FEW_SHOT > 0:
        # Select a balanced few-shot subset
        all_feats, all_labels = util.select_balanced_few_shot_subset(
            all_feats, all_labels, n_per_class=FEW_SHOT)
        print(
            f"Selected balanced few-shot subset with {FEW_SHOT} samples per class.")

    print(f"Class distribution: {torch.bincount(all_labels)}")

    # print(f"shape of all_feats: {all_feats.shape} and all_labels: {all_labels.shape}")

    print(f"Loaded {len(all_feats)} NIHChestXrays embeddings")

    # 3. load initial prompts (optional)
    # initial_prompts = util.load_initial_prompts()

    # 4. Initialize the LLM client
    # Set use_local_ollama to True if you want to use a local Ollama server
    client = util.LLMClient(
        use_local_ollama=False, ollama_model="hf.co/unsloth/medgemma-27b-text-it-GGUF:Q8_0")

    pq = util.PriorityQueue(max_capacity=1000, filter_threshold=0.5)
    prompt_content = ""

    for j in range(500):

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
            # print(f"Inverted BCE for prompt pair {i+1}: {results['inverted_bce']:.4f} {results['accuracy']}")
            pq.insert((negative_prompt, positive_prompt),
                      results[FITNESS_METRIC])

        n = 10
        print(f"\nCurrent Top {n} prompt pairs:")

        selected_prompts = pq.get_roulette_wheel_selection(
            n, isNormalizedInts=True)
        # selected_prompts = pq.get_best_n(n)
        # reverse the order to set it to acsending order: Recency Bias
        selected_prompts = sorted(
            selected_prompts, key=lambda x: x[1], reverse=False)

        # Prepare the content for the meta prompt
        prompt_content = f"Current Top {n} prompt pairs:\n"
        for i, (prompt_pair, score) in enumerate(selected_prompts):
            print(f"{i+1}. {prompt_pair}, Score: {score}")
            # prompt_content += f"{i+1}. {prompt_pair}, Score: {score:.2f}\n"
            # for ascending order
            prompt_content += f"{prompt_pair}, Score: {score:.2f}\n"

        # Save the best prompt pairs to a file, every 10 iterations
        if (j + 1) % 10 == 0 or j == 0:
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
