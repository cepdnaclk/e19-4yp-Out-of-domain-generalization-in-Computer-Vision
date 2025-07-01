"""
Optimization Only - No Evolutionary Algorithm (EA) - Prompt Optimization Script
"""
from typing import List
import util
import torch
import numpy as np
import os
# from chatgpt_initial import INITIAL_CHATGPT_PROMPTS


def main():
    # Name the experiment we are currently running
    experiment_name = "MultiClass-Experiment-1-gemma3"
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
    # Set use_local_ollama to True if you want to use a local Ollama server
    client = util.LLMClient(
        use_local_ollama=False, ollama_model="hf.co/unsloth/medgemma-27b-text-it-GGUF:Q8_0")

    # Configure the prompt templates
    meta_init_prompt = """The task is to generate 50 distinct textual description prompts of visual discriminative features to describe the central region of an histopathological image patch. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section. The patch can be either tumor or normal tissue.
Each prompt should have a placeholder <> that can be replaced with either "tumor" or "normal". I will fill in the placeholder later.

Eg: An histopathological image of a <>.
Output as python code in the format - prompts: list[prompt:str]. Let's think step-by-step
"""

    meta_optimizer_prompt_template = """The task is to generate distinct textual description prompts of visual discriminative features to describe the central region of an histopathological image patch. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section. The patch can be either tumor or normal tissue.
Each prompt should have a placeholder <> that can be replaced with either "tumor" or "normal". I will fill in the placeholder later.
Here are the best performing pairs in ascending order. High scores indicate higher quality visual discriminative features.
{content}

Write 10 new prompts that are different to from the old ones and has a score as high as possible. Formulate a strategy
Output as python code in the format - prompts: list[prompt:str]. Let's think step-by-step"""

    # Optimization loop
    # initial_prompts = util.load_initial_prompts(
    #     "experiment_results/medical_concepts.txt")
    pq = util.PriorityQueue(max_capacity=1000, filter_threshold=0.6)
    prompt_content = ""

    for j in range(1000):
        if j == 0:
            prompts = util.get_prompt_pairs(meta_init_prompt, client)
            # prompts = INITIAL_CHATGPT_PROMPTS
        else:
            meta_prompt = meta_optimizer_prompt_template.format(
                content=prompt_content)

            prompts = util.get_prompt_pairs(meta_prompt, client)

        for i, prompt_pair in enumerate(prompts):
            if len(prompt_pair) != 2:
                print(f"Invalid prompt pair: {prompt_pair}")
                continue
            negative_prompt, positive_prompt = prompt_pair
            results = util.evaluate_prompt_pair(
                negative_prompt, positive_prompt, all_feats, all_labels, model, tokenizer)
            pq.insert((negative_prompt, positive_prompt),
                      results['accuracy'])

        n = 10
        print(f"\nCurrent Top {n} prompt pairs:")

        # Selector Operator: Roulette Wheel Selection or Best N Prompts
        # Use Roulette Wheel Selection for the first 250 iterations
        # After 250 iterations, use the best n prompts
        # This is to ensure diversity in the early stages of optimization
        # if j < 250:
        #     selected_prompts = pq.get_roulette_wheel_selection(
        #         n, isNormalizedInts=False)
        # else:
        #     selected_prompts = pq.get_best_n(n)

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
            f"Iteration {j+1}: mean accuracy of top 10: {pq.get_average_score(10)}.\n")


if __name__ == "__main__":
    main()
