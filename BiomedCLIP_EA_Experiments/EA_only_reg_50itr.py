"""
 Evolutionary Algorithm (GA) Only with regularization- No pd- No Scores for optimization
"""
from API_KEY import GEMINI_API_KEY
from google import genai
import util
import torch
import re
import ast
from typing import List, Any
import numpy as np
import os


def main():
    # Name the experiment we are currently running
    experiment_name = "EA-Only-reg-with-1000-iterations"
    print(f"Running {experiment_name}...")

    # Create experiment results directory
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)

    # Create filename with experiment name
    results_filename = os.path.join(
        results_dir, f"{experiment_name}_pairs.txt")

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
        num_centers=1,  # trained only on center 0
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

    client = genai.Client(api_key=GEMINI_API_KEY)
    print("Gemini client initialized successfully.")

    # 4. Define the meta prompt and template
    meta_init_prompt = """Give 50 textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. \
                The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section. \
                Only give the output as python code in the format - prompts: list[tuple[negative: str, positive: str]]"""
    META_PROMPT_TEMPLATE = """\
            Please follow the instruction step-by-step to generate a better prompt pair for the above task.
            Following exemplar shows how to write the prompt pair: \
                    ("Tumor is not present in this image", "This is an image of a tumor") \
            1. Cross over the following prompts and generate a new prompt:

            Prompt Pair 1: {pair1}
            Prompt Pair 2: {pair2}

            2. Mutate the prompt generated in Step 1 keeping the word count under 20 for each prompt and generate a final prompt pair in a python tuple (str, str)
    """

    # initial_list = load_initial_prompts("selected_prompts.txt")
    # pq = PriorityQueue(max_capacity=40, initial=initial_list)
    pq = util.PriorityQueue(max_capacity=1000)

    meta_prompt = ""
    for j in range(1000):
        if j == 0:
            prompts = util.get_prompt_pairs(
                meta_init_prompt, client)
        else:
            prompts = [util.get_prompt_pairs(
                meta_prompt, client, parse_func=util.extract_and_parse_prompt_tuple)]  # wrapped in a list to make it a list of prompt pairs

        for i, prompt_pair in enumerate(prompts):
            if len(prompt_pair) != 2:
                print(f"Invalid prompt pair: {prompt_pair}")
                continue
            negative_prompt, positive_prompt = prompt_pair
            results = util.evaluate_prompt_pair(
                negative_prompt, positive_prompt, all_feats, all_labels, model, tokenizer)
            print(
                f"Iteration {j+1}, New Prompt Pair {i+1}: {negative_prompt}, {positive_prompt}, Accuracy: {results['accuracy']:.4f}")
            pq.insert((negative_prompt, positive_prompt), results['accuracy'])

        n = 2
        print(f"Selected {n} prompt pairs:")
        roulette = pq.get_roulette_wheel_selection(n)
        meta_prompt = META_PROMPT_TEMPLATE.format(
            pair1=roulette[0], pair2=roulette[1])

        for i, (prompt_pair, score) in enumerate(roulette):
            print(f"{i+1}. {prompt_pair}, Score: {score:.4f}")

        # Save the best prompt pairs to a file
        top_prompts = pq.get_best_n(100)
        with open(results_filename, "a") as f:
            f.write(f"Iteration {j+1}:\n")
            for prompt_pair, score in top_prompts:
                f.write(f"{prompt_pair}, Score: {score:.4f}\n")
            f.write("\n")

        print(
            f"Iteration {j+1}: mean accuracy of top 10: {pq.get_average_score(10)}.\n")


if __name__ == "__main__":
    main()
