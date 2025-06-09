"""
Optimization Only - This script optimizes the instruction prompt for generating visual discriminative features in histopathological images.
"""
from typing import List
import util
import torch
import numpy as np
import os
import re
import time
# from chatgpt_initial import INITIAL_CHATGPT_PROMPTS


def get_meta_instruction(prompt: str, llm_client: util.LLMClient, max_retries: 10) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            # Use the unified LLMClient to get the raw response
            raw = llm_client.get_llm_response(prompt)
            # print(f"Raw response on attempt {attempt}: {raw}...")

            # Extract the instruction from the raw response from <instruction> tags
            # let's use regex to extract the instruction
            match = re.search(
                r'<instruction>(.*?)</instruction>', raw, re.DOTALL)
            if match:
                instruction = match.group(1).strip()
                print(
                    f"Extracted instruction on attempt {attempt}: {instruction}")
                return instruction
            else:
                print(
                    f"No <instruction> tags found in response on attempt {attempt}. Retrying...")
        except Exception as e:
            print(f"Error on attempt {attempt}: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
            if attempt == max_retries:
                print("Max retries reached. Returning empty instruction.")
                return ""


def main():
    # Name the experiment we are currently running
    experiment_name = "Experiment-35-meta-prompt-gemma3"
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

    generate_n = 10  # Number of new prompt pairs to generate in each iteration
    # Configure the prompt templates
    meta_init_prompt = """Give 50 distinct textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section. Only give the output as python code in the format - prompts: list[tuple[negative: str, positive: str]]"""

    meta_prompt_template = """The task is to generate distinct textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section.
    Here are the best performing pairs in ascending order. High scores indicate higher quality visual discriminative features.
    {content}
    {instruction}
    Output as python code in the format - prompts: list[tuple[negative: str, positive: str]]. Let's think step-by-step,
    """

    intstruction_optimizer_template = """The task is to generate distinct textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section.
    Here are the best performing pairs in ascending order. High scores indicate higher quality visual discriminative features.
    {content}

    Last 10 iterations improvement: 0
    Last Instruction: {instruction}

    This instruction will be used in the following setting:
    "\{\{Problem Description and task to be performed\}\}
    \{\{Current Top 10 Prompts\}\}>
    <Instruction>"

    Example Instructions:
    Write 10 new prompt pairs that are different from the old ones and has a score as high as possible.
    Write 10 new prompt pairs that are more similar to the high scoring prompts
    Write 10 new prompt pairs by paraphrasing each of the above. Each pair should have distinct language style.
    Write 10 new prompt pairs appending rare or borderline patterns which are easily misclassified to score as high as possible.

    Write a short generalized instruction to improve the score. Avoid any specific characteristics in the instruction, as the current top prompts will be provided.
    Output in a <instruction> tag. Let's think step-by-step,
    """

    # Optimization loop
    # initial_prompts = util.load_initial_prompts(
    #     "experiment_results/medical_concepts.txt")
    pq = util.PriorityQueue(max_capacity=1000)
    prompt_content = ""
    current_instruction = f"Write {generate_n} new prompt pairs that are different from the old ones and has a score as high as possible."
    current_score = 0.0
    for j in range(1000):
        if j == 0:
            prompts = util.get_prompt_pairs(meta_init_prompt, client)
            # prompts = INITIAL_CHATGPT_PROMPTS
        else:
            meta_prompt = meta_prompt_template.format(
                content=prompt_content, instruction=current_instruction)
            prompts = util.get_prompt_pairs(meta_prompt, client)

        for i, prompt_pair in enumerate(prompts):
            if len(prompt_pair) != 2:
                print(f"Invalid prompt pair: {prompt_pair}")
                continue
            negative_prompt, positive_prompt = prompt_pair
            results = util.evaluate_prompt_pair(
                negative_prompt, positive_prompt, all_feats, all_labels, model, tokenizer)

            pq.insert((negative_prompt, positive_prompt), results['accuracy'])

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
            print(f"{i+1}. {prompt_pair}, Score: {score}")
            prompt_content += f"{prompt_pair}, Score: {score:.2f}\n"

        # Save the best prompt pairs to a file, every 5 iterations
        if (j + 1) % 5 == 0 or j == 0:
            top_prompts = pq.get_best_n(1000)
            with open(results_filename, "a") as f:
                f.write(f"Iteration {j+1}:\n")
                for prompt_pair, score in top_prompts:
                    f.write(f"{prompt_pair}, Score: {score:.4f}\n")
                f.write("\n")
            new_score = pq.get_average_score(20)

            # if the score has not improved, then we will update the instruction
            if new_score - current_score < 0.0001:
                print(
                    f"Score has not improved from {current_score:.4f}. Updating instruction.")
                current_instruction = get_meta_instruction(
                    prompt=intstruction_optimizer_template.format(
                        content=prompt_content, instruction=current_instruction),
                    llm_client=client,
                    max_retries=10
                )

            else:
                print(
                    f"Score has improved from {current_score:.4f} to {new_score:.4f}. Keeping the current instruction.")
            current_score = new_score

        # print the average score of the top n prompts
        print(
            f"Iteration {j+1}: mean accuracy of top 10: {pq.get_average_score(10)}.\n")


if __name__ == "__main__":
    main()
