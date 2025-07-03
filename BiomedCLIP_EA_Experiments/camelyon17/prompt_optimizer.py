"""
Optimization Only - No Evolutionary Algorithm (EA) - Prompt Optimization Script
"""
from typing import List
import util
import torch
import numpy as np
import os
# from chatgpt_initial import INITIAL_CHATGPT_PROMPTS


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
