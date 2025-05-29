import util
from gemini import Gemini

import re
import ast
from typing import List, Any


def main():
    # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. load dataset
    centers_features, centers_labels = util.extract_center_embeddings(
        model=model, preprocess=preprocess)
    print("Center embeddings extracted successfully.")

    # 3. load initial prompts (optional)
    # initial_prompts = util.load_initial_prompts()

    cookies = {"__Secure-1PSIDCC": "AKEyXzUqRjIED_nVsfkjfc4gCNP8gZVlIJSgP9nvYOwyyInbBLfR9fDrYFKJr4X_XrCYp5H1OVt4",
               "__Secure-1PSID": "g.a000xAh5UmC2BgFDEL7ifghVXxgaGkKxUI_E7SU8c6KeTfk4KXuyOMwOCWull1Ay_77sjDJF-QACgYKAR4SARQSFQHGX2MiOcqyvE84Gino-n2jvqMh-BoVAUF8yKqEYwQ1MWNHeeRQHWA3kX7b0076",
               "__Secure-1PSIDTS": "sidts-CjIB5H03P9NCn7cjO4YMFZj55r7pAOOhothT61h_hlTVUqzmlIZyTh6qFSgvHDb1LFGbshAA",
               }  # Cookies may vary by account or region. Consider sending the entire cookie file.

    client = Gemini(auto_cookies=False, cookies=cookies)
    print("Gemini client initialized successfully.")

    # 4. Define the meta prompt and template
    meta_init_prompt = """Give 50 textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section. Only give the output as python code in the format - prompts: list[tuple[negative: str, positive: str]]"""
    META_PROMPT_TEMPLATE = """\
            Please follow the instruction step-by-step to generate a better prompt pair.

            1. Cross over the following prompts and generate a new prompt:

            Prompt Pair 1: {pair1}
            Prompt Pair 2: {pair2}

            2. Mutate the prompt generated in Step 1 and generate a final prompt pair in a python tuple (str, str)
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
            prompts = util.get_prompt_pairs(
                meta_prompt, client, parse_func=util.extract_and_parse_prompt_tuple)

        for i, prompt_pair in enumerate([prompts]):
            if len(prompt_pair) != 2:
                print(f"Invalid prompt pair: {prompt_pair}")
                continue
            negative_prompt, positive_prompt = prompt_pair
            results = util.evaluate_prompt_pair(
                negative_prompt, positive_prompt, centers_features[0], centers_labels[0], model, tokenizer)
            pq.insert((negative_prompt, positive_prompt), results['accuracy'])

        n = 2
        print(f"\Selectd {n} prompt pairs:")
        roulette = pq.get_roulette_wheel_selection(n)
        meta_prompt = META_PROMPT_TEMPLATE.format(
            pair1=roulette[0], pair2=roulette[1])

        for i, (prompt_pair, score) in enumerate(meta_prompt):
            print(f"{i+1}. {prompt_pair}, Score: {score:.4f}")

        # Save the best prompt pairs to a file
        with open("EA_only.txt", "a") as f:
            f.write(f"Iteration {j+1}:\n")
            for prompt_pair, score in roulette:
                f.write(f"{prompt_pair}, Score: {score:.4f}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
