import util
from gemini import Gemini

import re
import ast
from typing import List, Any


def extract_bracketed_content(text: str) -> str:
    """
    Extracts the first top-level bracketed expression from `text`.

    E.g. given:
        "... Final Prompt Pair:\n[ [\"a\",\"b\"], [\"c\",\"d\"] ]\n"
    returns:
        '[ ["a","b"], ["c","d"] ]'

    Raises:
        ValueError if no bracketed expression is found.
    """
    # This regex looks for a '[' followed by anything until the matching ']' at the same nesting level.
    # Simplest: grab from the first '[' to the last ']' in the string.
    text = text.strip()
    # print(f"Extracting bracketed content from: {text}")
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No bracketed content found")
    return text[start:end+1]


def parse_func(input_string: str) -> List[List[Any]]:
    return util.convert_string_to_list_of_tuples(extract_bracketed_content(input_string))


def main():
    # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()

    # 2. load dataset
    centers_features, centers_labels = util.extract_center_embeddings(
        model=model, preprocess=preprocess)

    # 3. load initial prompts (optional)
    # initial_prompts = util.load_initial_prompts()

    cookies = {"__Secure-1PSIDCC": "AKEyXzUXJhsXJGTmxBCwbU67Vvu2YBLU-k7c-M4WHGl1O-feiLULrQJMWuXMQ5SyT3Sy3FruT0Qk",
               "__Secure-1PSID": "g.a000xAh5UmC2BgFDEL7ifghVXxgaGkKxUI_E7SU8c6KeTfk4KXuyOMwOCWull1Ay_77sjDJF-QACgYKAR4SARQSFQHGX2MiOcqyvE84Gino-n2jvqMh-BoVAUF8yKqEYwQ1MWNHeeRQHWA3kX7b0076",
               "__Secure-1PSIDTS": "sidts-CjIB5H03P1rfoBF445cLXd8NgMdu3avc4_52bHJNd9nd5GCvCH_WxhlS7Yx8BRQX-J1mzxAA",
               }  # Cookies may vary by account or region. Consider sending the entire cookie file.

    client = Gemini(auto_cookies=False, cookies=cookies)

    meta_init_prompt = "Give 50 textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section"
    META_PROMPT_TEMPLATE = """\
            Please follow the instruction step-by-step to generate a better prompt pair.

            1. Cross over the following prompts and generate a new prompt:

            Prompt Pair 1: {pair1}
            Prompt Pair 2: {pair2}

            2. Mutate the prompt generated in Step 1 and generate a final prompt pair in brackets [[], []]
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
                meta_prompt, client, parse_func=parse_func)

        for i, prompt_pair in enumerate(prompts):
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
