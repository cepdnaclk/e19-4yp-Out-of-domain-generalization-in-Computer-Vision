import util
from gemini import Gemini


def main():
    # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()

    # 2. load dataset
    centers_features, centers_labels = util.extract_center_embeddings(
        model=model, preprocess=preprocess)

    # 3. load initial prompts (optional)
    # initial_prompts = util.load_initial_prompts()

    cookies = {"__Secure-1PSIDCC": "8WqUIAmsCWWrmWr-/AqzGpTdQDEvsWgOSP",
               "__Secure-1PSID": "g.a000xAhtcFFJw-Pe2SfxFzHOJXUMClrKicX6q_b7mFELwJZbSoGutGYNkxA8kyX1FZpLmh29jwACgYKAXESARASFQHGX2Mi7J2NGrnruG68cQI02g7H6BoVAUF8yKqFxE1MZio3JvWDuqqc2aS90076",
               "__Secure-1PSIDTS": "AKEyXzW9DtAugRds_seZfS4OUpDvWkPzJFmyEjYz-Ytr-zQpaQ_8j4Ujce8w5aN4HjfI7Erxnmae",
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

    for j in range(1000):
        if j == 0:
            prompts = util.get_prompt_pairs(meta_init_prompt, client)
