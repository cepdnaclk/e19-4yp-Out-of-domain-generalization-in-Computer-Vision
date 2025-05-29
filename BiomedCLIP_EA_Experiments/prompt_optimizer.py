from util import *
from gemini import Gemini


def main():
    # load model from util
    model, preprocess, tokenizer = load_clip_model()

    # load features and labels 
    centers_features,centers_labels = extract_center_embeddings(model=model, preprocess=preprocess)
    
    # Configure Gemini
    cookies = {"__Secure-1PSIDCC": "8WqUIAmsCWWrmWr-/AqzGpTdQDEvsWgOSP",
               "__Secure-1PSID": "g.a000xAhtcFFJw-Pe2SfxFzHOJXUMClrKicX6q_b7mFELwJZbSoGutGYNkxA8kyX1FZpLmh29jwACgYKAXESARASFQHGX2Mi7J2NGrnruG68cQI02g7H6BoVAUF8yKqFxE1MZio3JvWDuqqc2aS90076",
               "__Secure-1PSIDTS": "AKEyXzW9DtAugRds_seZfS4OUpDvWkPzJFmyEjYz-Ytr-zQpaQ_8j4Ujce8w5aN4HjfI7Erxnmae",
               }  # Cookies may vary by account or region. Consider sending the entire cookie file.

    client = Gemini(auto_cookies=False, cookies=cookies)


    # Configure the prompt templates
    meta_prompt_template_initial = """Give 50 textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section.
                                    Each description should be about 5-20 words.
                                    Only give the output as [(negative prompt,positive prompt),...]"""
                       

    meta_prompt_template = """Give 50 textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section.
                       Each description should be about 5-20 words.
                       Only give the output as [(negative prompt,positive prompt).
                       Here are the best performing pairs. You should aim to get higher scores.
                       {content}
                       """
    
                       
                       
    

    # Optimization loop
    pq = PriorityQueue(max_capacity=1000)
    prompt_llm = ""
    for j in range(100):
        if j==0:
            prompts = get_prompt_pairs(meta_prompt_template_initial, client)
        else:
            prompts = get_prompt_pairs(meta_prompt_template, client)

        for i, prompt_pair in enumerate(prompts):
            if len(prompt_pair) != 2:
                print(f"Invalid prompt pair: {prompt_pair}")
                continue
            negative_prompt, positive_prompt = prompt_pair
            results = evaluate_prompt_pair(
                negative_prompt, positive_prompt, centers_features[0], centers_labels[0], model, tokenizer)
            
            
            pq.insert((negative_prompt, positive_prompt), results['accuracy'])

        n = 10
        print(f"\nCurrent Top {n} prompt pairs:")
        selected_prompts = pq.get_roulette_wheel_selection(10)
        # top_n = pq.get_best_n(n)
        prompt_llm = f"Current Top {n} prompt pairs:\n"
        for i, (prompt_pair, score) in enumerate(selected_prompts):
            print(f"{i+1}. {prompt_pair}, Score: {score:.4f}")
            prompt_llm += f"{i+1}. {prompt_pair}, Score: {score:.4f}\n"

        # Save the best prompt pairs to a file
        with open("selected_prompt_pairs.txt", "a") as f:
            f.write(f"Iteration {j+1}:\n")
            for prompt_pair, score in selected_prompts:
                f.write(f"{prompt_pair}, Score: {score:.4f}\n")
            f.write("\n")






if __name__ == "__main__":
    main()
