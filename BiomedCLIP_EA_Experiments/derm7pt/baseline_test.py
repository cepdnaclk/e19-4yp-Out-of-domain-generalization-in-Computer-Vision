import torch
import util


def main():
    # Set the dermoscopic feature to optimize prompts for
    label_type = "melanoma"

    # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. load dataset - MODIFIED FOR CHEXPERT
    features, labels = util.extract_embeddings(
        model=model,
        preprocess=preprocess,
        split="test",
        label_type=label_type,
    )

    # Convert to tensors - MODIFIED FOR MULTI-OBSERVATION SUPPORT
    all_feats = torch.from_numpy(features).float()
    all_labels = torch.from_numpy(labels).long()

    print(f"Loaded {len(all_feats)} Derm7pt embeddings")
    print("all_labels:\n", all_labels)

    prompt_set = (
        "A dermoscopic image of a skin lesion showing benign features",
        "A dermoscopic image of a skin lesion showing melanoma",
    )

    results = util.evaluate_prompt_set(
        prompt_set, all_feats, all_labels, model, tokenizer)
    print("Baseline Evaluation results:\n", results)

    optimized_prompt_set = (
        'Regression structures are minimal, appearing as subtle perifollicular hypopigmentation.', 'Regression structures are extensive, with large areas of depigmentation and scarring, resembling a moth-eaten appearance.'
    )

    optimized_results = util.evaluate_prompt_set(
        optimized_prompt_set, all_feats, all_labels, model, tokenizer)
    print("Optimized Evaluation results:\n", optimized_results)


if __name__ == "__main__":
    main()
