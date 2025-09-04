import torch
import util


def main():
    # Set the dermoscopic feature to optimize prompts for
    label_type = "pigment_network"

    # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. load dataset - MODIFIED FOR CHEXPERT
    features, labels = util.extract_embeddings(
        model=model,
        preprocess=preprocess,
        split="train",
        label_type=label_type,
    )

    # Convert to tensors - MODIFIED FOR MULTI-OBSERVATION SUPPORT
    all_feats = torch.from_numpy(features).float()
    all_labels = torch.from_numpy(labels).long()

    print(f"Loaded {len(all_feats)} Derm7pt embeddings")
    print("all_labels:\n", all_labels)

    prompt_set = (
        "A dermoscopic image showing no signs of a pigment network.",
        "A dermoscopic image showing a normal pigment network.",
        "A dermoscopic image showing an atypical pigment network.",
    )
    results = util.evaluate_prompt_set(
        prompt_set, all_feats, all_labels, model, tokenizer)
    print("Evaluation results:\n", results)


if __name__ == "__main__":
    main()
