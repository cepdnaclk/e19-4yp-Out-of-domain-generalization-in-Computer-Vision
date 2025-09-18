import torch
import util


def main():
   # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. load dataset - MODIFIED FOR CHEXPERT
    features, labels = util.extract_embeddings(
        model=model,
        preprocess=preprocess,
        split="test",
    )

    # Convert to tensors - MODIFIED FOR MULTI-OBSERVATION SUPPORT
    all_feats = torch.from_numpy(features).float()
    all_labels = torch.from_numpy(labels).long()

    print(f"Loaded {len(all_feats)} wbcatt embeddings")

    blood_types = ["basophil", "eosinophil",
                   "lymphocyte", "monocyte", "neutrophil"]
    template = "An image of a {blood_type} typed peripheral blood cell"
    prompt_set = tuple(template.format(blood_type=bt) for bt in blood_types)

    results = util.evaluate_prompt_set(
        prompt_set, all_feats, all_labels, model, tokenizer)
    print("Baseline Evaluation results:\n", results)


if __name__ == "__main__":
    main()
