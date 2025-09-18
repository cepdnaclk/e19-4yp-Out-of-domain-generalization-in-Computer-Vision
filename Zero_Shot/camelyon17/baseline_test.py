import torch
import util


def main():
    # Set the dermoscopic feature to optimize prompts for

    # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. load dataset - MODIFIED FOR CHEXPERT
    features, labels= util.extract_center_embeddings(
        model=model,
        preprocess=preprocess,
        num_centers=5,
        isTrain=False,
        
    )

    # we only need the test features
    features = features[4]
    labels = labels[4]
    
    # Convert to tensors - MODIFIED FOR MULTI-OBSERVATION SUPPORT
    all_feats = torch.from_numpy(features).float()
    all_labels = torch.from_numpy(labels).long()

    print(f"Loaded {len(all_feats)} Camelyon17 embeddings")
    print("all_labels:\n", all_labels)

    prompt_set = (
        "This is an image of a normal lymph node",
        "This is an image of a tumor lymph node",
    )

    results = util.evaluate_prompt_set(
        prompt_set, all_feats, all_labels, model, tokenizer)
    print("Baseline Evaluation results:\n", results)

    


if __name__ == "__main__":
    main()
