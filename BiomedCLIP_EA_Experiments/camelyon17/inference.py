from typing import List, Dict, Any
import util
import torch
import numpy as np
from PIL import Image
import os


def inference_on_images(
    image_paths: List[str],
    prompts_file_path: str = "experiment_results/distinct_medical_concepts.txt"
) -> List[Dict[str, Any]]:
    """
    Run inference on a list of image file paths using optimized utility functions.

    Args:
        image_paths: List of paths to image files
        prompts_file_path: Path to the prompts file

    Returns:
        List of dictionaries containing results for each image
    """
    # 1. Load model, preprocess, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. Load prompts
    initial_prompts = util.load_initial_prompts(prompts_file_path)
    print(f"Initial prompts loaded: {len(initial_prompts)} prompts.")

    pq = util.PriorityQueue(
        max_capacity=1000, filter_threshold=0.6, initial=initial_prompts)

    # 3. Select prompts using knee analysis (always)
    all_current_scores = [score for _, score in pq.get_best_n(len(pq))]
    knee_analyzer = util.KneePointAnalysis(all_current_scores)
    recommended_n = knee_analyzer.find_knee_point()
    print(
        f"Recommended number of prompts after knee analysis: {recommended_n}")
    prompts_population = pq.get_best_n(recommended_n)

    # Keep original prompts for detailed results
    original_prompts_population = prompts_population

    # 4. Normalize weights (always for weighted ensemble)
    # Extract original scores
    original_scores = np.array([score for _, score in prompts_population])

    # Normalize weights to sum to 1
    normalized_weights = original_scores / np.sum(original_scores)

    # Reconstruct prompts_population with normalized weights
    prompts_population = [(prompt_pair, weight)
                          for (prompt_pair, _), weight
                          in zip(prompts_population, normalized_weights)]

    print(
        f"Original scores range: [{original_scores.min():.4f}, {original_scores.max():.4f}]")
    print(
        f"Normalized weights range: [{normalized_weights.min():.4f}, {normalized_weights.max():.4f}]")
    print(f"Normalized weights sum: {normalized_weights.sum():.6f}")

    # 5. Process each image
    results = []

    for img_idx, image_path in enumerate(image_paths):
        print(
            f"Processing image {img_idx + 1}/{len(image_paths)}: {os.path.basename(image_path)}")

        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            results.append({
                'image_path': image_path,
                'error': f"File not found: {image_path}",
                'prompt_probabilities': [],
                'final_probability': None
            })
            continue

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(util.DEVICE)

            # Get normalized image features
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features = image_features / \
                    image_features.norm(dim=-1, keepdim=True)

            # Create dummy labels tensor for compatibility with util functions
            # Single image, dummy label
            dummy_labels = torch.tensor([0], dtype=torch.long)

            # Get individual prompt probabilities using util.compute_prompt_probs_matrix
            prob_matrix = util.compute_prompt_probs_matrix(
                prompts_population, image_features, model, tokenizer
            )  # Shape: (1, n_prompts)

            # Extract probabilities for the single image
            individual_probs = prob_matrix[0]

            # Get ensemble probability using util.evaluate_prompt_list
            ensemble_result = util.evaluate_prompt_list(
                prompts_population, image_features, dummy_labels, model, tokenizer, unweighted=False
            )
            # Extract for single image
            final_probability = ensemble_result['ensemble_probs'][0]

            # Build detailed results for each prompt (use original scores for display)
            prompt_results = []
            for j, (prompt_pair, original_score) in enumerate(original_prompts_population):
                # Get individual evaluation for this prompt to get similarity scores
                single_prompt_eval = util.evaluate_prompt_pair(
                    prompt_pair[1], prompt_pair[0], image_features, dummy_labels, model, tokenizer
                )

                # Calculate similarities manually for detailed output
                text_positive = tokenizer([prompt_pair[0]]).to(util.DEVICE)
                text_negative = tokenizer([prompt_pair[1]]).to(util.DEVICE)

                with torch.no_grad():
                    positive_features = model.encode_text(text_positive)
                    negative_features = model.encode_text(text_negative)
                    positive_features = positive_features / \
                        positive_features.norm(dim=-1, keepdim=True)
                    negative_features = negative_features / \
                        negative_features.norm(dim=-1, keepdim=True)

                    sim_positive = torch.cosine_similarity(
                        image_features, positive_features).item()
                    sim_negative = torch.cosine_similarity(
                        image_features, negative_features).item()

                # Get the normalized weight used in ensemble
                actual_weight = prompts_population[j][1]

                prompt_results.append({
                    'prompt_index': j,
                    'prompt_pair': prompt_pair,
                    'prompt_score': original_score,  # Show original score
                    'normalized_weight': actual_weight,  # Show weight used in ensemble
                    'probability': individual_probs[j],
                    'similarity_positive': sim_positive,
                    'similarity_negative': sim_negative
                })

            results.append({
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'prompt_probabilities': prompt_results,
                'final_probability': final_probability,
                'num_prompts_used': len(original_prompts_population),
                'ensemble_type': 'weighted_normalized'
            })

            print(f"  Final probability: {final_probability:.4f}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results.append({
                'image_path': image_path,
                'error': str(e),
                'prompt_probabilities': [],
                'final_probability': None
            })

    return results


def print_detailed_results(results: List[Dict[str, Any]], show_all_prompts: bool = False):
    """Print detailed inference results."""
    print("\n" + "="*80)
    print("DETAILED INFERENCE RESULTS")
    print("="*80)

    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {result.get('image_name', 'Unknown')}")
        print(f"Path: {result['image_path']}")

        if 'error' in result:
            print(f"ERROR: {result['error']}")
            continue

        print(f"Final Probability: {result['final_probability']:.4f}")
        print(f"Ensemble Type: {result.get('ensemble_type', 'N/A')}")
        print(f"Number of Prompts: {result.get('num_prompts_used', 0)}")

        if show_all_prompts and result['prompt_probabilities']:
            print("\nIndividual Prompt Results:")
            print("-" * 60)
            for prompt_result in result['prompt_probabilities']:
                print(
                    f"  Prompt {prompt_result['prompt_index']:2d}: {prompt_result['probability']:.4f}")
                print(f"    Positive: '{prompt_result['prompt_pair'][0]}'")
                print(f"    Negative: '{prompt_result['prompt_pair'][1]}'")
                print(
                    f"    Original Score: {prompt_result['prompt_score']:.4f}")
                if 'normalized_weight' in prompt_result:
                    print(
                        f"    Normalized Weight: {prompt_result['normalized_weight']:.4f}")
                print(f"    Sim+: {prompt_result['similarity_positive']:.4f}, "
                      f"Sim-: {prompt_result['similarity_negative']:.4f}")
                print()


def main():
    """Main function for command-line usage."""
    # Example usage with sample image paths
    image_paths = [
        # Add your image paths here
        "images/non_tumor.png",
        "images/tumor.png",
    ]

    print("Starting inference on provided images...")
    print(f"Processing {len(image_paths)} images")

    # Run inference
    results = inference_on_images(
        image_paths=image_paths,
        prompts_file_path="experiment_results/distinct_medical_concepts.txt"
    )

    # Print results
    print_detailed_results(results, show_all_prompts=True)

    # Summary statistics
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        final_probs = [r['final_probability'] for r in successful_results]
        print(f"\nSUMMARY STATISTICS:")
        print(
            f"Successfully processed: {len(successful_results)}/{len(results)} images")
        print(f"Mean probability: {np.mean(final_probs):.4f}")
        print(f"Std probability: {np.std(final_probs):.4f}")
        print(f"Min probability: {np.min(final_probs):.4f}")
        print(f"Max probability: {np.max(final_probs):.4f}")


if __name__ == "__main__":
    main()
