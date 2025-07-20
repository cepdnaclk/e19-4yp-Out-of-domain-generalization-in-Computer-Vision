import util
import re
import ast
from typing import List


def parse_grouped_indexes(text: str):
    """Parse grouped indexes from LLM response text."""
    return ast.literal_eval(text)


def get_unique_indexes(grouped_indexes: List[List[int]]) -> List[int]:
    """Extract the first index from each group to get unique representatives."""
    unique_indexes: List[int] = []
    for group in grouped_indexes:
        unique_indexes.append(group[0])  # Append the first index of each group
    return unique_indexes


def get_remaining_indexes(grouped_indexes: List[List[int]], total_count: int) -> List[int]:
    """Find missing indexes that weren't included in the grouping."""
    flat_list = [item for sublist in grouped_indexes for item in sublist]
    missing_indexes = set(range(1, total_count + 1)) - set(flat_list)
    return list(missing_indexes)


def get_grouped_indexes_from_llm(llm_prompt: str, client, max_retries: int) -> List[List[int]]:
    """Get grouped indexes from LLM with retry logic."""
    print("Sending Prompt: ", llm_prompt)
    for attempt in range(max_retries):
        try:
            response = client.get_llm_response(prompt=llm_prompt)
            print(response)
            m = re.search(r'```python\s*([\s\S]*?)\s*```', response)
            if not m:
                raise ValueError("No ```python ... ``` block found")
            list_str = m.group(1)
            grouped_indexes = ast.literal_eval(list_str)
            return grouped_indexes
        except Exception as e:
            print(f"Error in LLM response: {e}")
            if attempt == max_retries - 1:
                raise ValueError(
                    "Failed to get a valid response from the LLM after multiple retries.")


def perform_crowding_pruning(
    initial_prompts: List[util.InitialItem],
    number_of_prompts_to_group: int = 30,
    crowding_iterations: int = 20,
    max_retries: int = 5,
    provider: str = 'gemini'
) -> util.PriorityQueue:
    """
    Perform crowding-based pruning on prompts to remove duplicates.

    Args:
        initial_prompts_path: Path to initial prompts file
        number_of_prompts_to_group: Number of prompts to process per iteration
        crowding_iterations: Total number of crowding iterations
        max_retries: Maximum retries for LLM calls
        provider: LLM provider ('gemini', 'ollama', 'azure_openai')

    Returns:
        PriorityQueue with pruned prompts
    """

    # Load initial prompts
    pq = util.PriorityQueue(
        max_capacity=1000, filter_threshold=0.6, initial=initial_prompts)

    # Initialize LLM client
    client = util.LLMClient(provider=provider)

    # Define prompts for LLM
    group_prompt = """The task is to group textual description pairs of visual discriminative features for tumor detection in histopathology. 
Current Prompt Pairs:
{prompt_pairs_str}
Group the prompt pairs that has same observation but differ only in language variations. Give the indexes of the grouped pairs in the output.
Provide the output as follows: list[list[index:int]]. Make sure to include all pairs in the output, even if they are not grouped with others.
Let's think step by step. Count from 1-{num_of_prompts} to verify each item is in the list.
"""

    retry_prompt = """The task is to group textual description pairs of visual discriminative features for tumor detection in histopathology. 
Current Prompt Pairs:
{prompt_pairs_str}

You've already grouped some pairs, but there are still ungrouped pairs remaining.
Current Grouped indexes:
{current_grouped_indexes}

Remaining Prompt Pairs:
{prompt_pairs_str_remaining}

Provide the output as follows: list[list[index:int]]. Make sure to include all pairs in the output, even if they are not grouped with others.
Let's think step by step."""

    deleted_num = 0

    for i in range(crowding_iterations):
        print(f"=== Iteration {i+1} of {crowding_iterations} ===")

        # Check if we have enough prompts to continue
        if len(pq) < number_of_prompts_to_group:
            print(
                f"Not enough prompts remaining ({len(pq)} < {number_of_prompts_to_group}). Stopping.")
            break

        # Retrieve the best prompt pairs from the priority queue
        prompt_pairs = pq.get_best_n(n=number_of_prompts_to_group)
        prompt_pairs_str = "\n".join(
            [f"{i+1}. ('{pair[0]}' , '{pair[1]}')" for i,
             (pair, score) in enumerate(prompt_pairs)]
        )

        # Get initial grouping from LLM
        grouped_indexes = get_grouped_indexes_from_llm(
            llm_prompt=group_prompt.format(
                prompt_pairs_str=prompt_pairs_str,
                num_of_prompts=number_of_prompts_to_group
            ),
            client=client,
            max_retries=max_retries
        )

        # Handle remaining indexes if any
        remaining_indexes = get_remaining_indexes(
            grouped_indexes, number_of_prompts_to_group)
        if len(remaining_indexes) > 0:
            print(f"Remaining indexes: {remaining_indexes}")

            # Filter valid remaining indexes
            valid_remaining = [
                i for i in remaining_indexes if 1 <= i <= len(prompt_pairs)]
            if not valid_remaining:
                print("No valid remaining indexes to process.")
            else:
                retry_prompt_str = retry_prompt.format(
                    prompt_pairs_str=prompt_pairs_str,
                    current_grouped_indexes=str(grouped_indexes),
                    prompt_pairs_str_remaining="\n".join(
                        [f"{valid_remaining[i]}. ('{pair[0]}' , '{pair[1]}')"
                         for i, (pair, score) in enumerate([prompt_pairs[j-1] for j in valid_remaining])]
                    )
                )

                grouped_indexes = get_grouped_indexes_from_llm(
                    llm_prompt=retry_prompt_str,
                    client=client,
                    max_retries=max_retries
                )

        # Get unique representative prompts
        unique_indexes = get_unique_indexes(grouped_indexes)
        print(f"Unique indexes: {unique_indexes}")

        # Select the best prompts based on the unique indexes
        best_prompt_pairs_with_scores = [
            prompt_pairs[i-1] for i in unique_indexes if 1 <= i <= len(prompt_pairs)]

        # Delete the top n prompts from the priority queue
        pq.delete_top_n(number_of_prompts_to_group)

        # Add the selected unique prompts back to the priority queue
        for prompt_pair, score in best_prompt_pairs_with_scores:
            pq.insert(prompt_pair, score)

        # Print the number of deleted prompts
        deleted_this_iteration = number_of_prompts_to_group - \
            len(unique_indexes)
        deleted_num += deleted_this_iteration
        print(
            f"Iteration {i+1} completed. Deleted {deleted_this_iteration} duplicates this iteration. Total deleted: {deleted_num}")

    pq.delete_top_n(pq.max_capacity)  # Clear the queue at the end
    for prompt_pair, score in best_prompt_pairs_with_scores:
        # Reinsert the best prompts into the queue
        pq.insert(prompt_pair, score)

    print(f"\nCrowding completed. Total prompts deleted: {deleted_num}")
    print(f"Final queue size: {len(pq)}")

    return pq


def main():
    """Main function to run crowding pruning."""

    intial_prompts = util.load_initial_prompts(
        "experiment_results/medical_concepts.txt")

    if not intial_prompts:
        print("No initial prompts found. Exiting.")
        return

    print(f"Loaded {len(intial_prompts)} initial prompts.")
    pq = perform_crowding_pruning(
        initial_prompts=intial_prompts,
        number_of_prompts_to_group=30,
        crowding_iterations=20,
        max_retries=5,
        provider='gemini'
    )

    # Display final results
    final_prompts = pq.get_best_n(n=min(30, len(pq)))
    print("\nFinal best prompts after crowding:")
    for i, (prompt_pair, score) in enumerate(final_prompts):
        print(f"{i+1:2d}. {prompt_pair}, Score: {score:.4f}")


if __name__ == "__main__":
    main()
