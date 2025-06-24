import ast
from typing import List, Tuple
from google import genai
import re

def load_prompts_from_file(filename: str) -> List[Tuple[str, str, float]]:
    """Load prompts from text file with format ('neg', 'pos'), Score: x.xxxx"""
    prompts = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract the tuple part (handling both quoted and unquoted Score)
            tuple_part = line.split('Score:')[0].strip()
            score_str = line.split('Score:')[1].strip() if 'Score:' in line else '0'
            
            try:
                # Clean the tuple string if needed
                tuple_str = tuple_part.replace('(', '').replace(')', '').split(',')
                neg = tuple_str[0].strip().strip("'")
                pos = tuple_str[1].strip().strip("'")
                score = float(score_str)
                prompts.append((neg, pos, score))
            except (ValueError, IndexError) as e:
                print(f"Skipping malformed line: {line} - Error: {e}")
                continue
                
    return prompts
def filter_prompts(prompt_batch: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
    """Filter prompts using Gemini to remove similar medical concepts"""
    # Create a mapping from prompt text to score for lookup
    prompt_to_score = {(neg, pos): score for neg, pos, score in prompt_batch}
    
    # Prepare the input for Gemini
    prompt_pairs_str = "\n".join([f"('{neg}', '{pos}'), {score:.4f}" 
                                for neg, pos, score in prompt_batch])
    
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=f"""The task is to filter textual description pairs of visual discriminative features for tumor detection in histopathology. 
        Remove pairs that have the same medical concepts but differ only in language variations. Keep only conceptually distinct pairs.
        
        Current prompt pairs with scores:
        {prompt_pairs_str}
        
        Only provide the output as Python code in the exact format: 
        prompts = [
            ("negative description 1", "positive description 1"), score_1,
            ("negative description 2", "positive description 2"), score_2,
            ...
        ]
        Each pair should be followed by a comma and its original score.
        """
    )
    
    # Extract the Python code block from the response
    try:
        # Find the prompts list in the response
        start_idx = response.text.find('prompts = [')
        end_idx = response.text.find(']', start_idx) + 1
        code_block = response.text[start_idx:end_idx]
        
        # Parse the code block line by line
        filtered_pairs = []
        lines = code_block.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('prompts = [') or line.startswith(']'):
                continue
                
            # Split into prompt part and score part
            parts = line.rsplit(',',2)
            prompt_part = parts[0].strip()
            score_part = parts[1].replace(',', '').strip()
            
            try:
                # Parse the prompt tuple
                neg, pos = ast.literal_eval(prompt_part)
                # Parse the score
                score = float(score_part)
                filtered_pairs.append((neg, pos, score))
            except (ValueError, SyntaxError) as e:
                print(f"Skipping malformed line: {line} - Error: {e}")
                continue
                
        return filtered_pairs
        
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Response was: {response.text}")
        return []


# def save_filtered_prompts(prompts: List[Tuple[str, str, float]], filename: str):
#     """Save filtered prompts to a Python file in the specified format"""
#     with open(filename, 'w') as f:
#         f.write("prompts = [\n")
#         for neg, pos, score in prompts:
#             # Properly escape quotes
#             neg_escaped = neg.replace("'", "\\'")
#             pos_escaped = pos.replace("'", "\\'")
#             f.write(f"    (\"{neg_escaped}\", \"{pos_escaped}\"), {score:.4f},\n")
#         f.write("]\n")

def iterative_filtering(input_file: str, output_file: str, batch_size: int = 100) -> None:
    """Perform iterative filtering of prompts in batches"""
    all_prompts = load_prompts_from_file(input_file)
    total_prompts = len(all_prompts)
    print(f"Loaded {total_prompts} prompts from {input_file}")
    
    filtered_results = []
    
    # Process in batches
    for i in range(0, total_prompts, batch_size):
        batch = all_prompts[i:i + batch_size]
        # Combine with previous filtered results if not first batch
        if i > 0:
            batch = filtered_results[:] + batch  # Carry over previous results
            
        filtered_batch = filter_prompts(batch)
        filtered_results.extend(filtered_batch)
        
        print(f"Processed batch {i//batch_size + 1}: Input {len(batch)} -> Output {len(filtered_batch)}")
        print(f"Current total filtered: {len(filtered_results)}")
    
    return filtered_results
    # # Save the final results
    # save_filtered_prompts(filtered_results, output_file)
    # print(f"\nFinished filtering. Original: {total_prompts} pairs, Filtered: {len(filtered_results)} pairs")
    # print(f"Results saved to {output_file}")

