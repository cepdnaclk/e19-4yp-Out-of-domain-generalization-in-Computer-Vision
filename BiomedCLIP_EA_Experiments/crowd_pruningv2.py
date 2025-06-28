import ast
from typing import List, Tuple
from google import genai
import re
import time
from util import get_prompt_pairs
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
def filter_prompts(client,prompt_batch: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
    """Filter prompts using Gemini to remove similar medical concepts"""
    # Create a mapping from prompt text to score for lookup
    prompt_to_score = {(neg, pos): score for neg, pos, score in prompt_batch}
    
    # # Prepare the input for Gemini
    # prompt_pairs_str = "\n".join([f"('{neg}', '{pos}'), {score:.4f}" 
    #                             for neg, pos, score in prompt_batch])
    prompt_pairs_str = '''
        1. ('Interfollicular areas show a normal complement of T cells', 'Interfollicular areas are dominated by large, atypical cells'), Score: 0.8830
        2. ('Interfollicular areas contain predominantly small lymphocytes', 'Interfollicular areas are expanded by large, atypical cells with prominent nucleoli'), Score: 0.8827
        3. ('No evidence of fibrosis', 'Significant stromal fibrosis surrounding tumor nests'), Score: 0.8808
        4. ('No prominent nucleolus is seen', 'Large, prominent, and irregular nucleoli are present'), Score: 0.8807
        5. ('Lymphocytes exhibit a uniform population', 'Tumor cells demonstrate significant heterogeneity in size and morphology'), Score: 0.8794
        6. ('Scattered, small lymphocytes with condensed chromatin', 'Large, atypical cells with vesicular nuclei and prominent nucleoli'), Score: 0.8790
        7. ('No evidence of mitotic activity in lymphocytes', 'Increased mitotic figures observed in tumor cells'), Score: 0.8776
        8. ('Lymphocytes display uniform size and shape', 'Tumor cells exhibit marked variability in size and shape (anisocytosis/pleomorphism)'), Score: 0.8775
        9. ('Stroma is delicate and sparsely collagenized', 'Stroma is dense and desmoplastic, surrounding tumor nests'), Score: 0.8772
        10. ('No plasmacytic differentiation is observed', 'Plasmacytic differentiation is prominent within the tumor'), Score: 0.8771'''

    # Create the prompt for the LLM
    llm_prompt = f"""The task is to filter textual description pairs of visual discriminative features for tumor detection in histopathology. Remove pairs that have the same medical concepts but differ only in language variations.
        Current Prompt Pairs:
        {prompt_pairs_str}
        Group the numbers of the prompt pairs that has same medical concepts but differ only in language variations.
        Provide the output as follows: list[list[index:int]]
        Let’s think step by step.

    """


    # response = client.models.generate_content(
    #     model="gemma-3-27b-it",
    #     contents=f"""The task is to filter textual description pairs of visual discriminative features for tumor detection in histopathology. 
    #     Remove pairs that have the same medical concepts but differ only in language variations.
        
    #     Current prompt pairs with scores:
    #     {prompt_pairs_str}
        
    #     Only provide the output as Python code in the exact format: 
    #     prompts = [
    #         ("negative description 1", "positive description 1"), score_1,
    #         ("negative description 2", "positive description 2"), score_2,
    #         ...
    #     ]
    #     Each pair should be followed by a comma and its original score.
    #     """
    # )
    
    # Extract the Python code block from the response
    try:
        # Use the get_prompt_pairs function to handle the LLM interaction
        raw_pairs = get_prompt_pairs(
            prompt=llm_prompt,
            llm_client=client, 
            parse_func=util.extract_and_parse_prompt_list_with_scores 
        )
        
        # Process the parsed pairs
        filtered_pairs = []
        for item in raw_pairs:
            if isinstance(item, tuple) and len(item) == 2:
                # Handle case where we got (prompt, response) pairs
                neg, pos = item
                score = prompt_to_score.get((neg, pos), 0.0)
                filtered_pairs.append((neg, pos, score))
            elif isinstance(item, tuple) and len(item) == 3:
                # Handle case where we got (neg, pos, score) directly
                filtered_pairs.append(item)
            else:
                print(f"Skipping malformed item: {item}")
                
        return filtered_pairs
        
    except Exception as e:
        print(f"Error in get_prompt_pairs: {e}")
        return []
    # try:
    #     print(f"{response.text[-1]}")  # Debug: print the full response text
    #     # Find the prompts list in the response
    #     start_idx = response.text.find('prompts = [')
    #     end_idx = response.text.find(']', start_idx) + 1
    #     code_block = response.text[start_idx:end_idx]
        
    #     # Parse the code block line by line
    #     filtered_pairs = []
    #     lines = code_block.split('\n')
    #     for line in lines:
    #         line = line.strip()
    #         if not line or line.startswith('prompts = [') or line.startswith(']'):
    #             continue
                
    #         # Split into prompt part and score part
    #         parts = line.rsplit(',',2)
    #         prompt_part = parts[0].strip()
    #         score_part = parts[1].replace(',', '').strip()
            
    #         try:
    #             # Parse the prompt tuple
    #             neg, pos = ast.literal_eval(prompt_part)
    #             # Parse the score
    #             score = float(score_part)
    #             filtered_pairs.append((neg, pos, score))
    #         except (ValueError, SyntaxError) as e:
    #             print(f"Skipping malformed line: {line} - Error: {e}")
    #             continue
                
    #     return filtered_pairs
        
    # except Exception as e:
    #     print(f"Error parsing response: {e}")
    #     print(f"Response was: {response.text}")
    #     return []


def save_filtered_prompts(prompts: List[Tuple[str, str, float]], filename: str):
    """Save filtered prompts to a Python file in the specified format"""
    with open(filename, 'w') as f:
        f.write("prompts = [\n")
        for neg, pos, score in prompts:
            # Properly escape quotes
            neg_escaped = neg.replace("'", "\\'")
            pos_escaped = pos.replace("'", "\\'")
            f.write(f"    (\"{neg_escaped}\", \"{pos_escaped}\"), {score:.4f},\n")
        f.write("]\n")

def iterative_filtering(client,input_file: str, output_file: str, batch_size: int = 10) -> None:
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
        # time.sleep(2)  # Optional: sleep to avoid rate limiting
        filtered_batch = filter_prompts(client,batch)
        if len(filtered_batch)==0:
            print(f"Filtering returned no results for batch starting at index {i} {batch}.")
        filtered_results = filtered_batch
        
        print(f"Processed batch {i//batch_size + 1}: Input {len(batch)} -> Output {len(filtered_batch)}")
        print(f"Current total filtered: {len(filtered_results)}")
    
    # Save the final results
    save_filtered_prompts(filtered_results, output_file)
    print(f"\nFinished filtering. Original: {total_prompts} pairs, Filtered: {len(filtered_results)} pairs")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    input_filename = "experiment_results/medical_concepts.txt"  # Your input file
    output_filename = "filtered_prompts.py"  # Output Python file
    
    # Initialize your Gemini client here
    from API_KEY import GEMINI_API_KEY
    import util
    # client = genai.Client(api_key=GEMINI_API_KEY)
    client = util.LLMClient()
    
    iterative_filtering(client,input_filename, output_filename)