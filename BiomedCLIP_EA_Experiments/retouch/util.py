import heapq
import tokenize
import io
import random
from typing import Any, List, Optional, Set, Tuple
import re
import torch
import ast
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from typing import Callable, List, Tuple
import os
import pandas as pd
import numpy as np
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
import json
from tqdm import tqdm
import time
from API_KEY import GEMINI_API_KEY
from google import genai
import ollama
import torch.nn.functional as F
import SimpleITK as sitk


CONFIG_PATH = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/e19-4yp-Out-of-domain-generalization-in-Computer-Vision/BioMedClip/checkpoints/open_clip_config.json"
WEIGHTS_PATH = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/e19-4yp-Out-of-domain-generalization-in-Computer-Vision/BioMedClip/checkpoints/open_clip_pytorch_model.bin"
MODEL_NAME = "biomedclip_local"
CACHE_PATH = "cached"
CONTEXT_LENGTH = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PromptPair = Tuple[str, str]
InitialItem = Tuple[PromptPair, float]
PromptList = List[Tuple[PromptPair, float]]


class RETOUCHOCTDataset(Dataset):
    def __init__(self, root_dir, preprocess, target_fluid=1):  # target_fluid: 1=IRF, 2=SRF, 3=PED
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.target_fluid = target_fluid
        self.samples = []
        
        # Get all patient folders
        patient_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        
        for patient in patient_folders:
            oct_path = os.path.join(root_dir, patient, 'oct.mhd')
            ref_path = os.path.join(root_dir, patient, 'reference.mhd')
            
            if os.path.exists(oct_path) and os.path.exists(ref_path):
                # Load reference volume to check fluid presence
                ref_volume = sitk.GetArrayFromImage(sitk.ReadImage(ref_path))
                
                # Check if target fluid exists in any slice
                has_fluid = int(np.any(ref_volume == target_fluid))
                
                # Add all slices from this volume (same label for all slices)
                oct_volume = sitk.GetArrayFromImage(sitk.ReadImage(oct_path))
                for slice_idx in range(len(oct_volume)):
                    self.samples.append((patient, slice_idx, has_fluid))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        patient, slice_idx, label = self.samples[idx]
        oct_path = os.path.join(self.root_dir, patient, 'oct.mhd')
        
        # Load and preprocess the OCT slice
        oct_volume = sitk.GetArrayFromImage(sitk.ReadImage(oct_path))
        oct_slice = oct_volume[slice_idx]
        
        # Normalize and convert to PIL Image
        oct_slice = (oct_slice - oct_slice.min()) / (oct_slice.max() - oct_slice.min())
        oct_slice = (oct_slice * 255).astype(np.uint8)
        image = Image.fromarray(oct_slice).convert('RGB')
        
        if self.preprocess:
            image = self.preprocess(image)
            
        return image, torch.tensor(label)





def load_clip_model(
    config_path: str = CONFIG_PATH,
    model_name: str = MODEL_NAME,
    weights_path: str = WEIGHTS_PATH,
    device: torch.device = DEVICE,
) -> Tuple[torch.nn.Module, Callable]:
    """
    Load a CLIP model and its preprocessing pipeline from a JSON config.

    Args:
        config_path: Path to JSON config with model and preprocess settings.
        model_name: Name or HF hub identifier of the CLIP model.
        weights_path: Path to pretrained weights file.
        device: torch.device to place the model on.

    Returns:
        model: The loaded and eval()'d CLIP model on `device`.
        preprocess: A callable image preprocessing function.
    """
    # 1. Read configuration
    with open(config_path, "r") as f:
        cfg = json.load(f)
    model_cfg, preproc_cfg = cfg["model_cfg"], cfg["preprocess_cfg"]

    # 2. Register local config if needed
    if (
        not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
    ):
        _MODEL_CONFIGS[model_name] = model_cfg

    # 3. Build tokenizer, model, and preprocess
    tokenizer = get_tokenizer(model_name)
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=weights_path,
        **{f"image_{k}": v for k, v in preproc_cfg.items()}
    )

    model = model.to(device).eval()
    return model, preprocess, tokenizer


def extract_oct_embeddings(model, preprocess, 
                         root_dir="/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/retouch_data/RETOUCH-TrainingSet-Cirrus",
                         cache_dir="./retouch_cache",
                         target_fluid=1,  # 1=IRF, 2=SRF, 3=PED
                         train_or_test="train"):
    os.makedirs(cache_dir, exist_ok=True)
    features_cache = os.path.join(cache_dir, f"{train_or_test}_retouch_fluid{target_fluid}_features.npy")
    labels_cache = os.path.join(cache_dir, f"{train_or_test}_retouch_fluid{target_fluid}_labels.npy")
    
    if os.path.exists(features_cache) and os.path.exists(labels_cache):
        print(f"Loading cached {train_or_test} embeddings...")
        return np.load(features_cache), np.load(labels_cache)
    
    # Create full dataset
    full_dataset = RETOUCHOCTDataset(root_dir, preprocess, target_fluid)
    
    # Create dataloader
    loader = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    features, labels = [], []
    with torch.no_grad():
        for batch, batch_labels in tqdm(loader, desc=f"Extracting {train_or_test} embeddings"):
            batch = batch.to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # Normalize features
            features.append(feats.cpu().numpy())
            labels.append(batch_labels.numpy())
    
    features_array = np.concatenate(features)
    labels_array = np.concatenate(labels)
    
    # Save to cache
    np.save(features_cache, features_array)
    np.save(labels_cache, labels_array)
    
    return features_array, labels_array



def evaluate_prompt_list(
    prompt_list: PromptList,
    image_feats: torch.Tensor,  # (N, D), precomputed
    image_labels: torch.Tensor,  # (N,)
    model,
    tokenizer,
    unweighted: bool = False  # If True, all prompts are treated equally
):
    all_weighted_probs = []
    total_weight = 0.0

    # Ensure image feats and labels are on the correct device once
    print(f"shape of image_feats:dtype: {image_feats} ")
    feats = image_feats.to(DEVICE)
    labels = image_labels.to(DEVICE)

    with torch.no_grad():
        for (negative_prompt, positive_prompt), original_weight in prompt_list:
            # Determine the effective weight
            effective_weight = 1.0 if unweighted else original_weight

            # Skip if effective_weight is zero to avoid unnecessary computations
            if effective_weight == 0 and not unweighted:
                continue

            text_inputs = tokenizer(
                [negative_prompt, positive_prompt],
                context_length=CONTEXT_LENGTH
            ).to(DEVICE)

            text_feats = model.encode_text(text_inputs)  # (2, D)
            text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
            logit_scale = model.logit_scale.exp()

            # compute all logits at once: (N, 2)
            logits = logit_scale * (feats @ text_feats.t())
            probs = logits.softmax(dim=1)

            # Extract probability for the positive class (assuming index 1)
            # If your tumor class is indeed index 0 as per your previous code, change `probs[:, 1]` to `probs[:, 0]`
            current_positive_class_probs = probs[:, 1].cpu().numpy()

            all_weighted_probs.append(
                current_positive_class_probs * effective_weight)
            total_weight += effective_weight

    # Perform ensemble
    if total_weight == 0:
        # If all weights were zero, or prompt_list was empty and unweighted was False
        # This case handles scenarios where no valid prompts or non-zero weights were found.
        # It might be better to return an error or specific empty results depending on desired behavior.
        # For now, let's return a dict with NaN or 0 for metrics for clarity,
        # or raise an error if this state indicates a problem.
        raise ValueError(
            "Total effective weight of prompts is zero. Cannot perform ensemble.")

    # Convert list of arrays to a single array and sum along the first dimension
    ensemble_probs_unnormalized = np.sum(all_weighted_probs, axis=0)
    ensemble_probs = ensemble_probs_unnormalized / total_weight

    # Convert probabilities to predictions
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    y_true = labels.cpu().numpy()

    # metrics
    acc = accuracy_score(y_true, ensemble_preds)
    auc = roc_auc_score(y_true, ensemble_probs)
    cm = confusion_matrix(y_true, ensemble_preds)
    report = classification_report(y_true, ensemble_preds, digits=4)

    return {'accuracy': acc, 'auc': auc, 'cm': cm, 'report': report, 'ensemble_probs': ensemble_probs}

def evaluate_prompt_pair(
    negative_prompt: str,
    positive_prompt: str,
    image_feats: torch.Tensor,    # (N, D), precomputed
    image_labels: torch.Tensor,   # (N,)
    model,
    tokenizer
):
    # encode prompts once
    text_inputs = tokenizer(
        [negative_prompt, positive_prompt],
        context_length=CONTEXT_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        text_feats = model.encode_text(text_inputs)           # (2, D)
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
        logit_scale = model.logit_scale.exp()

        # move image feats to DEVICE for one matrix multiply
        feats = image_feats.to(DEVICE)                        # (N, D)
        labels = image_labels.to(DEVICE)                      # (N,)

        # compute all logits at once: (N, 2)
        logits = logit_scale * (feats @ text_feats.t())
        probs = logits.softmax(dim=1)
        preds = logits.argmax(dim=1)

        y_pred = preds.cpu().numpy()
        y_prob = probs[:, 1].cpu().numpy()    # tumor-class prob
        y_true = labels.cpu().numpy()

        # Compute Binary Cross-Entropy Loss
        bce_loss = F.binary_cross_entropy(
            input=torch.tensor(y_prob, device=DEVICE).float(),
            target=torch.tensor(y_true, device=DEVICE).squeeze().float()
        ).item()

        # Invert BCE loss: 1/(1 + loss) (so lower loss → higher value)
        inverted_bce = 1.0 / (1.0 + bce_loss)


    # metrics
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4,zero_division=0)
    return {'accuracy': acc, 'auc': auc, 'cm': cm, 'report': report, 'inverted_bce': inverted_bce}

def convert_prompts_raw_to_double_quotes(raw_code: str) -> str:
    # Match inside the prompts list
    list_body = re.search(r"prompts\s*=\s*\[(.*)\]", raw_code, re.DOTALL)
    if not list_body:
        raise ValueError("Could not find 'prompts = [...]' assignment.")
    
    content = list_body.group(1)

    # Regex to find tuples: ('...', '...')
    pattern = re.compile(r"\(\s*'(.*?)'\s*,\s*'(.*?)'\s*\)", re.DOTALL)

    # Replace with double-quoted strings
    result = "prompts = [\n"
    for match in pattern.finditer(content):
        left = match.group(1).replace('"', '\\"')
        right = match.group(2).replace('"', '\\"')
        result += f'    ("{left}", "{right}"),\n'
    result = result.rstrip(',\n') + "\n]"
    return result

def force_double_quotes(code: str) -> str:
    """
    Rewrites every Python string-literal in `code` to use double-quotes,
    properly handling apostrophes and other special characters.
    """
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    new_tokens = []
    for toknum, tokval, start, end, line in tokens:
        if toknum == tokenize.STRING:
            value = ast.literal_eval(tokval)

            tokval = json.dumps(value)

            

        new_tokens.append((toknum, tokval))
    return tokenize.untokenize(new_tokens)

# import tokenize
# import io
# import ast

# def force_double_quotes(code: str) -> str:
#     """
#     Rewrites every Python string-literal in `code` to use double-quotes,
#     properly handling apostrophes and other special characters without
#     escaping apostrophes.
#     """
#     tokens = tokenize.generate_tokens(io.StringIO(code).readline)
#     new_tokens = []
#     for toknum, tokval, start, end, line in tokens:
#         if toknum == tokenize.STRING:
#             value = ast.literal_eval(tokval)
#             # Use a custom replacement that preserves apostrophes
#             if "'" in value:
#                 # If the string contains apostrophes, we need to handle them carefully
#                 tokval = '"' + value.replace('"', '\\"') + '"'
#             else:
#                 # Simple case - just use double quotes
#                 tokval = '"' + value + '"'
#         new_tokens.append((toknum, tokval))
#     return tokenize.untokenize(new_tokens)


def extract_and_parse_prompt_list(code: str) -> List[Tuple[str, str]]:
    """
    From a string of Python code, finds the first occurrence of
        = [ ... ]
    and parses that bracketed literal into a List[Tuple[str,str]].

    Raises:
        ValueError if no list literal is found or it’s malformed.
    """
    # 1) grab everything from the first '=' up to the matching ']'
    m = re.search(r'=\s*(\[\s*[\s\S]*?\])', code)
    if not m:
        raise ValueError("No list literal found after an '=' in the code")
    list_str = m.group(1)

    # # Added by Mansitha
    # # Try to fix common string literal issues
    # list_str = list_str.replace('\\', '\\\\')  # handle existing escapes
    # list_str = re.sub(r'(?<!\\)\'', '\\\'', list_str)  # escape unescaped single quotes


    print(f"Inside of parse Function {code}")
    # 2) safely evaluate it (only literals)
    try:
        data: Any = ast.literal_eval(list_str)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Malformed list literal: {e}")

    # 3) validate shape
    if not isinstance(data, list) or not all(
        isinstance(item, (list, tuple)) and len(item) == 2 for item in data
    ):
        raise ValueError(
            "Parsed object is not a list of 2-element lists/tuples")

    # 4) convert to List[Tuple[str,str]]
    return [(str(a), str(b)) for a, b in data]

def extract_and_parse_prompt_list_with_scores(code: str) -> List[Tuple[str, str, float]]:
    """
    From a string of Python code, finds the first occurrence of
        prompts = [ ... ]
    and parses that bracketed literal into a List[Tuple[str, str, float]].
    Expects format: ("neg", "pos"), score,

    Raises:
        ValueError if no list literal is found or it's malformed.
    """
    # 1) Find the prompts list declaration
    m = re.search(r'prompts\s*=\s*(\[\s*[\s\S]*?\])', code)
    if not m:
        raise ValueError("No 'prompts = [...]' list literal found in the code")
    list_str = m.group(1)

    # 2) Clean and normalize the string
    # Remove newlines and extra spaces for easier parsing
    cleaned = ' '.join(list_str.split())
    # Ensure we have proper comma separation between items
    cleaned = cleaned.replace('),', '), ')

    # 3) Split into individual items while preserving structure
    items = []
    current_item = []
    depth = 0  # track nesting level for tuples
    buffer = ""
    
    for char in cleaned[1:-1]:  # skip outer brackets
        if char == '(':
            depth += 1
            buffer += char
        elif char == ')':
            depth -= 1
            buffer += char
        elif char == ',' and depth == 0:
            # Only split on top-level commas
            if buffer.strip():
                current_item.append(buffer.strip())
                buffer = ""
            if len(current_item) == 2:  # we have both tuple and score
                items.append(tuple(current_item))
                current_item = []
        else:
            buffer += char
    
    # Add the last item if any
    if buffer.strip():
        current_item.append(buffer.strip())
    if current_item:
        items.append(tuple(current_item))

    # 4) Parse each item into (neg, pos, score)
    parsed_items = []
    for item in items:
        if len(item) != 2:
            raise ValueError(f"Expected tuple and score, got: {item}")
        
        # Parse the prompt tuple
        try:
            prompt_tuple = ast.literal_eval(item[0])
            if not isinstance(prompt_tuple, tuple) or len(prompt_tuple) != 2:
                raise ValueError(f"Expected 2-element tuple, got: {prompt_tuple}")
            neg, pos = prompt_tuple
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Malformed prompt tuple: {e}")

        # Parse the score
        try:
            score = float(item[1])
        except ValueError as e:
            raise ValueError(f"Malformed score value: {e}")

        parsed_items.append((str(neg), str(pos), score))

    return parsed_items




def extract_and_parse_prompt_tuple(code: str) -> Tuple[str, str]:
    """
    From a string of Python code, finds the first literal tuple of two strings
    (e.g. ("neg prompt","pos prompt")) and returns it as (str, str).

    Raises:
        ValueError if no suitable 2-element string tuple is found.
    """
    # Parse into an AST
    tree = ast.parse(code)

    # Walk the tree looking for a Tuple node with exactly two string constants
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 2:
            a, b = node.elts
            if (
                isinstance(a, ast.Constant) and isinstance(a.value, str)
                and isinstance(b, ast.Constant) and isinstance(b.value, str)
            ):
                return (a.value, b.value)

    raise ValueError("No 2-element string tuple found in code")


class LLMClient:
    """
    A unified client for interacting with different LLM providers (Gemini, Ollama).
    """

    def __init__(self, use_local_ollama: bool = False, ollama_model: str = "hf.co/unsloth/medgemma-27b-text-it-GGUF:Q8_0", gemini_model: str = "gemma-3-27b-it"):
        self.use_local_ollama = use_local_ollama
        self.gemini_client = None
        self.ollama_model = ollama_model  # Default Ollama model
        self.gemini_model = gemini_model  # Default Gemini model
        self.ollama_host = "http://[::1]:11434"
        self._ollama_client_instance = ollama.Client(host=self.ollama_host)

        if not use_local_ollama:
            if GEMINI_API_KEY:
                self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            else:
                raise ValueError(
                    "Gemini API key must be provided if not using local Ollama.")

    def _get_response_from_gemini(self, prompt: str) -> str:
        """Sends a prompt to the Gemini client and returns the response text."""
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized.")
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model, contents=prompt)
        return response.text

    def _get_response_from_ollama(self, prompt: str) -> str:
        """Sends a prompt to the Ollama client and returns the response text."""
        # Use the initialized _ollama_client_instance
        response = self._ollama_client_instance.chat(
            model=self.ollama_model, messages=[
                {"role": "user", "content": prompt}]
        )
        return response['message']['content']
    # def _get_response_from_ollama(self, prompt: str) -> str:
    #     """Sends a prompt to the Ollama client and returns the response text."""
    #     # The ollama client typically manages its connection internally, no explicit 'client' object
    #     response = ollama.chat(model=self.ollama_model, messages=[
    #                            {"role": "user", "content": prompt}])
    #     return response['message']['content']

    def get_llm_response(self, prompt: str) -> str:
        """Gets a response from the configured LLM (Gemini or Ollama)."""
        if self.use_local_ollama:
            return self._get_response_from_ollama(prompt)
        else:
            return self._get_response_from_gemini(prompt)


def get_prompt_pairs(
    prompt: str,
    llm_client: LLMClient,  # Accept the unified LLMClient instance
    parse_func: Callable = extract_and_parse_prompt_list,
    max_retries: int = 1
) -> List[Tuple[str, str]]:
    """
    Retrieves and parses a list of prompt-response pairs from an LLM.

    Args:
        prompt: The initial prompt to send to the LLM to generate the pairs.
        llm_client: An instance of LLMClient configured for Gemini or Ollama.
        parse_func: A function to parse the raw LLM response into a list of tuples.
                    Defaults to extract_and_parse_prompt_list.
        max_retries: The maximum number of attempts to get and parse a valid response.

    Returns:
        A list of (prompt_string, response_string) tuples.

    Raises:
        RuntimeError: If unable to get and parse prompts after multiple attempts.
        ValueError: If the LLM response does not contain a valid Python block
                    or if parsing fails.
    """
    for attempt in range(1, max_retries + 1):
        try:
            # Use the unified LLMClient to get the raw response
            raw = llm_client.get_llm_response(prompt)
            # print(f"Raw response on attempt {attempt}: {raw}...")

            # 1) extract the python block

            m = re.search(r'```python\s*([\s\S]*?)\s*```', raw)
            if not m:
                raise ValueError("No ```python ... ``` block found")
            code = m.group(1)

            # 2) normalize all literals to double-quoted form
            print(f"******************* Before forcing double quotes: {code}...")
            # code = force_double_quotes(code)
            # code = convert_prompts_raw_to_double_quotes(code)
            print(f"******************* After forcing double quotes: {code}...")

            # print(f"Normalized code on attempt {attempt}: {code}...")

            # 3) convert the string to a list of tuples
            prompts_list = parse_func(code)
            prompts: List[Tuple[str, str]] = prompts_list
            print(f"Loaded {len(prompts)} prompt-pairs.")
            print("First pair:", prompts[0])
            return prompts

        except Exception as e:
            print(
                f"[Warning] get_prompt_pairs parse error on attempt {attempt}/{max_retries}: {e}")
            # sleep for 2 secs per attempt
            time.sleep(2 * attempt)  # exponential backoff

            if attempt == max_retries:
                raise RuntimeError(
                    "Failed to parse prompts after multiple attempts") from e
            # otherwise, retry immediately

    # Should never reach here
    raise RuntimeError("Unreachable")


class PriorityQueue:
    # type: ignore
    def __init__(self, max_capacity: int = 10, initial: Optional[List[Tuple[PromptPair, float]]] = None,filter_threshold: float = 0.4):

        self.filter_threshold = filter_threshold
        self.max_capacity: int = max_capacity
        # Store (score, prompt_pair); min-heap root is the lowest score.
        self._heap: List[Tuple[float, PromptPair]] = []  # type: ignore

        # Track prompts for O(1) membership checks
        self._set: Set[PromptPair] = set()

        # If the user passed some initial prompt-pairs, insert them now:
        if initial is not None:
            for prompt_pair, score in initial:
                self.insert(prompt_pair, score)

    def insert(self, prompt_pair: PromptPair, score: float) -> None:  # type: ignore
        # Skip if prompt pair already exists
        if prompt_pair in self._set:
            return
        # Skip low scores
        if score < self.filter_threshold:
            return

        if len(self._heap) < self.max_capacity:
            # Add new entry
            heapq.heappush(self._heap, (score, prompt_pair))
            self._set.add(prompt_pair)
        else:
            # Only replace if new score beats the current minimum
            if score > self._heap[0][0]:
                # Replace smallest entry, capturing the popped item
                old_score, old_pair = heapq.heapreplace(
                    self._heap, (score, prompt_pair))
                # Update prompt set
                self._set.remove(old_pair)
                self._set.add(prompt_pair)

    def get_best(self) -> Optional[Tuple[PromptPair, float]]:
        if not self._heap:
            return None
        best_score, best_pair = max(self._heap, key=lambda x: x[0])
        return best_pair, best_score

    def get_best_n(self, n: int, isNormalizedInts: bool = False) -> List[Tuple[PromptPair, float]]:
        if n <= 0:
            return []
        top_n = sorted(self._heap, key=lambda x: x[0], reverse=True)[:n]

        if isNormalizedInts:
            # Normalize scores to [60, 90] range
            min_score = 60
            max_score = 90

            raw_scores = [score for score, _ in top_n]
            mn, mx = min(raw_scores), max(raw_scores)

            if mx == mn:
                # everyone identical → give all max_score
                norm_scores = [max_score] * len(raw_scores)
            else:
                norm_scores = [
                    int(round((s - mn) / (mx - mn) *
                        (max_score - min_score) + min_score))
                    for s in raw_scores
                ]

            return [(pair, norm) for (_, pair), norm in zip(top_n, norm_scores)]

        return [(pair, score) for score, pair in top_n]

    def __len__(self) -> int:
        return len(self._heap)

    def __str__(self) -> str:
        ordered = sorted(self._heap, key=lambda x: x[0], reverse=True)
        return str([(pair, score) for score, pair in ordered])

    def get_roulette_wheel_selection(self, n: int, isNormalizedInts: bool = True) -> List[Tuple[PromptPair, float]]:
        """
        Perform roulette-wheel (fitness-proportional) selection without replacement.

        Args:
            n: number of items to select.

        Returns:
            A list of up to n (prompt_pair, score) tuples,
            selected without replacement according to fitness weights.
        """
        # Work on a temporary copy of the heap data
        pool = list(self._heap)  # each element is (score, prompt_pair)
        total_fitness = sum(score for score, _ in pool)
        selected: List[Tuple[PromptPair, float]] = []

        # Don't request more than available
        n = min(n, len(pool))

        for _ in range(n):
            # Normalize weights and pick one
            r = random.random() * total_fitness
            cum = 0.0
            for idx, (score, pair) in enumerate(pool):
                cum += score
                if cum >= r:
                    # select this individual
                    selected.append((pair, score))
                    # remove it from pool & update total fitness
                    total_fitness -= score
                    pool.pop(idx)
                    break

        if not selected:
            return []

        if not isNormalizedInts:
            # If we want raw scores, return them directly
            return selected

        # --- Step B: normalize the selected raw scores to [min_score..max_score] ints ---
        min_score = 60
        max_score = 90

        raw_scores = [score for (_, score) in selected]
        mn, mx = min(raw_scores), max(raw_scores)

        if mx == mn:
            # everyone identical → give all max_score
            norm_scores = [max_score] * len(raw_scores)
        else:
            norm_scores = [
                int(round((s - mn) / (mx - mn) * (max_score - min_score) + min_score))
                for s in raw_scores
            ]

        # --- Step C: assemble final list ---
        selected_normalized: List[Tuple[PromptPair, int]] = [
            (pair, norm)
            for (pair, _), norm in zip(selected, norm_scores)
        ]
        return selected_normalized

    def get_average_score(self, top_n) -> float:
        """
        Calculate the average score of the top n items in the queue.
        If there are fewer than n items, it averages all available.
        """
        if not self._heap:
            return 0.0
        top_items = self.get_best_n(top_n)
        if not top_items:
            return 0.0
        total_score = sum(score for _, score in top_items)
        return total_score / len(top_items)


def load_initial_prompts(path: str) -> List[InitialItem]:
    """
    Reads a text file where each line is of the form:
    ('neg', 'pos'), Score: 0.9364
    and returns a list of ((neg, pos), score) tuples.
    """
    results = []

    with open(path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                # Use regex to parse the line format: ('neg', 'pos'), Score: 0.9364
                pattern = r"\('([^']*)', '([^']*)'\), Score: ([\d.]+)"
                match = re.match(pattern, line)

                if match:
                    neg_prompt = match.group(1)
                    pos_prompt = match.group(2)
                    score = float(match.group(3))

                    prompt_pair = (neg_prompt, pos_prompt)
                    results.append((prompt_pair, score))
                else:
                    print(f"Warning: Could not parse line {line_num}: {line}")

            except Exception as e:
                print(f"Error parsing line {line_num}: {line}")
                print(f"Error details: {e}")
                continue

    return results
