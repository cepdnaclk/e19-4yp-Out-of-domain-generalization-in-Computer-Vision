from API_KEY import GEMINI_API_KEY, CHATGPT_API_KEY, CHATGPT_ENDPOINT
from openai import AzureOpenAI
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

# 1. Paths & constants
METADATA_CSV = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/metadata.csv"
PATCHES_DIR = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/patches"

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


class BiomedCLIPDataset(Dataset):
    def __init__(self, df, preprocess):
        self.filepaths = df["filepath"].tolist()
        self.labels = df["tumor"].astype(int).tolist()
        self.preproc = preprocess

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        with Image.open(self.filepaths[idx]) as img:
            img = img.convert("RGB")
            img = self.preproc(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def append_filename_and_filepath(df):
    df["filename"] = df.apply(
        lambda r: f"patch_patient_{r.patient:03d}_node_{r.node}_x_{r.x_coord}_y_{r.y_coord}.png",
        axis=1
    )
    df["filepath"] = df.apply(
        lambda r: os.path.join(
            PATCHES_DIR,
            f"patient_{r.patient:03d}_node_{r.node}",
            r.filename
        ),
        axis=1
    )
    return df


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


def extract_center_embeddings(
    model: torch.nn.Module,
    preprocess: Callable,
    num_centers: int = 1,
    metadata_csv: str = METADATA_CSV,
    isTrain: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Reads metadata, splits into centers, and computes normalized image embeddings.

    Args:
        metadata_csv: Path to your metadata CSV.
        preprocess: Preprocessing function for images.
        model: A CLIP-style model with .encode_image().
        device: torch.device (e.g. torch.device("cuda")).
        batch_size: DataLoader batch size.
        num_workers: Number of DataLoader workers.
        num_centers: How many 'center' values you expect (default 4).

    Returns:
        centers_features: List of length `num_centers`, each a tensor of shape
                          [N_i, D] with normalized embeddings.
        centers_labels:   List of length `num_centers`, each a tensor of shape
                          [N_i] with corresponding labels.
    """
    # Load and annotate metadata
    metadata_df = pd.read_csv(metadata_csv, index_col=0)
    metadata_df = append_filename_and_filepath(metadata_df)

    # Filter for training split 0: train, 1: test
    df = metadata_df[metadata_df.split == int(0 if isTrain else 1)]

    # Build datasets per center
    centers_ds = [
        BiomedCLIPDataset(df[df.center == i], preprocess)
        for i in range(num_centers)
    ]

    centers_features: List[torch.Tensor] = [[]
                                            for _ in range(num_centers)]
    centers_labels:   List[torch.Tensor] = [[]
                                            for _ in range(num_centers)]

    os.makedirs(f"{CACHE_PATH}/centers", exist_ok=True)
    for i, ds in enumerate(centers_ds):
        if isTrain:
            feature_path = f"{CACHE_PATH}/centers/train-center{i}_features.npy"
            label_path = f"{CACHE_PATH}/centers/train-center{i}_labels.npy"
        else:
            feature_path = f"{CACHE_PATH}/centers/test-center{i}_features.npy"
            label_path = f"{CACHE_PATH}/centers/test-center{i}_labels.npy"

        if os.path.exists(feature_path) and os.path.exists(label_path):
            print(f"Loading cached features for center {i}...")
            centers_features[i] = np.load(feature_path)
            centers_labels[i] = np.load(label_path)
            continue

        print(f"Extracting features for center {i}...")

        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

        all_feats = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=f"Center {i}", unit="batch"):
                imgs = imgs.to(DEVICE, non_blocking=True)
                feats = model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)

                all_feats.append(feats.cpu())
                all_labels.append(labels)  # labels are already on CPU

        centers_features[i] = torch.cat(all_feats).numpy()
        centers_labels[i] = torch.cat(all_labels).numpy()

        np.save(feature_path, centers_features[i])
        np.save(label_path, centers_labels[i])

    return centers_features, centers_labels


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
            target=torch.tensor(y_true, device=DEVICE).float()
        ).item()

        # Invert BCE loss: 1/(1 + loss) (so lower loss → higher value)
        inverted_bce = 1.0 / (1.0 + bce_loss)

    # metrics
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    return {'accuracy': acc, 'auc': auc, 'cm': cm, 'report': report, 'inverted_bce': inverted_bce}


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


def _force_double_quotes(code: str) -> str:
    """
    Rewrites every Python string-literal in `code` to use double-quotes,
    properly handling apostrophes and other special characters.
    """
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    new_tokens = []
    for toknum, tokval, start, end, line in tokens:
        if toknum == tokenize.STRING:
            # Get the actual string value
            value = ast.literal_eval(tokval)

            # Create a new string literal with double quotes
            # Properly escape any double quotes or backslashes in the string
            # This automatically handles escaping correctly
            tokval = json.dumps(value)

        new_tokens.append((toknum, tokval))
    return tokenize.untokenize(new_tokens)


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
    Extracts and parses the list defined after 'prompts:' in the format:
        prompts:[
            (("neg", "pos"), score),
            ...
        ]

    Returns a list of (neg, pos, score) tuples.
    Raises ValueError if malformed.
    """
    # Step 1: Find the content inside prompts:[ ... ]
    match = re.search(r'prompts\s*:\s*(\[\s*[\s\S]*?\])', code)
    if not match:
        raise ValueError("Could not find 'prompts:[...]' in the code")

    list_str = match.group(1)

    # Step 2: Parse using ast.literal_eval
    try:
        data = ast.literal_eval(list_str)
    except Exception as e:
        raise ValueError(f"Could not parse prompt list: {e}")

    # Step 3: Validate and extract
    parsed = []
    for item in data:
        if (
            isinstance(item, tuple) and len(item) == 2 and
            isinstance(item[0], tuple) and len(item[0]) == 2 and
            all(isinstance(x, str) for x in item[0]) and
            isinstance(item[1], (int, float))
        ):
            neg, pos = item[0]
            score = float(item[1])
            parsed.append((neg, pos, score))
        else:
            raise ValueError(f"Invalid format in item: {item}")

    return parsed


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


def _force_double_quotes(code: str) -> str:
    """
    Rewrites every Python string-literal in `code` to use double-quotes,
    properly handling apostrophes and other special characters.
    """
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    new_tokens = []
    for toknum, tokval, start, end, line in tokens:
        if toknum == tokenize.STRING:
            # Get the actual string value
            value = ast.literal_eval(tokval)

            # Create a new string literal with double quotes
            # Properly escape any double quotes or backslashes in the string
            # This automatically handles escaping correctly
            tokval = json.dumps(value)

        new_tokens.append((toknum, tokval))
    return tokenize.untokenize(new_tokens)


class LLMClient:
    """
    A unified client for interacting with Gemini, Ollama, or Azure OpenAI (ChatGPT).
    Usage:
        client = LLMClient(provider="gemini", ...)
        client = LLMClient(provider="ollama", ...)
        client = LLMClient(provider="azure_openai", ...)
        response = client.get_llm_response(prompt)
    """

    def __init__(
        self,
        provider: str = "gemini",  # "gemini", "ollama", or "azure_openai"
        gemini_model: str = "gemma-3-27b-it",
        ollama_model: str = "hf.co/unsloth/medgemma-27b-text-it-GGUF:Q8_0",
        azure_openai_model: str = "gpt-4.1",
        azure_openai_endpoint: str = CHATGPT_ENDPOINT,
        azure_openai_api_key: str = CHATGPT_API_KEY,
        azure_openai_api_version: str = "2025-01-01-preview"
    ):
        self.provider = provider.lower()
        self.gemini_model = gemini_model
        self.ollama_model = ollama_model
        self.azure_openai_model = azure_openai_model

        # Gemini
        self.gemini_client = None
        if self.provider == "gemini":
            if GEMINI_API_KEY:
                self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            else:
                raise ValueError("Gemini API key must be provided.")

        # Ollama
        self.ollama_host = "http://[::1]:11434"
        self._ollama_client_instance = None
        if self.provider == "ollama":
            self._ollama_client_instance = ollama.Client(host=self.ollama_host)

        # Azure OpenAI
        self.azure_openai_client = None
        if self.provider == "azure_openai":
            self.azure_openai_client = AzureOpenAI(
                azure_endpoint=azure_openai_endpoint,
                api_key=azure_openai_api_key,
                api_version=azure_openai_api_version,
            )

    def _get_response_from_gemini(self, prompt: str) -> str:
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized.")
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model, contents=prompt)
        return response.text

    def _get_response_from_ollama(self, prompt: str) -> str:
        if not self._ollama_client_instance:
            raise RuntimeError("Ollama client not initialized.")
        response = self._ollama_client_instance.chat(
            model=self.ollama_model, messages=[
                {"role": "user", "content": prompt}]
        )
        return response['message']['content']

    def _get_response_from_azure_openai(self, prompt: str) -> str:
        if not self.azure_openai_client:
            raise RuntimeError("Azure OpenAI client not initialized.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        completion = self.azure_openai_client.chat.completions.create(
            model=self.azure_openai_model,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )

        # Parse the response to JSON, then extract the content
        response = json.loads(completion.to_json())[
            'choices'][0]['message']['content']
        return response

    def get_llm_response(self, prompt: str) -> str:
        """
        Gets a response from the configured LLM provider.
        Args:
            prompt: Either a string (for Gemini/Ollama) or a list of messages (for Azure OpenAI).
        Returns:
            The response text from the LLM.
        """
        if self.provider == "gemini":
            return self._get_response_from_gemini(prompt)
        elif self.provider == "ollama":
            return self._get_response_from_ollama(prompt)
        elif self.provider == "azure_openai":
            return self._get_response_from_azure_openai(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


def get_prompt_pairs(
    prompt: str,
    llm_client: LLMClient,  # Accept the unified LLMClient instance
    parse_func: Callable = extract_and_parse_prompt_list,
    max_retries: int = 10
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
            print(f"Raw response on attempt {attempt}: {raw}...")

            # 1) extract the python block

            m = re.search(r'```python\s*([\s\S]*?)\s*```', raw)
            if not m:
                raise ValueError("No ```python ... ``` block found")
            code = m.group(1)

            # 2) normalize all literals to double-quoted form
            code = _force_double_quotes(code)

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
    def __init__(self, max_capacity: int = 10, initial: Optional[List[Tuple[PromptPair, float]]] = None, filter_threshold: float = 0.6):

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

    def delete_top_n(self, n: int) -> None:
        """
        Remove the top n items from the priority queue.
        If n exceeds the current size, it removes all items.
        """
        if n <= 0:
            return
        for _ in range(min(n, len(self._heap))):
            if self._heap:
                score, pair = heapq.heappop(self._heap)
                self._set.remove(pair)


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
