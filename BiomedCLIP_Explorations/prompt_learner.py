import tokenize
import io
import re
import ast
from typing import List, Tuple
import os
from gemini import Gemini
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from tqdm import tqdm
import heapq
from typing import Tuple, List, Optional
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.multiprocessing.set_sharing_strategy('file_system')

# 1. Paths & constants
METADATA_CSV = "/home/E19_FYP_Domain_Gen_Data/metadata.csv"
PATCHES_DIR = "/home/E19_FYP_Domain_Gen_Data/patches"
CONFIG_PATH = "../BioMedClip/checkpoints/open_clip_config.json"
WEIGHTS_PATH = "../BioMedClip/checkpoints/open_clip_pytorch_model.bin"
MODEL_NAME = "biomedclip_local"
CONTEXT_LENGTH = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_prompt_pairs(prompt_template: Tuple[str, str], content: str, client, max_retries=10) -> List[Tuple[str, str]]:

    if content == "":
        prompt = prompt_template[0] + "\n" + prompt_template[2]
    else:
        prompt = prompt_template[0] + "\n" + prompt_template[1] + \
            "\n" + content + "\n" + prompt_template[2]
    for attempt in range(1, max_retries + 1):
        try:
            response = client.generate_content(prompt)
            raw = response.text

            # 1) extract the python block
            m = re.search(r'```python\s*(.*?)\s*```', raw, re.S)
            code = m.group(1) if m else raw

            # 2) normalize all literals to double-quoted form
            code = _force_double_quotes(code)

            # 3) parse and extract
            tree = ast.parse(code)
            prompts_list = None
            for node in tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'prompts' for t in node.targets)
                ):
                    prompts_list = ast.literal_eval(node.value)
                    break

            if not isinstance(prompts_list, list):
                raise ValueError("`prompts` is not a list")

            prompts: List[Tuple[str, str]] = prompts_list  # type: ignore
            print(f"Loaded {len(prompts)} prompt-pairs.")
            print("First pair:", prompts[0])
            return prompts

        except Exception as e:
            print(
                f"[Warning] get_prompt_pairs parse error on attempt {attempt}/{max_retries}: {e}")
            if attempt == max_retries:
                raise RuntimeError(
                    "Failed to parse prompts after multiple attempts") from e
            # otherwise, retry immediately

    # Should never reach here
    raise RuntimeError("Unreachable")


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
        y_prob = probs[:, 0].cpu().numpy()    # tumor-class prob
        y_true = labels.cpu().numpy()

    # metrics
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    return {'accuracy': acc, 'auc': auc, 'cm': cm, 'report': report}


# def evaluate_prompt_pair_with_adapter(
#     negative_prompt,
#     positive_prompt,
#     image_feats: torch.Tensor,    # (N, D), precomputed
#     image_labels: torch.Tensor,   # (N,)
#     model,
#     tokenizer,
#     adapter,
# ):

#     # 2. Encode & adapt text prompts
#     # --------------------------------
#     text_inputs = tokenizer(
#         [negative_prompt, positive_prompt],
#         context_length=CONTEXT_LENGTH
#     ).to(DEVICE)

#     with torch.no_grad():
#         text_feats = model.encode_text(text_inputs)          # (2, D)
#         text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
#         text_feats = adapter(text_feats)                     # adapt
#         text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

#         logit_scale = model.logit_scale.exp()

#         # move image feats to DEVICE for one matrix multiply
#         feats = image_feats.to(DEVICE)                        # (N, D)
#         labels = image_labels.to(DEVICE)

#         # adapter network
#         feats = adapter(feats)                     # adapt
#         # (N,)
#         feats = feats / feats.norm(dim=1, keepdim=True)

#         # compute all logits at once: (N, 2)
#         logits = logit_scale * (feats @ text_feats.t())
#         probs = logits.softmax(dim=1)
#         preds = logits.argmax(dim=1)

#         y_pred = preds.cpu().numpy()
#         y_prob = probs[:, 0].cpu().numpy()    # tumor-class prob
#         y_true = labels.cpu().numpy()

#     acc = accuracy_score(y_true, y_pred)
#     auc = roc_auc_score(y_true, y_prob)
#     cm = confusion_matrix(y_true, y_pred)
#     report = classification_report(y_true, y_pred, digits=4)

#     return {'accuracy': acc, 'auc': auc, 'cm': cm, 'report': report}


PromptPair = Tuple[str, str]
InitialItem = Tuple[PromptPair, float]


class PriorityQueue:
    # type: ignore
    def __init__(self, max_capacity: int = 10, initial: Optional[List[Tuple[PromptPair, float]]] = None):

        self.max_capacity: int = max_capacity
        # Store (score, prompt_pair); min-heap root is the lowest score.
        self._heap: List[Tuple[float, PromptPair]] = []  # type: ignore
        # Track negative prompts for O(1) membership checks
        self._neg_set: set[str] = set()

        # If the user passed some initial prompt‐pairs, insert them now:
        if initial is not None:
            for prompt_pair, score in initial:
                self.insert(prompt_pair, score)

    def insert(self, prompt_pair: PromptPair, score: float) -> None:  # type: ignore
        negative = prompt_pair[1]
        # Skip if negative prompt already exists
        if negative in self._neg_set:
            return
        # Skip low scores
        if score < 0.5:
            return

        if len(self._heap) < self.max_capacity:
            # Add new entry
            heapq.heappush(self._heap, (score, prompt_pair))
            self._neg_set.add(negative)
        else:
            # Only replace if new score beats the current minimum
            if score > self._heap[0][0]:
                # Replace smallest entry, capturing the popped item
                old_score, old_pair = heapq.heapreplace(
                    self._heap, (score, prompt_pair))
                # Update negative-prompt set
                self._neg_set.remove(old_pair[1])
                self._neg_set.add(negative)

    def get_best(self) -> Optional[Tuple[PromptPair, float]]:
        if not self._heap:
            return None
        best_score, best_pair = max(self._heap, key=lambda x: x[0])
        return best_pair, best_score

    def get_best_n(self, n: int) -> List[Tuple[PromptPair, float]]:
        if n <= 0:
            return []
        top_n = sorted(self._heap, key=lambda x: x[0], reverse=True)[:n]
        return [(pair, score) for score, pair in top_n]

    def __len__(self) -> int:
        return len(self._heap)

    def __str__(self) -> str:
        ordered = sorted(self._heap, key=lambda x: x[0], reverse=True)
        return str([(pair, score) for score, pair in ordered])


def load_initial_prompts(path: str) -> List[InitialItem]:
    """
    Reads a text file where each line is of the form:
      1. ('neg', 'pos'), Score: 0.9364
    and returns a list of ((neg, pos), score) tuples.
    """
    initial: List[InitialItem] = []
    line_re = re.compile(r"""
        ^\s*\d+\.       # leading index and dot
        \s*(\(.+\))     # group(1): the tuple literal "('neg','pos')"
        \s*,\s*Score:\s*
        ([0-9]+\.[0-9]+)  # group(2): the floating score
        """, re.VERBOSE)

    with open(path, 'r') as f:
        for line in f:
            m = line_re.match(line)
            if not m:
                continue
            pair_literal, score_str = m.groups()
            try:
                prompt_pair = ast.literal_eval(pair_literal)
                score = float(score_str)
                # validate
                if (
                    isinstance(prompt_pair, tuple)
                    and len(prompt_pair) == 2
                    and all(isinstance(s, str) for s in prompt_pair)
                ):
                    initial.append((prompt_pair, score))
            except Exception:
                # skip malformed lines
                continue

    return initial


def main():
    random_state = 42

    # 2. Load metadata and filter center=0
    metadata_df = pd.read_csv(METADATA_CSV, index_col=0)
    metadata_df = append_filename_and_filepath(metadata_df)

    train_df = pd.concat([
        metadata_df[(metadata_df['tumor'] == 1) &
                    (metadata_df['center'] == 0)],
        metadata_df[(metadata_df['tumor'] == 0) &
                    (metadata_df['center'] == 0)],
        metadata_df[(metadata_df['tumor'] == 1) &
                    (metadata_df['center'] == 1)],
        metadata_df[(metadata_df['tumor'] == 0) &
                    (metadata_df['center'] == 1)],
        metadata_df[(metadata_df['tumor'] == 1) &
                    (metadata_df['center'] == 2)],
        metadata_df[(metadata_df['tumor'] == 0) &
                    (metadata_df['center'] == 2)],
    ]).reset_index(drop=True)

    # Load BiomedCLIP model + tokenizer + preprocess
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    model_cfg, preproc_cfg = cfg["model_cfg"], cfg["preprocess_cfg"]

    # register local config if needed
    if (not MODEL_NAME.startswith(HF_HUB_PREFIX)
            and MODEL_NAME not in _MODEL_CONFIGS):
        _MODEL_CONFIGS[MODEL_NAME] = model_cfg

    tokenizer = get_tokenizer(MODEL_NAME)
    model, _, preprocess = create_model_and_transforms(
        model_name=MODEL_NAME,
        pretrained=WEIGHTS_PATH,
        **{f"image_{k}": v for k, v in preproc_cfg.items()}
    )

    model = model.to(DEVICE).eval()

    # load adapter
    adapter = Adapter(dim=512).to(DEVICE)
    adapter.load_state_dict(torch.load(
        "adapter_weights.pth", map_location=DEVICE))
    adapter.eval()

    # 7. Prepare DataLoader for test set
    train_ds = BiomedCLIPDataset(train_df, preprocess)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    # ——— Cache/load precomputed features ———
    cache_path = "cached_image_feats.pt"
    if os.path.exists(cache_path):
        print(f"Loading cached image features from {cache_path}")
        data = torch.load(cache_path)
        all_img_feats, all_img_labels = data["feats"], data["labels"]
    else:
        print("Precomputing image features …")
        all_img_feats = []
        all_img_labels = []
        with torch.no_grad():
            for imgs, labels in tqdm(train_loader, desc="Precompute Feats"):
                imgs = imgs.to(DEVICE)
                feats = model.encode_image(imgs)                    # (B, D)
                feats = feats / feats.norm(dim=1, keepdim=True)
                all_img_feats.append(feats.cpu())
                all_img_labels.append(labels)
        all_img_feats = torch.cat(all_img_feats, dim=0)   # (N, D)
        all_img_labels = torch.cat(all_img_labels, dim=0)  # (N,)

        # Save to cache
        # print(f"Saving cached features to {cache_path}")
        # torch.save(
        # {"feats": all_img_feats, "labels": all_img_labels},
        # cache_path
        # )

    cookies = {"__Secure-1PSIDCC": "8WqUIAmsCWWrmWr-/AqzGpTdQDEvsWgOSP",
               "__Secure-1PSID": "g.a000xAhtcFFJw-Pe2SfxFzHOJXUMClrKicX6q_b7mFELwJZbSoGutGYNkxA8kyX1FZpLmh29jwACgYKAXESARASFQHGX2Mi7J2NGrnruG68cQI02g7H6BoVAUF8yKqFxE1MZio3JvWDuqqc2aS90076",
               "__Secure-1PSIDTS": "AKEyXzW9DtAugRds_seZfS4OUpDvWkPzJFmyEjYz-Ytr-zQpaQ_8j4Ujce8w5aN4HjfI7Erxnmae",
               }  # Cookies may vary by account or region. Consider sending the entire cookie file.

    client = Gemini(auto_cookies=False, cookies=cookies)

    prompt_template = ["""Give 50 textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section.""",
                       """Here are the best performing pairs. You should aim to get higher scores. Each description should be about 5-20 words.
                        1-10: Generate the first 10 pairs exploring variations of the top 1 (best) given. Remove certain words, add words, change order and generate variations.
                        11-20: Generate 10 pairs using the top 10, explore additional knowledge and expand on it. 
                        21-30: The next 10 pairs should maintain similar content as middle pairs but use different language style and sentence structures. 
                        31-40: The next 10 pairs should combine knowledge of top pairs and bottom pairs.
                        41-50: The remaining 10 pairs should be randomly generated. 
                        """,
                       """Only give the output as python code in the format - prompts: list[tuple[negative: str, positive: str]]"""]

    # initial_list = load_initial_prompts("selected_prompts.txt")
    # pq = PriorityQueue(max_capacity=40, initial=initial_list)
    pq = PriorityQueue(max_capacity=40)
    prompt_llm = ""
    for j in range(100):
        prompts = get_prompt_pairs(prompt_template, prompt_llm, client)

        for i, prompt_pair in enumerate(prompts):
            if len(prompt_pair) != 2:
                print(f"Invalid prompt pair: {prompt_pair}")
                continue
            negative_prompt, positive_prompt = prompt_pair
            results = evaluate_prompt_pair(
                negative_prompt, positive_prompt, all_img_feats, all_img_labels, model, tokenizer, adapter)
            pq.insert((negative_prompt, positive_prompt), results['accuracy'])

        n = 40
        print(f"\nCurrent Top {n} prompt pairs:")
        top_n = pq.get_best_n(n)
        prompt_llm = f"Current Top {n} prompt pairs:\n"
        for i, (prompt_pair, score) in enumerate(top_n):
            print(f"{i+1}. {prompt_pair}, Score: {score:.4f}")
            prompt_llm += f"{i+1}. {prompt_pair}, Score: {score:.4f}\n"

        # Save the best prompt pairs to a file
        with open("best_prompt_pairs.txt", "a") as f:
            f.write(f"Iteration {j+1}:\n")
            for prompt_pair, score in top_n:
                f.write(f"{prompt_pair}, Score: {score:.4f}\n")
            f.write("\n")


if __name__ == "__main__":
    from train_adapter import (Adapter)
    main()
