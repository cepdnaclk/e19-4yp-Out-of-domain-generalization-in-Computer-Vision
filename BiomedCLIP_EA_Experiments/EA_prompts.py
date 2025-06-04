EXPERIMENT_1 = "EA_only"
PROMPT_1 = """\
            Please follow the instruction step-by-step to generate a better prompt pair.

            1. Cross over the following prompts and generate a new prompt:

            Prompt Pair 1: {pair1}
            Prompt Pair 2: {pair2}

            2. Mutate the prompt generated in Step 1 and generate a final prompt pair in a python tuple (str, str)
    """

EXPERIMENT_2 = "EA_only_with_pd"
PROMPT_2 = """\
            The task is to generate a textual descriptions pair of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section.
            Please follow the instruction step-by-step to generate a better prompt pair for the above task.

            1. Cross over the following prompts and generate a new prompt:

            Prompt Pair 1: {pair1}
            Prompt Pair 2: {pair2}

            2. Mutate the prompt generated in Step 1 and generate a final prompt pair in a python tuple (str, str)
    """

EXPERIMENT_3 = "EA_only_with_pd_v2"
PROMPT_3 =  """\
            TASK: Generate a textual descriptions pair of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section.
            Please follow the instruction step-by-step to generate a better prompt pair.

            1. Cross over the following prompts and generate a new prompt:

            Prompt Pair 1: {pair1}
            Prompt Pair 2: {pair2}

            2. Mutate the prompt generated in Step 1 and generate a final prompt pair in a python tuple (str, str)
    """

EXPERIMENT_4 = "EA-Only-with-reg"
PROMPT_4 = """\
            Please follow the instruction step-by-step to generate a better prompt pair for the above task.

            1. Cross over the following prompts and generate a new prompt:

            Prompt Pair 1: {pair1}
            Prompt Pair 2: {pair2}

            2. Mutate the prompt generated in Step 1 keeping the word count under 20 for each prompt and generate a final prompt pair in a python tuple (str, str)
    """

EXPERIMENT_5 = "EA-Only-reg-with-exemplar"
META_INIT_PROMPT_5 = """Give 50 textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. \
                The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section. \
                Following exemplar shows how to write the prompt pair: \
                    ("This is an image of a tumor", "Tumor is not present in this image") \
                Only give the output as python code in the format - prompts: list[tuple[negative: str, positive: str]]"""
PROMPT_5 = """\
            Please follow the instruction step-by-step to generate a better prompt pair for the above task.
            1. Cross over the following prompts and generate a new prompt:

            Prompt Pair 1: {pair1}
            Prompt Pair 2: {pair2}

            2. Mutate the prompt generated in Step 1 keeping the word count under 20 for each prompt and generate a final prompt pair in a python tuple (str, str)
    """

EXPERIMENT_6 = "EA-Only-reg-with-exemplar_2"
META_INIT_PROMPT_6 = """Give 50 textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. \
                The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section. \
                Following exemplar shows how to write the prompt pair: \
                    ("Tumor is not present in this image", "This is an image of a tumor") \
                Only give the output as python code in the format - prompts: list[tuple[negative: str, positive: str]]"""
PROMPT_6 = """\
            Please follow the instruction step-by-step to generate a better prompt pair for the above task.
            1. Cross over the following prompts and generate a new prompt:

            Prompt Pair 1: {pair1}
            Prompt Pair 2: {pair2}

            2. Mutate the prompt generated in Step 1 keeping the word count under 20 for each prompt and generate a final prompt pair in a python tuple (str, str)
    """