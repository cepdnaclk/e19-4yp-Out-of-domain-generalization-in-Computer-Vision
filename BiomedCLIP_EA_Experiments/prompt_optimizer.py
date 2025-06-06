"""
Optimization Only - No Evolutionary Algorithm (EA) - Prompt Optimization Script
"""
from typing import List
import util
import torch
import numpy as np
import os


def get_prompt_template(iteration_num: int, prompt_content: str, generate_n: int = 10) -> str:
    """
    Returns the appropriate instruction based on the iteration number range.

    Args:
        iteration_num: Current iteration number (1-indexed)

    Returns:
        String containing the iteration-specific instruction

    """
    # define a dictionary to map iteration ranges to instructions
    instruction_map = {
        "medical_concepts": f"Write {generate_n} new prompt pairs that are different from the old ones and has a score as high as possible.",
        "similar": f"Write {generate_n} new prompt pairs that are more similar to the high scoring prompts.",
        "combined_medical_concepts": f"Write {generate_n} new prompt pairs by combining multiple medical concepts only from the above prompts to make the score as high as possible.",
        "language_styles": f"Write {generate_n} new prompt pairs by paraphrasing each of the above. Each pair should have distinct language style.",
        "slight_changes": f"Write {generate_n} new prompt pairs similar to the above pairs only making slight changes to the language style to make the score as high as possible."
    }

    # Base meta prompt template
    base_meta_prompt_template = """The task is to generate textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section.
    Here are the best performing pairs in descending order. High scores indicate higher quality visual discriminative features.
    {content}
    {iteration_specific_instruction}
    Only give the output as python code in the format - prompts: list[tuple[negative: str, positive: str]]
    """

    if 1 <= iteration_num <= 2000:
        # Iterations 1-50: Basic exploration
        return base_meta_prompt_template.format(
            content=prompt_content,
            iteration_specific_instruction=instruction_map["medical_concepts"]
        )
    elif 2001 <= iteration_num <= 3000:
        # Iterations 51-100: Concept combination
        return base_meta_prompt_template.format(
            content=prompt_content,
            iteration_specific_instruction=instruction_map["similar"]
        )
    elif 3001 <= iteration_num <= 4000:
        # Iterations 101-200: Language style variation
        return base_meta_prompt_template.format(
            content=prompt_content,
            iteration_specific_instruction=instruction_map["combined_medical_concepts"]
        )
    elif iteration_num > 4000:
        # Iterations 201+: Fine-tuning with slight modifications
        return base_meta_prompt_template.format(
            content=prompt_content,
            iteration_specific_instruction=instruction_map["slight_changes"]
        )
    else:
        # Fallback (shouldn't happen with normal iteration numbering)
        raise IndexError("Error occured when getting prompt template")


def main():
    # Name the experiment we are currently running
    experiment_name = "Experiment-19-llm-client-gemma"
    print(f"Running {experiment_name}...")

    # Create experiment results directory
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)

    # Create filename with experiment name
    results_filename = os.path.join(
        results_dir, f"{experiment_name}_opt_pairs.txt")

   # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. load dataset
    # 1) Unpack—annotate what extract_center_embeddings returns
    centers_features: List[np.ndarray]
    centers_labels:   List[np.ndarray]
    centers_features, centers_labels = util.extract_center_embeddings(
        model=model,
        preprocess=preprocess,
        num_centers=3,  # trained only on center 0
    )

    # 2) Concatenate and convert—annotate the resulting tensors
    all_feats: torch.Tensor = torch.from_numpy(
        np.concatenate(centers_features, axis=0)
    ).float()   # shape: (N_total, D), dtype=torch.float32

    all_labels: torch.Tensor = torch.from_numpy(
        np.concatenate(centers_labels, axis=0)
    ).long()    # shape: (N_total,), dtype=torch.int64

    print("Center embeddings extracted successfully.")

    # 3. load initial prompts (optional)
    # initial_prompts = util.load_initial_prompts()

    # 4. Initialize the LLM client
    # Set use_local_ollama to True if you want to use a local Ollama server
    client = util.LLMClient(
        use_local_ollama=True, ollama_model="hf.co/unsloth/medgemma-27b-text-it-GGUF:Q4_K_M")

    # Configure the prompt templates
    meta_init_prompt = """Give 50 textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section. Only give the output as python code in the format - prompts: list[tuple[negative: str, positive: str]]"""

    # meta_prompt_template = """The task is to generate 50 textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section.
    # Here are the best performing pairs. You should aim to get higher scores. Each description should be about 5-20 words.
    # {content}
    # 1-10: Generate the first 10 pairs exploring variations of the top 1 (best) given. Remove certain words, add words, change order and generate variations
    # 11-20: Generate 10 pairs using the top 10, explore additional knowledge and expand on it.
    # 21-30: The next 10 pairs should maintain similar content as middle pairs but use different language style and sentence structures.
    # 31-40: The next 10 pairs should combine knowledge of top pairs and bottom pairs.
    # 41-50: The remaining 10 pairs should be randomly generated.
    # Only give the output as python code in the format - prompts: list[tuple[negative: str, positive: str]]
    # """

    # Optimization loop
    initial_prompts = [
        (
            "Uniform population of small lymphocytes with round nuclei and scant cytoplasm preserving nodal architecture",
            "Sheets of large pleomorphic cells with high nuclear-to-cytoplasmic ratio disrupting normal lymphoid architecture"
        ),
        (
            "Evenly spaced follicles with pale germinal centers and no evidence of cellular atypia",
            "Effacement of follicular architecture by monotonous population of atypical lymphoid cells with irregular nuclear contours"
        ),
        (
            "Fine, delicate trabecular stroma with clear sinusoids and minimal mitotic activity",
            "Dense, fibrotic stroma infiltrated by clusters of hyperchromatic tumor cells exhibiting frequent mitoses"
        ),
        (
            "Homogeneous distribution of small round lymphocytes with uniform chromatin and inconspicuous nucleoli",
            "Heterogeneous population of large cells with coarse chromatin, prominent nucleoli, and nuclear pleomorphism"
        ),
        (
            "Absence of necrosis and preservation of normal paracortical zones containing scattered immunoblasts",
            "Focal zones of coagulative necrosis surrounded by sheets of atypical lymphoid cells forming cohesive clusters"
        ),
        (
            "Delicate sinusoidal pattern with interspersed macrophages and no evidence of glandular structures",
            "Loss of sinusoidal pattern replaced by solid sheets of tumor cells forming abortive gland-like structures"
        ),
        (
            "Scant cytoplasm, small nuclei, and no prominent nucleoli in lymphoid cells",
            "Abundant eosinophilic cytoplasm, vesicular nuclei, and prominent nucleoli in malignant cells"
        ),
        (
            "Clear demarcation between cortex and medulla with well-formed germinal centers",
            "Blurring of cortical-medullary junction by infiltrative atypical cells permeating nodal tissue"
        ),
        (
            "Regularly spaced blood vessels with thin endothelial lining and no vascular proliferation",
            "Marked angiogenesis with proliferating endothelial cells and tumor cell nests encasing irregular vessels"
        ),
        (
            "Uniform staining pattern with light basophilic nuclei and no hyperchromasia",
            "Intensely hyperchromatic nuclei with coarse chromatin clumping and irregular chromatin distribution"
        ),
        (
            "Low nuclear-to-cytoplasmic ratio in lymphocytes with small, round, well-defined nuclei",
            "High nuclear-to-cytoplasmic ratio in tumor cells with large, irregular, and overlapping nuclei"
        ),
        (
            "Minimal variation in cell size, shape, and nuclear morphology across the field",
            "Marked anisocytosis and anisokaryosis with variation in size and shape of tumor cells and nuclei"
        ),
        (
            "Absence of multinucleated giant cells or Reed-Sternberg–like cells in the paracortex",
            "Presence of binucleated or multinucleated atypical cells with prominent nucleoli resembling Reed-Sternberg cells"
        ),
        (
            "Even distribution of lymphocytes without clusters of proliferating immunoblasts",
            "Confluent sheets of immunoblasts with open chromatin and prominent nucleoli forming clusters"
        ),
        (
            "Fine reticular network of stromal fibers with normal histiocyte distribution",
            "Dense collagenous stroma with frequent dust-like calcifications and infiltrating tumor cells"
        ),
        (
            "Sinusal pattern intact with histiocytes and macrophages prominently visible",
            "Sinusoidal spaces obliterated by solid sheets of malignant lymphoid cells"
        ),
        (
            "No perivascular cuffing or perisinusoidal infiltration by atypical cells",
            "Perivascular aggregates of large atypical cells forming cuff-like structures around vessels"
        ),
        (
            "Cytologically bland lymphocytes with smooth nuclear membranes and no nucleolar prominence",
            "Pleomorphic cells with irregular nuclear membranes, indentations, and conspicuous nucleoli"
        ),
        (
            "Regular arrangement of reticular fibers and no desmoplastic reaction",
            "Disorganized reticular framework replaced by dense desmoplastic reaction and tumor cell nests"
        ),
        (
            "Absence of mitotic figures or presence of only occasional normal mitoses",
            "Numerous atypical mitotic figures scattered throughout the field indicating high proliferative index"
        ),
        (
            "Uniform pale eosinophilic cytoplasm and small, round nuclei without vesicular changes",
            "Dense basophilic cytoplasm and vesicular nuclear changes with open chromatin in malignant cells"
        ),
        (
            "Normal lymph node parenchyma showing cortical lymphoid follicles and medullary cords",
            "Diffuse infiltration of medullary cords by large lymphoid cells with irregular chromatin distribution"
        ),
        (
            "Preserved capsule and subcapsular sinus without evidence of capsular invasion",
            "Capsular breach by aggregates of tumor cells extending beyond the node boundary"
        ),
        (
            "Scant interstitial edema and lack of inflammatory cell infiltrates beyond baseline",
            "Marked interstitial edema with mixed inflammatory infiltrate secondary to tumor-induced cytokine release"
        ),
        (
            "Smooth, well-defined nuclear borders and absence of nuclear grooves in lymphocytes",
            "Irregular nuclear borders with deep grooves and folds in malignant cells"
        ),
        (
            "No evidence of lymphoepithelial lesions or destruction of native tissue structures",
            "Lymphoepithelial lesions with invasion and destruction of normal nodal tissue by tumor cells"
        ),
        (
            "Normal sinus histiocytosis with scattered pale macrophages and no cellular atypia",
            "Proliferation of pale-staining histiocytes intermingled with atypical malignant cells"
        ),
        (
            "Evenly distributed chromatin with no coarse clumping or clearing",
            "Clumped or cleared chromatin with irregular distribution and chromatin margination in tumor nuclei"
        ),
        (
            "Absence of vascular proliferation and endothelial cell hyperplasia",
            "Marked endothelial hyperplasia with irregular, dilated vascular channels infiltrated by tumor cells"
        ),
        (
            "Regularly arranged reticulum cells with no cytoplasmic vacuolization",
            "Spindle-shaped tumor cells with vacuolated cytoplasm forming irregular fascicles"
        ),
        (
            "No presence of necrotic debris or karyorrhectic nuclear fragments",
            "Areas of karyorrhexis and necrotic debris amidst clusters of atypical cells"
        ),
        (
            "Lymphocytes exhibit uniform nuclear size and shape without membrane irregularities",
            "Tumor cells show nuclear membrane irregularities, indentations, and lobulations"
        ),
        (
            "Intact capsule with minimal subcapsular sinus expansion and absence of tumor nodules",
            "Distended subcapsular sinus filled with irregularly shaped tumor nodules protruding against the capsule"
        ),
        (
            "Well-defined germinal centers with tingible body macrophages and no atypical blast cells",
            "Germinal centers replaced by sheets of atypical blast-like cells lacking tingible body macrophages"
        ),
        (
            "No paracortical hyperplasia; paracortex composed of mature lymphocytes with few immunoblasts",
            "Paracortical expansion with clusters of immunoblasts and large atypical cells showing mitoses"
        ),
        (
            "Sinus histiocytes with pale cytoplasm interspersed among lymphocytes and no epithelial clusters",
            "Clusters of epithelial-like cells with eosinophilic cytoplasm forming alveolar patterns within the sinus"
        ),
        (
            "Regularly spaced nuclei with smooth chromatin and absent nucleolar prominence",
            "Irregular nuclear size with prominent nucleoli and clumped chromatin in tumor cells"
        ),
        (
            "No evidence of capsular or perinodal fat invasion by lymphoid cells",
            "Tumor cells breaching capsule invading perinodal adipose tissue forming irregular cell sheets"
        ),
        (
            "Absence of lymphovascular invasion; vessels clear of atypical cells",
            "Tumor emboli visible within lymphatic channels and small blood vessels"
        ),
        (
            "Uniform distribution of small, mature lymphocytes without variation in staining intensity",
            "Mixed staining intensity with hyperchromatic nuclei and pale cytoplasm in malignant cells"
        ),
        (
            "Preserved trabecular bone marrow interface without tumor cell infiltration",
            "Bone marrow interface disrupted by infiltration of atypical lymphoid cells"
        ),
        (
            "No fibrous bands traversing the node; delicate reticular meshwork present",
            "Thick fibrous bands splitting nodal parenchyma with tumor cells aligned along bands"
        ),
        (
            "Absence of Reed-Sternberg–like binucleated cells and lacunar histiocytes",
            "Presence of lacunar histiocytes and lacunar variant Reed-Sternberg cells amidst mixed infiltrate"
        ),
        (
            "Proliferation index (Ki-67) low with few cells in cell cycle",
            "Elevated proliferation index with numerous Ki-67–positive tumor nuclei indicating high mitotic activity"
        ),
        (
            "Regularly organized plasma cells in medullary cords with no plasmablastic features",
            "Sheets of large plasmablast-like cells with eccentric nuclei and prominent nucleoli replacing cords"
        ),
        (
            "Absence of necrotizing granulomas or histiocytic infiltration beyond normal range",
            "Necrotizing granulomas with central caseation and rim of atypical histiocytes and malignant lymphocytes"
        ),
        (
            "Uniform small germinal centers with polarized dark and light zones intact",
            "Disruption of germinal center polarization with monomorphic infiltrate and absent mantle zones"
        ),
        (
            "Normal follicular dendritic cell network present without regression or pale zones",
            "Disrupted follicular dendritic cell meshwork with regressive pale zones replaced by tumor cells"
        ),
        (
            "Intact capsular vessels with no perivascular tumor cell cuffing",
            "Perivascular cuffing of atypical cells around capsular and medullary vessels"
        ),
        (
            "Absence of sclerosis or broad fibrotic bands dividing nodal tissue",
            "Broad fibrotic bands creating nodular sclerosis pattern with tumor cells trapped in collagen"
        ),
        (
            "No evidence of epithelial infiltration; parenchyma devoid of cohesive epithelial clusters",
            "Cohesive clusters of epithelial-like tumor cells invading lymphoid parenchyma"
        ),
        (
            "Regular, small mitochondria-rich lymphocyte cytoplasm faintly basophilic",
            "Abundant pale, vacuolated cytoplasm in tumor cells with perinuclear clearing"
        ),
        (
            "Absence of apoptotic bodies and karyolysis among lymphoid cells",
            "Numerous apoptotic bodies and karyolytic nuclei indicating high turnover in malignant population"
        ),
        (
            "Fine granular chromatin with inconspicuous nucleoli across lymphocytes",
            "Coarse granular chromatin with conspicuous central nucleoli in neoplastic cells"
        ),
        (
            "No evidence of spindle cell proliferation or sarcomatoid features",
            "Spindle-shaped fusiform tumor cells forming irregular fascicles and whorled patterns"
        )
    ]

    pq = util.PriorityQueue(max_capacity=1000)
    prompt_content = ""

    for j in range(2000):
        if j == 0:
            prompts = util.get_prompt_pairs(meta_init_prompt, client)
        else:
            meta_prompt = get_prompt_template(
                iteration_num=j, prompt_content=prompt_content, generate_n=10)

            prompts = util.get_prompt_pairs(meta_prompt, client)

        for i, prompt_pair in enumerate(prompts):
            if len(prompt_pair) != 2:
                print(f"Invalid prompt pair: {prompt_pair}")
                continue
            negative_prompt, positive_prompt = prompt_pair
            results = util.evaluate_prompt_pair(
                negative_prompt, positive_prompt, all_feats, all_labels, model, tokenizer)

            pq.insert((negative_prompt, positive_prompt), results['accuracy'])

        n = 10
        print(f"\nCurrent Top {n} prompt pairs:")

        # Selector Operator: Roulette Wheel Selection or Best N Prompts
        # Use Roulette Wheel Selection for the first 250 iterations
        # After 250 iterations, use the best n prompts
        # This is to ensure diversity in the early stages of optimization
        # if j < 250:
        #     selected_prompts = pq.get_roulette_wheel_selection(
        #         n, isNormalizedInts=False)
        # else:
        #     selected_prompts = pq.get_best_n(n)

        selected_prompts = pq.get_roulette_wheel_selection(
            n, isNormalizedInts=True)
        # selected_prompts = pq.get_best_n(n)
        # reverse the order to set it to acsending order: Recency Bias
        selected_prompts = sorted(
            selected_prompts, key=lambda x: x[1], reverse=True)

        # Prepare the content for the meta prompt
        prompt_content = f"Current Top {n} prompt pairs:\n"
        for i, (prompt_pair, score) in enumerate(selected_prompts):
            print(f"{i+1}. {prompt_pair}, Score: {score}")
            prompt_content += f"{i+1}. {prompt_pair}, Score: {score:.2f}\n"

        # Save the best prompt pairs to a file, every 10 iterations
        if (j + 1) % 20 == 0 or j == 0:
            top_prompts = pq.get_best_n(1000)
            with open(results_filename, "a") as f:
                f.write(f"Iteration {j+1}:\n")
                for prompt_pair, score in top_prompts:
                    f.write(f"{prompt_pair}, Score: {score:.4f}\n")
                f.write("\n")

        # print the average score of the top n prompts
        print(
            f"Iteration {j+1}: mean accuracy of top 10: {pq.get_average_score(10)}.\n")


if __name__ == "__main__":
    main()
