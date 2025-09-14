# Initial meta prompt for the first iteration
meta_init_template = """The task is to generate 50 textual description templates for peripheral blood cells.
Each template must be a single string that contains only one placeholder, which is "<cell_type>".
The cell types are Basophil, Eosinophil, Lymphocyte, Monocyte, and Neutrophil.
The template string should be a complete sentence that becomes a meaningful description when the <cell_type> placeholder is filled.
Do not include any other placeholders, variables, or specific details about a single class of cell type's features.

These are the following features an expert would include in the description: Cell Size, Cell Shape, Nucleus Shape, Nuclear-Cytoplasmic Ratio, Chromatin-Density, Cytoplasm-Vacuole, Cytoplasm-Texture, Cytoplasm-Color, Granule-Type, Granule-Color, Granularity


Example output format:
prompts = [
    "An image of a peripheral blood cell of type <cell_type>",
    "A microscopic image of a <cell_type>",
    "A blood smear showing a <cell_type>", 
    "The cell size of the blood cell indicate a <cell_type>",
    "A blood cell segmentation indicated its likely a <cell_type>"
]

Only provide the output as Python code in the following format: prompts = list[str]. Let's think step-by-step
"""
# Meta prompt template for subsequent iterations
base_meta_prompt_template = """The task is to generate {generate_n} textual description templates for peripheral blood cells.
Each template must be a single string that contains only one placeholder, which is "<cell_type>".
The cell types are Basophil, Eosinophil, Lymphocyte, Monocyte, and Neutrophil.
The template string should be a complete sentence that becomes a meaningful description when the <cell_type> placeholder is filled.
Do not include any other placeholders, variables, or specific details about the cell's features.

These are the following features an expert would include in the description: Cell Size, Cell Shape, Nucleus Shape, Nuclear-Cytoplasmic Ratio, Chromatin-Density, Cytoplasm-Vacuole, Cytoplasm-Texture, Cytoplasm-Color, Granule-Type, Granule-Color, Granularity


Here are the best performing templates in ascending order. High scores indicate higher quality visual discriminative features.
{content}
Write {generate_n} new descriptions templates that are different from the old ones and has a score as high as possible, formulate a strategy.
Only provide the output as Python code in the following format: prompts = list[str]. Let's think step-by-step
"""
