# BioMedXPro: Concept-Driven Island Evolution for Biomedical VLM Adapters

**BioMedXPro** is an evolutionary framework designed to optimize Vision-Language Model (VLM) prompts for biomedical tasks (e.g., Melanoma detection, Histopathology classification).

## 1. The Problem: Mode Collapse in Prompt Engineering

Standard automated prompt engineering suffers from **Mode Collapse**. When evolving prompts for a complex medical task (e.g., Melanoma), the optimization process naturally converges on the single most "lucrative" feature (e.g., "Dark Color") because it yields the easiest fitness gains.

As a result, the final model ignores subtle but critical diagnostic features (e.g., "Blue-Whitish Veil" or "Asymmetric Streaks"), leading to poor generalization and lack of interpretability.

## 2. The Solution: Concept-Driven Island Evolution

We solve this by restructuring the evolutionary process into an **Island Model**. Instead of one large mixed population, we maintain several isolated populations ("Islands"), where each island is constrained to a specific visual medical concept.

### Key Features

* **Diversity by Design:** An island dedicated to "Texture" will strictly evolve texture-related prompts. It cannot be invaded or out-competed by "Color" prompts.
* **LLM as a Directed Operator:** The LLM is used not as a random text generator, but as a concept-aware operator that mutates prompts while adhering to the island's semantic constraints.
* **Ensemble Consensus:** The final model is a weighted ensemble of the "Champions" from each island, providing a decision that considers all diagnostic criteria.

### Algorithm Flow

1. **Concept Discovery:** The system identifies key diagnostic concepts (e.g., "Pigment Network", "Vascular Structures"), either from configuration or by querying an LLM.
2. **Archipelago Initialization:** Distinct populations are seeded for each concept.
3. **Parallel Evolution:**
    * **Select:** Best prompts are chosen within the island.
    * **Mutate:** LLM generates improved variants focusing *only* on the island's concept.
    * **Evaluate:** BioMedCLIP scores the new prompts against the few-shot dataset.
    * **Survive:** Worst prompts are culled.
4. **Convergence:** The top prompt from each island is harvested.
5. **Ensemble:** A weighted voting mechanism combines the concept-experts into a final robust classifier.

## 3. Usage Guide

### Prerequisites

* Python 3.10+
* Environment variables set in `.env` (e.g., `GROQ_API_KEY`, `OPENAI_API_KEY`).

### Running an Experiment

BioMedXPro uses a **Composable Configuration** system. You assemble an experiment by mixing and matching 4 components:

1. **Task (`-t`)**: The medical problem (Dataset path, Classes, Role).
2. **Evolution (`-e`)**: The hyperparameters (Generations, Population Size).
3. **LLM (`-l`)**: The intelligence provider (Groq, OpenAI, Llama).
4. **Execution (`-x`)**: The hardware settings (CPU/GPU workers, Batch size).

**Example Command:**

```bash
uv run main.py run \
  -t configs/tasks/melanoma_derm7pt.yaml \
  -e configs/evolution/standard.yaml \
  -l configs/llm/groq.yaml \
  -x configs/execution/cpu_local.yaml \
  --shots 16 \
  --metric f1_macro
```

### Adding New Components

#### 1. Adding a New Dataset

Create a new file in `configs/tasks/my_new_task.yaml`:

```yaml
task:
  task_name: "Pneumonia Detection"
  image_modality: "Chest X-Ray"
  # ... role and concepts ...

dataset:
  adapter: "my_adapter_name" # This key must match the python register!
  root: "/path/to/data"
  # ...
```

**Registering the Adapter Logic:**
You must implement the `IDatasetAdapter` interface in `src/biomedxpro/impl/adapters.py`.

```python
class MyAdapter:
    def load_samples(self, split: DataSplit) -> list[StandardSample]:
        # Parse your specific CSV/Folder structure here
        return [StandardSample(path, label), ...]

# Register it
def get_adapter(name: str, ...):
    if name == "my_adapter_name":
        return MyAdapter(...)
```

#### 2. Changing Evolution Parameters

Create a new config in `configs/evolution/deep_search.yaml`:

```yaml
evolution:
  generations: 50      # Run longer
  island_capacity: 100 # Larger islands
  num_parents: 20
  offspring_per_gen: 20
```

#### 3. Swapping LLMs

Create a new config in `configs/llm/gpt4.yaml`:

```yaml
llm:
  provider: "openai"
  model_name: "gpt-4-turbo"
  temperature: 0.7
```

## Directory Structure

```text
configs/
├── tasks/          # Problem definitions (Derm7pt, Camelyon17)
├── evolution/      # Search hyperparameters (Standard GA, Random Search)
├── llm/            # Provider settings (Groq, OpenAI)
└── execution/      # Hardware settings (Local CPU, Cluster GPU)

src/biomedxpro/
├── core/           # Domain Entities (Individual, Population)
├── engine/         # Orchestration & Builder logic
├── impl/           # Implementations (Adapters, Evaluator, LLM Client)
└── utils/          # Reporting, Metrics, Logging
```
