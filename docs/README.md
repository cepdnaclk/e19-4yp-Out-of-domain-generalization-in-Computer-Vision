---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: eYY-4yp-project-template
title: Enhancing Domain Generalization in Medical Imaging using Prompt Optimization
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# Enhancing Domain Generalization in Medical Imaging using Prompt Optimization

#### Team

- E/19/094, Eashwara M., [email](mailto:e19094@eng.pdn.ac.lk)  
- E/19/372, Silva A.K.M, [email](mailto:e19372@eng.pdn.ac.lk)  
- E/19/408, Ubayasiri S.J, [email](mailto:e19408@eng.pdn.ac.lk)

#### Supervisors

- Dr. Damayanthi Herath, [email](mailto:damayanthiherath@eng.pdn.ac.lk)  
- Dr. Ruwan Tennakoon, [email](mailto:ruwan.tennakoon@rmit.edu.au)

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

<!-- 
DELETE THIS SAMPLE before publishing to GitHub Pages !!!
This is a sample image, to show how to add images to your page. To learn more options, please refer [this](https://projects.ce.pdn.ac.lk/docs/faq/how-to-add-an-image/)
![Sample Image](./images/sample.png) 
-->


## Abstract

This research addresses domain generalization challenges in medical imaging using BiomedCLIP as the baseline model, a vision-language model optimized via advanced prompting strategies. We propose an automatic prompting method that improves interpretability and generalization through iterative feedback to large language models (LLMs), specifically adapting prompts for disease classification tasks from histopathological images.

---

## Related works

Vision-language models (VLMs) such as CLIP and BiomedCLIP have shown great promise in biomedical tasks, with models like BiomedCoOp and XCoOp introducing domain-specific prompt learning. However, many lack interpretability and rely on single static LLM outputs. Our method builds on these by integrating iterative feedback for prompt refinement, improving both robustness and transparency in clinical tasks.

---
## Methodology

We use BiomedCLIP as our base and apply a series of prompt optimization techniques to enhance out-of-domain generalization. The methodology includes:

### Preprocessing

- **Cleaning**: File integrity checks and metadata validation  
- **Normalization**: Standardizing pixel values  
- **Resizing**: Images resized to 224×224 pixels  
- **Standardization**: Label unification and demographic balancing  
- **Splitting**: Domain-generalization-based data split

### Prompt Optimization

An LLM-driven prompt generation framework starts with an initial set of prompts from Gemini. Using performance scores, prompts are iteratively refined to improve classification. The process includes:

- Prompt diversity strategies inspired by evolutionary algorithms  
- Scoring and feedback loops  
- Final prompts remain human-readable, improving interpretability

### CLIP Fine-Tuning Techniques

Three strategies are explored and compared:
- **Prompt Tuning**: CoOp, CoCoOp  
- **OOD Fine-Tuning**: CLIPood strategy  
- **Adapter Layers**: Task-specific feature learning

### Validation Strategy

Model performance is validated both in-domain and out-of-domain using accuracy, F1-score, AUC, and OOD metrics.

---

## Results and Analysis

## Conclusion

The integration of iterative, interpretable prompt generation using LLMs significantly improves the domain generalization capabilities of vision-language models in medical imaging. The approach offers a path forward for deploying robust and explainable AI tools in clinical settings.

---


## Publications
[//]: # "Note: Uncomment each once you uploaded the files to the repository"

<!-- 1. [Semester 7 report](./) -->
<!-- 2. [Semester 7 slides](./) -->
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->
<!-- 5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./). -->


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/repository-name)
- [Project Page](https://cepdnaclk.github.io/repository-name)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
