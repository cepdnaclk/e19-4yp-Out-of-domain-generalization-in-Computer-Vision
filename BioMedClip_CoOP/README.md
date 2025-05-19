## Setup Instructions

Follow these steps to set up and run the project:

# Download Dataset

Download the dataset using the follwing link.

Update the path of the dataset in the main file in the following line
    'data_path': '{Data Path}/patches',



### 1. Clone the Repository

```bash
git clone https://github.com/cepdnaclk/e19-4yp-Out-of-domain-generalization-in-Computer-Vision
cd e19-4yp-Out-of-domain-generalization-in-Computer-Vision/BioMedClip_CoOP

conda env create -f environment.yaml
conda activate biomedclip

python main.py
