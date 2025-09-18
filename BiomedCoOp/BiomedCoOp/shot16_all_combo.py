import subprocess

methods = ["coop", "biomedcoop", "cocoop"]
datasets = ["nih_chest", "camelyon17", "derm7pt", "wbc_att"]


shots = 16  

for method in methods:
    for dataset in datasets:
        script_path = f"./scripts/{method}/few_shot_{dataset}.sh"
        log_file = f"{dataset}_{shots}_shot_{method}.log"

        # nohup command
        cmd = f"nohup {script_path} {shots} > {log_file} 2>&1"
        print(f"Running: {cmd}")

        # Run the command
        subprocess.Popen(cmd, shell=True)

print("All 12 jobs launched.")
