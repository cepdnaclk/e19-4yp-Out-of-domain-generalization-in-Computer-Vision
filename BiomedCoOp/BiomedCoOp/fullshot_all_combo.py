import subprocess

methods = ["coop", "biomedcoop", "cocoop"]
datasets = ["nih_chest", "camelyon17", "wbc_att"]

shots = 0 

for method in methods:
    for dataset in datasets:
        script_path = f"./scripts/{method}/few_shot_{dataset}.sh"
        log_file = f"{dataset}_full_shot_{method}.log"

        # nohup command
        cmd = f"nohup {script_path} {shots} > {log_file} 2>&1"
        print(f"Running: {cmd}")

        # Run the command and wait for it to complete
        result = subprocess.run(cmd, shell=True)
        
        # Optional: Check if the command succeeded
        if result.returncode == 0:
            print(f"Completed successfully: {dataset} with {method}")
        else:
            print(f"Failed: {dataset} with {method} (return code: {result.returncode})")

print("All 9 jobs completed sequentially.")