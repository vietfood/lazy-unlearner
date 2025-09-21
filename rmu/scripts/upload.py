from huggingface_hub import HfApi

api = HfApi()

# api.upload_folder(
#     folder_path="/home/ubuntu/thesis/code/utils/eval/gemma-2b/__home__ubuntu__thesis__code__rmu_results__rmu_ablated",
#     path_in_repo="eval/unlearn_ablated", # Upload to a specific folder
#     repo_id="lenguyen1807/thesis",
# )

# api.upload_folder(
#     folder_path="/home/ubuntu/thesis/code/notebooks/eval/gemma-2b/__home__ubuntu__thesis__code__rmu_results__layer9_alpha300.0_steer750.0",
#     path_in_repo="eval/unlearn_it",
#     repo_id="lenguyen1807/thesis",
# )

# api.upload_large_folder(
#     repo_id="lenguyen1807/gemma-2-2b-RMU-ablated",
#     repo_type="model",
#     folder_path="/home/ubuntu/thesis/sae/results/models/shared",
# )

api.upload_folder(
    repo_id="lenguyen1807/thesis",
    folder_path="/home/ubuntu/thesis/sae/results/ablated",
    path_in_repo="ablated/",
)

# api.upload_file(
#     path_or_fileobj="/home/ubuntu/thesis/code/baseline_cache.pt",
#     path_in_repo="ablted/baseline.pt",
#     repo_id="lenguyen1807/thesis",
# )

# api.upload_file(
#     path_or_fileobj="/home/ubuntu/thesis/code/rmu_cache.pt",
#     path_in_repo="refusal_cache/rmu.pt",
#     repo_id="lenguyen1807/thesis",
# )

# api.upload_large_folder(
#     repo_id="lenguyen1807/gemma-2b-it-RMU-Ablated",
#     repo_type="model",
#     folder_path="/home/ubuntu/thesis/code/rmu_results/rmu_ablated",
# )
