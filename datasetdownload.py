from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dasgringuen/assettoCorsaGym",
    repo_type="dataset",
    local_dir="AssettoCorsaGymDataSet",  # same folder as before
    allow_patterns=["data_sets/ks_red_bull_ring-layout_gp/bmw_z4_gt3/*"],
    resume_download=True,  # <-- tells it to continue, not overwrite
    max_workers=2,
)
