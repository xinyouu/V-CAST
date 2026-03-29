# Use an environment variable instead of committing a Hugging Face token.
import os

from huggingface_hub import snapshot_download

HF_TOKEN = os.getenv("HF_TOKEN")
# 下载 LongVideoBench
# snapshot_download(repo_id='longvideobench/LongVideoBench',
# # snapshot_download(repo_id='exiawsh/videobench',
#     repo_type='dataset',
#     local_dir='/data/share/250010029/LongVideoBench',
#     resume_download=True,  # 显式开启断点续传
#     endpoint='https://hf-mirror.com',
#     token=HF_TOKEN)# 下载 MVLU
# snapshot_download(repo_id='sy1998/MLVU_dev',
#     repo_type='dataset',
#     local_dir='/data/share/250010029/sy1998_MVLU_dev',
#     resume_download=True,
#     endpoint='https://hf-mirror.com',
#     token=HF_TOKEN)# 下载 MVBench
# snapshot_download(repo_id='OpenGVLab/MVBench',
# snapshot_download(repo_id='VLM2Vec/MVBench',
#     repo_type='dataset',
#     local_dir='/data/share/250010029/MVBench_new',
#     resume_download=True,
#     endpoint='https://hf-mirror.com',
#     token=HF_TOKEN)

snapshot_download(repo_id='lmms-lab/worldsense',
    repo_type='dataset',
    local_dir='/data/share/worldsense',
    resume_download=True,
    endpoint='https://hf-mirror.com',
    token=HF_TOKEN)

# snapshot_download(repo_id='Qwen/Qwen3-VL-30B-A3B-Instruct',
#     repo_type='model',
#     local_dir='/data/share/250010029',
#     resume_download=True,
#     endpoint='https://hf-mirror.com',
#     token=HF_TOKEN)
