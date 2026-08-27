import torch
import os
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor
from safetensors.torch import save_file

api = HfApi()
executor = ThreadPoolExecutor(max_workers=1)

def _upload_and_clean(file_path, repo_id):
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_id,
            repo_type="model"
        )
        #os.remove(file_path)
    except Exception as e:
        print(e)

def save_checkpoint(raw_model, optimizer, scheduler, step, repo_id):
    file_name = f"checkpoint_{step}.pt"
    
    checkpoint = {
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step
    }
    
    torch.save(checkpoint, file_name)
    executor.submit(_upload_and_clean, file_name, repo_id)

def save_final_model(raw_model, repo_id):
    file_name = "model.safetensors"
    save_file(raw_model.state_dict(), file_name)
    executor.submit(_upload_and_clean, file_name, repo_id)

def wait_for_uploads():
    executor.shutdown(wait=True)