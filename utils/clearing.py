import torch
import gc

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    with torch.no_grad():
        for _ in range(3):
          torch.cuda.empty_cache()
          torch.cuda.ipc_collect()


def replace_in_memory(device, to_gpu = [], to_cpu = [], ):
    if len(to_cpu):
        for obj in to_cpu:
           if obj.device != 'cpu': obj.to('cpu')
    flush_memory()

    if len(to_gpu):
        for obj in to_gpu:
           if obj.device != 'gpu': obj.to(device)
    flush_memory()