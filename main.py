import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image
import sys
project_root = ''
if project_root not in sys.path:
    sys.path.append(project_root)

import torch, torch.nn as nn, torch.nn.functional as F, argparse, transformers, sys
import argparse
from transformers import set_seed
def prepare_model(args,gpt_model=None):
    return gpt_model

def token_prefill(model, seq, cond=None, max_seq_length=257, cfg_scale=4.0):
    t = seq
    cond_combined = torch.tensor(cond)
    model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    model.to(t.device)
    logs = t
    logits = model(idx=logs,cond_idx=cond_combined,input_pos=torch.arange(0,257,device=t.device))[0]
    p = logits[0].argmax(dim=-1)
    return p[None,0:256]

def get_input(args, gpt_model, class_labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = args.image_size // args.downsample_size
    c_indices = torch.tensor(class_labels, device=device)
    # index_sample = generate()
    index_sample = None
    return index_sample

from tqdm import tqdm
import args_utils
from gptq_utils import gptq_fwrd

def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = []
    # for i in tqdm(range(len(class_labels))):
    #     class_label = class_labels[i]
    #     index = get_input(args, gpt_model, [class_label])
    #     index = index[:,0:256]
    #     class_label = torch.tensor([class_label])
    #     input_pos=torch.arange(0,257)
    #     #logs = torch.cat([index,index])
    #     #cond_null = (torch.ones_like(class_label) * args.num_classes)
    #     #cond_combined = torch.cat([class_label, cond_null])
    #     logs = index
    #     cond_combined = class_label
    #     train_loader.append((logs,cond_combined,input_pos))

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gptq_args = args_utils.parser_gen()
    gpt_model = prepare_model(args)
    gptq_fwrd(gpt_model,train_loader,dev,gptq_args)


    gpt_model = prepare_model(args)
    model_path = './updated_model_weights.pt'
    gpt_model.load_state_dict(torch.load(model_path), strict=False) #ignore KV cache

    gpt_model.eval()
    gpt_model.to(device)
    set_seed(args.seed) 
    # save_image()
    print('ok')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)

