"""
Sample from the trained model with PyTorch
"""
import os
import json
from contextlib import nullcontext
import torch
from model import ModelArgs, Transformer
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import numpy as np

def compute_bleu(labels, preds, weights=None):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    weights = weights or (0.25, 0.25, 0.25, 0.25)
    return np.mean([sentence_bleu(references=[label],
                                  hypothesis=pred,
                                  smoothing_function=SmoothingFunction().method1,
                                  weights=weights) for label, pred in zip(labels, preds)])
# -----------------------------------------------------------------------------
from setting import parser_args
opt = parser_args()
# start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
#exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------


torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in opt.device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[opt.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast()

# init from a model saved in a specific directory
ckpt_path = f'./out/{opt.save_path}_sft_bell/epoch_{1}.pth'
state_dict = torch.load(ckpt_path, map_location=opt.device)

model_args = dict(
        dim=opt.dim,
        n_layers=opt.n_layers,
        n_heads=opt.n_heads,
        n_kv_heads=opt.n_heads,
        vocab_size=opt.vocab_size,#64793,
        multiple_of=opt.multiple_of,
        max_seq_len=opt.max_seq_len,
        dropout=opt.dropout,
    )  # 

gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(opt.device)
if opt.compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load the tokenizer
tokenizer=ChatGLMTokenizer(vocab_file=opt.vocab_file)
#
answer_list=[]
predict_lst=[]
with open(opt.eval_data_path,'r',encoding='utf-8') as f:
    from tqdm import tqdm
    line_num=0
    for row in f:
        line=json.loads(row)
        if line_num>100:
            break

        line_num+=1
        # run generation
        prompt=line['instruction']#+line['input']
        x=tokenizer.encode(prompt,add_special_tokens=False)+[tokenizer.special_tokens['<eos>']]
        x = (torch.tensor(x, dtype=torch.long, device=opt.device)[None, ...])
        answer=line['output']
        answer_list.append(answer)
        with torch.no_grad():
            with ctx:
                y = model.generate(x, 2, opt.max_new_tokens, temperature=opt.temperature, top_k=opt.top_k)
                #
                predict=tokenizer.decode(y[0].tolist())
                predict=predict.replace(prompt,'')
                predict_lst.append(predict)
                print('\n---------------')
                print('[prompt]:',prompt)
                print('[answer]:',answer)
                print('[predict]:',predict)
#
import jieba
target_lst=[jieba.lcut(result.lower()) for result in answer_list]
preds_lst=[jieba.lcut(result.lower()) for result in predict_lst]
scores = compute_bleu(preds_lst, target_lst)
print(scores)