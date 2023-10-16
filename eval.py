"""
Sample from the trained model with PyTorch
"""
import os
import json
from contextlib import nullcontext
import torch
import pandas as pd
from model import ModelArgs, Transformer
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import numpy as np
from setting import parser_args,parser_config


def compute_bleu(labels, preds, weights=None):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    weights = weights or (0.25, 0.25, 0.25, 0.25)
    return np.mean([sentence_bleu(references=[label],
                                  hypothesis=pred,
                                  smoothing_function=SmoothingFunction().method1,
                                  weights=weights) for label, pred in zip(labels, preds)])

def eval_medical(model,tokenizer,ctx,logger):
    answer_list=[]
    predict_lst=[]
    print(f'*************medical eval*************')

    medical_path_list = [
        'data/valid_zh_0.json',
        'data/valid_en_1.json',
    ]

    scores = dict()
    for eval_data_path in medical_path_list:
        with open(eval_data_path,'r',encoding='utf-8') as f:
            from tqdm import tqdm
            line_num=0
            for row in tqdm(f):
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
                        # print('\n---------------')
                        # print('[prompt]:',prompt)
                        # print('[answer]:',answer)
                        # print('[predict]:',predict)
        #
        import jieba
        target_lst=[jieba.lcut(result.lower()) for result in answer_list]
        preds_lst=[jieba.lcut(result.lower()) for result in predict_lst]
        score = compute_bleu(preds_lst, target_lst)
        print(f'{eval_data_path}: eval_scores: {score}')
        scores[eval_data_path] = score

    weighted_acc = sum(scores.values())/len(scores)
    logger.info(f'model: {model_path_}. medical_eval_scores: {weighted_acc}')


def eval_ceval(model, tokenizer, opt, logger):
    print(f'*************CEval*************')
    from eval.ceval import CEval
    ceval = CEval(model, tokenizer, opt)
    average_acc=ceval.run('data/ceval-exam',opt.shot)
    logger.info(f'model: {opt.save_path}. Ceval_eval_scores: {average_acc}')


def eval_mmlu(model, tokenizer, opt, logger):
    print(f'*************MMLU*************')
    from eval.mmlu import mmlu_eval_func
    weighted_acc=mmlu_eval_func('data/mmlu', opt, model, tokenizer)
    logger.info(f'model: {opt.save_path}. MMLU_eval_scores: {weighted_acc}')


def eval_longbench(model, tokenizer, opt, logger):
    print(f'*************LongBench*************')
    from eval.longbench import longbench_eval_func
    weighted_acc=longbench_eval_func('data/longBench', opt, model, tokenizer)
    logger.info(f'model: {opt.save_path}. LongBench_eval_scores: {weighted_acc}')


# def eval_LongEval(model, tokenizer, opt, logger):
#     print(f'*************LongEval*************')
#     from eval.longeval import longeval_eval_func
#     weighted_acc=longeval_eval_func('data/longbench', opt, model, tokenizer)
#     logger.info(f'model: {opt.save_path}. LongEval_eval_scores: {weighted_acc}')


# def eval_GSM8K(model, tokenizer, opt, logger):
#     print(f'*************GSM8K*************')
#     gsm8k = GSM8K(model, tokenizer, opt.output_dir)
#     weighted_acc=gsm8k.run(opt.shot, opt.split)
#     logger.info(f'model: {opt.save_path}. gsm8k_eval_scores: {weighted_acc}')


def eval(model_path_,opt,logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in opt.device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[opt.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast()

    # init from a model saved in a specific directory
    state_dict = torch.load(model_path_, map_location=opt.device)

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

    eval_medical(model, tokenizer, ctx, logger)
    eval_ceval(model, tokenizer, opt, logger)
    eval_mmlu(model, tokenizer, opt, logger)
    eval_longbench(model, tokenizer, opt, logger)
    # eval_LongEval(model, tokenizer, opt, logger)
    # eval_GSM8K(model, tokenizer, opt, logger)



# I/O
if __name__=="__main__":
    # -----------------------------------------------------------------------------
    opt = parser_args()
    # start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    #dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    #exec(open('configurator.py').read()) # overrides from command line or config file
    # -----------------------------------------------------------------------------
    from share import get_logger
    logger = get_logger(os.path.join(opt.out_dir,'eval_all.log'))

    model_path_list = os.listdir(opt.out_dir)
    for model_path in model_path_list:
        model_path_ = os.path.join(opt.out_dir, model_path)

        model_list = os.listdir(model_path_)
        for model_ in model_list:
            if model_.endswith('pth'):
                opt.config = os.path.join(model_path_, 'config.yaml')
                opt,_ = parser_config(opt)
                opt.save_path = os.path.join(model_path_, model_)
                print(f'*************eval model: {model_path_}*************')
                eval(opt.save_path,opt,logger)