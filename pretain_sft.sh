if [ false ]
then
    CUDA_VISIBLE_DEVICES=3,4,5,6 nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env pretrain.py >out/pretrain_log
    CUDA_VISIBLE_DEVICES=3 nohup python sft.py >out/sft_log
    CUDA_VISIBLE_DEVICES=3 nohup python eval.py >out/eval_log
else
    CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --nproc_per_node=4 --use_env pretrain.py
    CUDA_VISIBLE_DEVICES=3 python sft.py
    CUDA_VISIBLE_DEVICES=3 python eval.py
fi


