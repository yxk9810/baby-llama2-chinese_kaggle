python data_prepare.py

if [ false ]
then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=8 --use_env pretrain.py >out/pretrain_1_log
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=8 --use_env fine_tuning.py >out/fine_tuning_log
    CUDA_VISIBLE_DEVICES=0 nohup python eval.py >out/eval_log
else
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env pretrain.py
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env fine_tuning.py
    CUDA_VISIBLE_DEVICES=0 python eval.py
fi
