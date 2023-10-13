import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
import time
import numpy as np
import torch
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import PretrainDataset
from share import get_lr, get_logger, init_model,init_ddp

#To run with DDP on 4 gpus on 1 node, example:
# torchrun --standalone --nproc_per_node=4 pretrain.py OR python -m torch.distributed.launch --nproc_per_node=4 pretrain.py
def pretrain_epoch(epoch, opt):
    start_time=time.time()
    for step, (X, Y) in enumerate(train_loader):
        X=X.to(opt.device)
        Y=Y.to(opt.device)
        lr = get_lr(epoch*iter_per_epoch+step, opt) if opt.decay_lr else opt.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # and using the GradScaler if data type is float16
        #for micro_step in range(gradient_accumulation_steps):
        
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = 0 == opt.gradient_accumulation_steps - 1
        with ctx:
            logits = model(X, Y)
            loss = raw_model.last_loss
            #loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        #
        # clip the gradient
        if opt.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        #打印日志
        if step % opt.log_interval == 0:
            spend_time=time.time()-start_time
            logger.info(
                    'Epoch:[{}/{}] ({}/{}) loss:{:.3f} lr:{:.7f}  epoch_Time: {} min.'.format(
                        epoch,
                        opt.max_epoch, 
                        step, 
                        iter_per_epoch,
                        loss.item(), 
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))

@torch.no_grad()
def valid_epoch(epoch):
    global best_val_loss
    losses = []
    model.eval()
    for _, (X, Y) in enumerate(val_loader):
        X=X.to(device)
        Y=Y.to(device)
        with ctx:
            logits, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    val_loss=np.mean(losses)
    #
    logger.info('valid loss = {:.4f}'.format(val_loss))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        logger.info('best val_loss: {} best_epoch: {} '.format(best_val_loss,epoch))
        torch.save(raw_model.state_dict(),'{}/best.pth'.format(save_dir))
    #
    return val_loss

# I/O
if __name__=="__main__":
    
    from setting import parser_args
    opt = parser_args()

    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    # exec(open("configurator.py").read())  # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    save_dir =os.path.join(opt.out_dir , f'{opt.save_path}_pretrain')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    logger = get_logger(os.path.join(save_dir,'log.log'))
    # various inits, derived attributes, I/O setup
   # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?

    master_process,ddp_local_rank,ctx=init_ddp(ddp, opt)

    if master_process:
        os.makedirs(opt.out_dir, exist_ok=True)
    
    #init model
    model=init_model(opt)
    model.to(opt.device)
    print(f"====================models====================\n",model)
    print(f"====================models====================")

    # optimizer
    optimizer = model.configure_optimizers(opt.weight_decay, opt.learning_rate, (opt.beta1, opt.beta2), opt.device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(opt.dtype == 'float16'))
    #
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.max_epoch, T_mult=1, eta_min=1e-6, last_epoch=-1)
    
    # compile the model
    if opt.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    # wrap model into DDP container
    if ddp:
        # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
        # construction time since NCCL does not support `ComplexFloat`
        prefix = "_orig_mod." if opt.compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])
        #
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    

    print(f"====================prepear dataset====================")

    best_val_loss = 1e9
    #
    #-----init dataloader------
    data_path_list=opt.train_data_path
    train_ds = PretrainDataset(data_path_list, max_length=opt.max_seq_len,memmap=True)
    # val_ds = PretrainDataset(data_path_list, max_seq_len=256)
    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0 if os.name == 'nt' else 4,
        sampler=train_sampler
    )
    # val_loader = torch.utils.data.DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     pin_memory=False,
    #     drop_last=False,
    #     shuffle=False,        
    #     num_workers=0,
    # )

    print(f"====================pretrain_epoch====================")

    iter_per_epoch=len(train_loader)
    warmup_epoch=1
    
    # training loop
    for epoch in range(opt.max_epoch):
        pretrain_epoch(epoch,opt)
        #val_loss=valid_epoch(epoch)
        if torch.distributed.get_rank() == 0:  #一般用0，当然，可以选任意的rank保存。
            #
            torch.save(raw_model.state_dict(),'{}/epoch_{}.pth'.format(save_dir,epoch))
    if ddp:
        destroy_process_group()
