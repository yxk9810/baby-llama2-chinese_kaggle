## baby-llama2-chinese-fix
参考https://github.com/DLLXW/baby-llama2-chinese， 用于从头预训练+SFT一个小参数量的中文LLaMa2的仓库；24G单卡即可运行得到一个流畅中文问答的chat-llama2.

>20231013更新，fork代码

## 训练数据
- Wiki中文百科（25w词条）[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
- BaiduBaiKe（563w词条）
[百度网盘](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb)
 提取码: bwvb
- [Medical Dataset](https://huggingface.co/datasets/shibing624/medical/tree/main)

除此之外，为了让模型具备在某一个专有领域的能力，这里选用了“医疗问答”作为切入点，尝试收集了很多的医疗数据和上面的通用语料一起喂给模型。


## 中文分词器

采用ChatGLM2的分词器。

## 预训练语料预处理
```python
#脚本里面每一个函数对应一个语料库的预处理，搭建新加语料可以自行扩展。
python data_process.py
#运行结束后，会在./data目录下产生.bin文件
```
数据预处理采取GPT的通用做法，对语料进行提前分词，对一个样本做完分词后在末尾加上一个结束符号，与下一个样本区分开。然后将所有的训练语料拼接成一个数组（np.uint16）以.bin二进制格式存储到磁盘上。如果语料过大，避免内存溢出，可以选择mmap格式。

## SFT样本构建
中文SFT语料最近陆陆续续开源了很多（[bell](https://huggingface.co/datasets/BelleGroup/train_1M_CN)、[MOSS](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data)、[alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)等），但是坦白讲，质量都不高，大家可自行下载并需要进行清洗，清洗SFT数据是个耗时耗力的工作。
中文SFT语料网上最近很多，大家自行下载。参考dataset_sft.py即可！

基本逻辑如下：
- prompt和answer之间一定要有一个开始符隔开，然后answer后需要一个结束符。
- 计算loss的时候，对prompt部分的loss进行mask，只计算answer部分的loss即可。

## 预训练+SFT
因为用到了torch的分布式训练，我们需要在运行的时候设置环境变量。使用python -m torch.distributed.launch --use_env pretrain.py，或直接使用torchrun替代python命令。

```python
# 预训练——多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env pretrain.py
# 单卡
CUDA_VISIBLE_DEVICES=0 python pretrain.py

# SFT
 CUDA_VISIBLE_DEVICES=3 python sft.py

# eval
 CUDA_VISIBLE_DEVICES=3 python eval.py
```


根据自己算力的情况合理的调节以下参数，控制模型的计算量和参数量，这是第一版使用的参数
- max_seq_len = 256
- dim = 512
- n_layers = 8
- n_heads = 8

推理脚本可以参考eval.py。


