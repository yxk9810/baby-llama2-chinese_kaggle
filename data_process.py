import json
import glob
import numpy as np
from tqdm import tqdm
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd

def process_baidu():
    BATCH_SIZE = 1000000

    cnt=0
    batch_cnt=0
    token=0
    doc_ids=[]

    f1=open('./data/train_data/563w_baidubaike.json','r',encoding='utf-8')
    
    while True:
        line = f1.readline()
        if not line:
            break
        line=json.loads(line)
        text=''
        try:
            text+=line['title']+': '+line['summary']
        except:
            pass
        for per in line['sections']:
            text+=per['title']+': '+per['content']+'。'
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
        cnt+=1
        print(f'read 563w_baidubaike lines: {cnt}')
        if cnt%BATCH_SIZE==0:
            batch_cnt+=1
            arr = np.array(doc_ids,dtype=np.uint16)
            doc_ids=[]
            print('cnt:',cnt,'arr_shape:',arr.shape)
            with open('./data/baidubaike_563w_{}.bin'.format(batch_cnt),'wb') as f2:
                f2.write(arr.tobytes())
            del arr

    if not doc_ids:
        batch_cnt+=1
        arr = np.array(doc_ids,dtype=np.uint16)
        print('cnt:',cnt,'arr_shape:',arr.shape)
        with open('./data/baidubaike_563w_{}.bin'.format(batch_cnt),'wb') as f:
            f.write(arr.tobytes())
    

def sft_to_pretrain():
    doc_ids=[]

    '''
    df=pd.read_csv('./data/medical_qa_144w.csv')
    for _,q,a in tqdm(df.itertuples()):
        q_id = tokenizer.encode(q,add_special_tokens=False)
        a_id = tokenizer.encode(a,add_special_tokens=False)
        #
        print(q)
        print(a)
        print('-----')
        text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
        if len(text_id)>5:
            doc_ids+=text_id
    '''
    print('sft_to_pretrain: train_en_1')
    with open('./data/train_data/train_en_1.json','r',encoding='utf-8') as f:
        for row in tqdm(f):
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id

    print('sft_to_pretrain: test_en_1')
    with open('./data/train_data/test_en_1.json','r',encoding='utf-8') as f:
        for row in tqdm(f):
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id

    print('sft_to_pretrain: valid_en_1')
    with open('./data/train_data/valid_en_1.json','r',encoding='utf-8') as f:
        for row in tqdm(f):
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id

    print('sft_to_pretrain: train_zh_0')
    with open('./data/train_data/train_zh_0.json','r',encoding='utf-8') as f:
        for row in tqdm(f):
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    
    print('sft_to_pretrain: test_zh_0')
    with open('./data/train_data/test_zh_0.json','r',encoding='utf-8') as f:
        for row in tqdm(f):
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id

    print('sft_to_pretrain: valid_zh_0')
    with open('./data/train_data/valid_zh_0.json','r',encoding='utf-8') as f:
        for row in tqdm(f):
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id

    arr = np.array(doc_ids,dtype=np.uint16)
    print(arr.shape)
    with open('./data/train_data/medical_qa.bin','wb') as f:
        f.write(arr.tobytes())


#from zhconv import convert
def process_wiki_clean():
    with open('./data/train_data/wikipedia-cn-20230720-filtered.json','r',encoding='utf-8') as f:
        data=json.load(f)
    doc_ids=[]
    for line in tqdm(data):
        text=line['completion']
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/train_data/wiki.bin','wb') as f:
        f.write(arr.tobytes())

def process_medical(data_path,name):
    with open(data_path,'r',encoding='utf-8') as f:
        data=json.load(f)

    doc_ids=[]
    for line in tqdm(data):
        if not line:
            break
        line=json.loads(line)
        text=line['text']
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/train_data/medical_{}.bin'.format(name),'wb') as f:
        f.write(arr.tobytes()) 



def sft_process():
    with open('./data/sft_data/alpaca_gpt4_data_zh.json','r',encoding='utf-8') as f:
        data=json.load(f)

    q_lst=[]
    a_lst=[]
    for per in tqdm(data):
        q=per['instruction']
        i=per['input']
        a=per['output']
        q=q+i
        if len(q)<10 or len(a)<5:
            continue
        if len(q)>256 or len(a)>256:
            continue
        q_lst.append(q)
        a_lst.append(a)
    #
    # with open('../track1/train_valid.json','r') as f:
    #     data=json.load(f)
    # #
    # for l in data:
    #     q_lst.append(l['question'])
    #     a_lst.append(l['answer'])
    #

    f = open('./data/sft_data/Belle_open_source_1M.json','r',encoding='utf-8')
    #s
    while True:
        line = f.readline()
        if not line:
            break
        per=json.loads(line)
        q=per['instruction']
        i=per['input']
        a=per['output']
        q=q+i
        if len(q)<10 or len(a)<5:
            continue
        if len(q)>256 or len(a)>256:
            continue
        q_lst.append(q)
        a_lst.append(a)
    df=pd.DataFrame(columns=['prompt','answer'])
    df['prompt']=q_lst
    df['answer']=a_lst
    df.to_csv('data/sft_data/sft_data.csv',index=False)
    print(df)

    
if __name__=="__main__":
    tokenizer=ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
    print('process_baidu.')
    process_baidu()
    print('process_medical: medical_book_zh')
    process_medical('./data/train_data/medical_book_zh.json','book')
    print('process_medical: train_encyclopedia')
    process_medical('./data/train_data/train_encyclopedia.json','encyclopedia')
    print('sft_to_pretrain.')
    sft_to_pretrain()
    print('process_wiki_clean.')
    process_wiki_clean()

    data_path_list=[
        './data/train_data/baidubaike_563w_1.bin',
        './data/train_data/baidubaike_563w_2.bin',
        './data/train_data/baidubaike_563w_3.bin',
        './data/train_data/baidubaike_563w_4.bin',
        './data/train_data/baidubaike_563w_5.bin',
        './data/train_data/medical_book.bin',
        './data/train_data/medical_encyclopedia.bin',
        './data/train_data/medical_qa.bin',
        './data/train_data/wiki.bin',
    ]
    print('concat pretrain_data.')

    data_lst=[]
    for data_path in data_path_list:
        print(f'read data: {data_path}')
        with open(data_path,'rb') as f:
            data=np.fromfile(f,dtype=np.uint16)
            data_lst.append(data)
    arr = np.concatenate(data_lst)
    print(arr.shape)
    with open('./data/train_data/pretrain_data.bin','wb') as f:
        f.write(arr.tobytes())


    print('sft_process.')
    sft_process()