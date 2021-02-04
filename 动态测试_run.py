import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
from datetime import datetime

from bert_hrtp.models import create_model, InputFeatures
from bert_hrtp import tokenization, modeling
# chinese_L-12_H-768_A-12
# 标签
str_Labels_list = 'X [CLS] [SEP] O MASK'
for i in ['dis', 'sym', 'pro', 'equ', 'dru', 'ite', 'bod', 'dep', 'mic']:
    str_Labels_list+=' '+'B-' + i.upper()
    str_Labels_list+=' '+'I-' + i.upper()
args = {
    'bert_config_file': 'bert_hrtp/bert_base/bert_config.json',
    'bert_dir': 'bert_hrtp/bert_base',
    'init_checkpoint': 'bert_hrtp/bert_base/bert_model.ckpt',
    'model_dir': 'models/output_nosym',
    'vocab_file': 'bert_hrtp/bert_base/vocab.txt',
    'data_dir': 'data',
    'labels_list': str_Labels_list,
    'batch_size': 3,
    'cell': 'lstm',
    'clean': True,
    'clip': 0.5,
    'device_map': '0',
    'do_eval': True,
    'do_lower_case': True,
    'do_predict': False,
    'do_train': True,
    'dropout_rate': 0.5,
    'filter_adam_var': False,
    'learning_rate': 2e-05,
    'lstm_size': 128,
    'max_seq_length': 128,
    'ner': 'ner',
    'num_layers': 1,
    'num_train_epochs': 3.0,
    'save_checkpoints_steps': 500,
    'save_summary_steps': 500,
    'verbose': False,
    'warmup_proportion': 0.1

}

model_dir = args['model_dir']
bert_dir = args['bert_dir']

is_training=False
use_one_hot_embeddings=False
batch_size=1

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess=tf.Session(config=gpu_config)
model=None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None


print('checkpoint path:{}'.format(os.path.join(model_dir, "checkpoint")))
# 必须要存在checkpoint文件
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

# 加载label->id的词典
with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

# with codecs.open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
#     label_list = pickle.load(rf)
label_list=args['labels_list'].split(' ')
num_labels = len(label_list) + 1

graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, args['max_seq_length']], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, args['max_seq_length']], name="input_mask")

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    (total_loss, logits, trans, pred_ids) = create_model(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None,
        labels=None, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))


tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args['do_lower_case'])




def convert(line):
    feature = convert_single_example(0, line, label_list, args['max_seq_length'], tokenizer, 'p')
    input_ids = np.reshape([feature.input_ids],(batch_size, args['max_seq_length']))
    input_mask = np.reshape([feature.input_mask],(batch_size, args['max_seq_length']))
    segment_ids = np.reshape([feature.segment_ids],(batch_size, args['max_seq_length']))
    label_ids =np.reshape([feature.label_ids],(batch_size, args['max_seq_length']))
    return input_ids, input_mask, segment_ids, label_ids




def trymask(string):
    # ****************标签集********************
    print(id2label)
    duiying={}
    for i in id2label.items():
        duiying[i[1]]=i[0]
    # print(duiying)
    # print(duiying['MASK'])
    sentence = string
    # start = datetime.now()
    sentence = tokenizer.tokenize(sentence)
    # print(sentence,11)
    # print('your input is:{}'.format(sentence))
    input_ids, input_mask, segment_ids, label_ids = convert(sentence)

    feed_dict = {input_ids_p: input_ids,
                 input_mask_p: input_mask}
    # print(sentence,11)
    # print(feed_dict,22)
    # print(pred_ids,33)
    # run session get current feed_dict result
    pred_ids_result = sess.run([pred_ids], feed_dict)
    # print(pred_ids_result)
    pos=[]
    # print(pred_ids_result[0][0])
    for i in range(len(pred_ids_result[0][0])):
        if pred_ids_result[0][0][i]==duiying['[CLS]'] or pred_ids_result[0][0][i]==duiying['[SEP]']:
            pos.append(i)
    for i in range(1,len(pos)-1):
        pred_ids_result[0][0][pos[i]]=duiying['O']
    print(pred_ids_result[0][0])
    # print(pred_ids_result)
    pred_label_result = convert_id_to_label(pred_ids_result, id2label)
    # print(pred_label_result[0])
    # timeused='time used: {} sec'.format((datetime.now() - start).total_seconds())
    outputdata=[sentence,pred_label_result[0]]
    # print(len(sentence),len(pred_label_result[0]),8888888888888888888888)
    return outputdata

def convert_id_to_label(pred_ids_result, idx2label):
    # print("##1")
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    # print(pred_ids_result)
    for row in range(batch_size):
        curr_seq = []

        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result



def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    # print("##2")
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(model_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


class Pair(object):
    # print("##3")
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type


    def __str__(self) -> str:
        line = []
        line.append('entity:{}'.format(self.__word))
        line.append('start:{}'.format(self.__start))
        line.append('end:{}'.format(self.__end))
        line.append('merge:{}'.format(self.__merge))
        line.append('types:{}'.format(self.__types))
        return '\t'.join(line)



def fenpian(string,n):
    strr=[]
    labs=[]

    # string=list(string)
    string_first=0
    string_end=len(string)
    # 查看列表长度是否大于n-2
    # def jieguo(string):
    #     return [list(string),list(string)]
    # 大于则从n-3+string_first后往前找句号感叹号逗号问号
    # 找到了就把strr和labs和输出结果相加
    # 直到首位相等
    # n=10
    p=[]
    if len(string)<=n-2:
        g=trymask(string)
        print(g)
        strr = strr + g[0]
        labs = labs + g[1]
        # print(strr)
        # print(labs)
        # print(strr,labs,11111111111111)
        for iii in range(len(strr)):
            if strr[iii] in ['，','。','？','！','；','：']:
                # print(iii)
                labs[iii]='O'
        # print(strr,labs)
    else:
        # count=0
        while string_first!=string_end:
            # count+=1
            # print(count)
            MAX=n-3+string_first
            if MAX>string_end:
                MAX=string_end
            p=[]
            for i in range(string_first,MAX):
                if string[i] in ['，','。','？','！','；','：','、','.']:
                    p.append(i)

            if len(p)>=1:
                # print(string_first,p[-1]+1)
                qiestrr=string[string_first:p[-1]+1]
                print(qiestrr)
                gg = trymask(qiestrr)

                for i in range(len(gg[0])):
                    if gg[0][i] in ['，','。','？','！','；','：']:
                        gg[1][i]='O'
                strr = strr + gg[0]
                labs = labs + gg[1]
                string_first = p[-1]+1
            else:break
    return [strr,labs]

def maskrecon(anss,n):
    ans=anss
    string = ans[0]
    labels = ans[1]
    mask=0
    biaoji=[]
    for i in labels:
        if i=='MASK':
            mask=1
            break

    if mask:
        # 把每一个mask都识别一下然后送到biaoji里面
        first=0
        endd=len(labels)

        p=0
        q=0
        way=0
        for i in range(first,endd):
           if labels[i]=='MASK' and way==0:
               p=i
               way=1
           if way==1  and (labels[i]!='MASK' or i==endd-1):
               if i == endd - 1 and labels[i]!='O':
                   q += 1
               q=i
               way=0
               # print(string[p:q])
               if labels[q] == 'I-SYM' or labels[p - 1][2:] == 'SYM':
                   anss=fenpian(''.join(string[p:q]),n=n)
                   # anss = w1
                   st1=''.join(anss[0])
                   # j_count=st1.count('#')
                   # st1.replace('#','')
                   biaoji.append([st1,anss[1][0][2:],p,q])
               if i==endd-1 or labels[q][0]=='B' or labels[q]=='MASK':
                   for la in range(p,q):
                       labels[la] = 'I'+labels[p-1][1:]
               else:
                   for la in range(p,q):
                       labels[la] = 'I' + labels[q][1:]
                   labels[p] = 'B' + labels[q][1:]
    p = 0
    q = 0
    way = 0
    noww = ""
    # first = 0
    endd = len(labels)
    labels.append('O')
    for i in range(0, len(labels)):
        if way == 1 and labels[i] != 'I-' + noww:
            q = i
            way = 0
            st1 = ''.join(string[p:q])
            # j_count = st1.count('#')
            # st1.replace('#', '')

            anss = [st1, noww, p, q]

            biaoji.append(anss)
        if labels[i][0] == 'B':
            way = 1
            noww = labels[i][2:]
            p = i
            way = 1
            continue
        if way == 1 and labels[i] == 'I-' + noww:
            continue

    return biaoji
# print(maskrecon(['你去那了',["MASK",'MASK','I-dd','I-dd']],['你去',['B-i','I-i']]))

def transss(ggg):
    d1=[]
    d2=[]
    # 从前到后便利
    data = ggg
    for i in range(len(data[0])):
        data[0][i]=data[0][i].replace('#','')
        # print(len(data[0]),len(data[1]))
        if len(data[0][i])==1 or data[0][i][0:5]=='[UNK]':
            d1.append(data[0][i])
            d2.append(data[1][i])
        else:
            lab=''
            if len(data[1][i])>=2 and data[1][i][1]=='-':
                lab='I'+data[1][i][1:]
            else:
                lab=data[1][i]
            lll=len(data[0][i])
            for ii in range(lll):
                d1.append(data[0][i][ii])
            d2.append(data[1][i])
            for ii in range(lll-1):
                d2.append(lab)
    return([d1,d2])

def tiaozheng(ans):
    res=[]
    for i in ans:
        res.append([i[1],i[2],i[3]-1])
    return res

def getlinepostion(string):

    # with open(errofile, 'a+', encoding='utf-8') as fff:
    pf = 0
    position = []
    while True:
        pf = (string.find("|||", pf)) + 3
        if pf == -1 + 3: break
        position.append(pf - 3)
    # 得到“|”的位置
    # print(position)
    res = []
    for i in range(len(position) - 1):
        gg = string[position[i] + 3:position[i + 1]].split('    ')
        gg[0] = int(gg[0])
        gg[1] = int(gg[1])
        ggg=[gg[2].upper(),gg[0],gg[1]]
        res.append(ggg)
    # print(position)
    # print(string)
    string=string[:position[0]]
    return string,res


def maskrecon2(anss,n):
    ans=[]
    for i in anss:
        ans.append(i)
    string = ans[0]
    labels = ans[1]
    mask=0
    biaoji=[]
    for i in labels:
        if i=='MASK':
            mask=1
            break
    print(labels)
    if mask:
        # 把每一个mask都识别一下然后送到biaoji里面
        first=0
        endd=len(labels)

        p=0
        q=0
        way=0
        for i in range(first,endd):
           if labels[i]=='MASK' and way==0:
               p=i
               way=1
           if way==1  and (labels[i]!='MASK' or i==endd-1):
               if i == endd - 1 and labels[i]!='O':
                   q += 1
               q=i
               way=0
               # print(string[p:q])
               if labels[q] == 'I-SYM' or labels[p - 1][2:] == 'SYM':
                   anss=trymask(''.join(string[p:q]))
                   # anss = w1
                   st1=''.join(anss[0])
                   # j_count=st1.count('#')
                   # st1.replace('#','')
                   biaoji.append([st1,anss[1][0][2:],p,q])
               if i==endd-1 or labels[q][0]=='B' or labels[q]=='MASK':
                   for la in range(p,q):
                       labels[la] = 'I'+labels[p-1][1:]
               else:
                   for la in range(p,q):
                       labels[la] = 'I' + labels[q][1:]
                   labels[p] = 'B' + labels[q][1:]

    print(labels)
    # 对是I的进行处理
    # 第一种，前面是O后边是I
    # 如果前面没有B，则放弃该I
    # 如果有则标注
    def geet(labels,i):
        gg=labels[i]
        for p in range(i,len(labels)):
            if labels[p]!=gg:
                return p
    for i in range(len(string)-1):
        if string[i] in ['，','。','？','！'] and labels[i+1][0]=='I':
            labels[i+1] ='B-'+labels[i+1][2:]

    for i in range(len(labels)-1):
        if labels[i]=='O' and labels[i+1][0]=='I':
            p=i+1
            q=geet(labels,i+1)
            an=fenpian(''.join(string[i:q]),n)
            if an[1][0][0]!='B' and an[1][0]!='MASK':
                for u in range(p,q):
                    labels[u]='O'
            else:
                labels[i]='B'+labels[i+1][1:]
                # labels[ls_f]='B-'+ls_l
    print(labels)


                #****************************************************************可继续
            # 先看该位置能不能扩展

            # pass # 从该位置往前查找非O的，把该位置到I末尾进行整体识别，如果首位为B则标记该答案的最后一个I的标记，其他设为O，如果首位为O则返回上一个结果




    # 第二种，前边是I后边也是I
    p=0
    q=0
    way=0
    noww=""
    # first = 0
    labels.append('O')
    endd = len(labels)
    for i in range(0,len(labels)):
        if way == 1 and (labels[i] != 'I-' + noww ):
            q=i
            # if i==len(labels)-1:
            #     q+=1
            way=0
            st1 = ''.join(string[p:q])
            # j_count = st1.count('#')
            # st1.replace('#', '')

            anss = [st1,noww,p,q]

            biaoji.append(anss)
        if labels[i][0] == 'B':
            # way = 1
            noww=labels[i][2:]
            p = i
            way = 1
            continue
        if way==1 and labels[i]=='I-'+noww:
            continue

    return biaoji
# print(maskrecon(['你去那了',["MASK",'MASK','I-dd','I-dd']],['你去',['B-i','I-i']]))


def maskrecon1(anss,n):
    ans=[]
    for i in anss:
        ans.append(i)
    # ans=anss
    string = ans[0]
    labels = ans[1]
    mask=0
    biaoji=[]
    for i in labels:
        if i=='MASK':
            mask=1
            break

    if mask:
        # 把每一个mask都识别一下然后送到biaoji里面
        first=0
        endd=len(labels)

        p=0
        q=0
        way=0
        for i in range(first,endd):
           if labels[i]=='MASK' and way==0:
               p=i
               way=1
           if way==1  and (labels[i]!='MASK' or i==endd-1):
               if i == endd - 1 and labels[i]!='O':
                   q += 1
               q=i
               way=0
               # print(string[p:q])
               if labels[q] == 'I-SYM' or labels[p - 1][2:] == 'SYM':
                   anss=trymask(''.join(string[p:q]))
                   # anss = w1
                   st1=''.join(anss[0])
                   # j_count=st1.count('#')
                   # st1.replace('#','')
                   biaoji.append([st1,anss[1][0][2:],p,q])
               if i==endd-1 or labels[q][0]=='B' or labels[q]=='MASK':
                   for la in range(p,q):
                       labels[la] = 'I'+labels[p-1][1:]
               else:
                   for la in range(p,q):
                       labels[la] = 'I' + labels[q][1:]
                   labels[p] = 'B' + labels[q][1:]


    # 对是I的进行处理
    # 第一种，前面是O后边是I
    # 如果前面没有B，则放弃该I
    # 如果有则标注
    def geet(labels,i):
        gg=labels[i]
        for p in range(i,len(labels)):
            if labels[p]!=gg:
                return p
    for i in range(len(labels)-1):
        if labels[i]=='O' and labels[i+1][0]=='I':
            p=i+1
            q=geet(labels,i+1)
            an=fenpian(''.join(string[i:q]),n)
            if an[1][0][0]!='B' and an[1][0]=='MASK':
                for u in range(p,q):
                    labels[u]='O'
            else:
                # 如果前面的标号不为-1，且标记为O，则测试一下是否首字母为B，直到不为B则返回上面的结果
                ls_f = i
                ls_l = an[1][-1][2:]
                ls_e = i
                # 相对
                for u in range(1,len(an[1])):
                    if an[1][u][0]=='I':
                        ls_e=u+i+1
                        ls_l=an[1][u][2:]
                    else:break
                now_i=i-1
                # 绝对
                while now_i>=0 and labels[now_i]=='O':
                    re=fenpian(''.join(string[now_i:ls_e]),n)
                    if re[1][0][0]=='B' or an[1][0]=='MASK':
                        ls_f=now_i
                        # 相对
                        for u in range(1, len(re[1])):
                            if re[1][u][0] == 'I':
                                ls_e = u + now_i + 1
                                ls_l = re[1][u][2:]
                            else:
                                break
                        now_i-=1
                    else:
                        now_i-=1
                        break
                # 全设为O
                for kk in range(p,q):
                    labels[kk]='O'
                for kk in range(ls_f,ls_e):
                    labels[kk]='I-'+ls_l
                labels[ls_f]='B-'+ls_l


                #****************************************************************可继续
            # 先看该位置能不能扩展

            # pass # 从该位置往前查找非O的，把该位置到I末尾进行整体识别，如果首位为B则标记该答案的最后一个I的标记，其他设为O，如果首位为O则返回上一个结果




    # 第二种，前边是I后边也是I
    p = 0
    q = 0
    way = 0
    noww = ""
    # first = 0
    endd = len(labels)
    labels.append('O')
    for i in range(0, len(labels)):
        if way == 1 and labels[i] != 'I-' + noww:
            q = i
            way = 0
            st1 = ''.join(string[p:q])
            # j_count = st1.count('#')
            # st1.replace('#', '')

            anss = [st1, noww, p, q]

            biaoji.append(anss)
        if labels[i][0] == 'B':
            way = 1
            noww = labels[i][2:]
            p = i
            way = 1
            continue
        if way == 1 and labels[i] == 'I-' + noww:
            continue

    return biaoji
# print(maskrecon(['你去那了',["MASK",'MASK','I-dd','I-dd']],['你去',['B-i','I-i']]))

if __name__ == "__main__":
    from bert_hrtp.langconv import Converter
    acc_count = 0
    err_count = 0
    ig_count = 0
    N=60
    with graph.as_default():
        while True:
            # try:
                line=input("输入:")
                # line,laa=getlinepostion(line)
                # print(line)
                anss=fenpian(line,N)
                # print(anss)
                # a=[]
                # 三个的
                print([anss[0][m] + anss[1][m] for m in range(len(anss[0]))])
                # # 211 III/BII
                # for i in range(len(anss[1])-1,0-1+2,-1):
                #     if anss[1][i]==anss[1][i-1] and anss[1][i-2]!='O' and anss[1][i][0]=='I':
                #         anss[1][i-2]=anss[1][i-2][:1]+anss[1][i][1:]


                # 121 III IOI BOI

                # for i in range(len(anss[1]) - 1, 0 - 1 + 2, -1):
                #     if anss[1][i][0]=='I' and anss[1][i-2][2:]==anss[1][i][2:]:
                #         anss[1][i-1]=anss[1][i]

                for i in range(len(anss[1]) - 1, 0 - 1 + 2, -1):
                    xxx = ['', '', ''];
                    yyy = ['', '', '']
                    xxx[0]=anss[1][i-2][0];xxx[1]=anss[1][i-1][0];xxx[2]=anss[1][i][0];
                    yyy[0] = anss[0][i - 2][0];yyy[1] = anss[0][i - 1][0];yyy[2] = anss[0][i][0];
                    xxx = ''.join(xxx)
                    yyy = ''.join(yyy)
                    if xxx=='BII' or xxx=='III':
                        if anss[1][i]==anss[1][i-1]:
                            anss[1][i - 2]=anss[1][i - 2][:2]+anss[1][i][2:]

                for i in range(len(anss[1]) - 1, 0 - 1 + 2, -1):
                    xxx = ['', '', ''];
                    yyy = ['', '', '']
                    xxx[0] = anss[1][i - 2][0];
                    xxx[1] = anss[1][i - 1][0];
                    xxx[2] = anss[1][i][0];
                    yyy[0] = anss[0][i - 2][0];
                    yyy[1] = anss[0][i - 1][0];
                    yyy[2] = anss[0][i][0];
                    xxx=''.join(xxx)
                    yyy=''.join(yyy)
                    if xxx == 'IOI' or xxx == 'BOI':
                        if anss[0][i-1] not in ['，', '。', '！', '？', '；']:
                            if anss[1][i][2:] == anss[1][i - 2][2:]:
                                anss[1][i - 1] = 'I-' + anss[1][i][2:]

                for i in range(len(anss[1]) - 1, 0 - 1 + 1, -1):
                    xx = ['', ''];
                    yy = ['', '']
                    xx[0] = anss[1][i - 1][0];
                    xx[1] = anss[1][i][0];
                    yy[0] = anss[0][i - 1][0];
                    yy[1] = anss[0][i][0];
                    xx=''.join(xx)
                    if xx == 'BI':
                        anss[1][i-1]='B-'+anss[1][i][2:]



                print([anss[0][m]+anss[1][m] for m in range(len(anss[0]))])
                print(anss,111)
                anss = transss(anss)
                print([anss[0][m]+ anss[1][m] for m in range(len(anss[0]))])
                print(anss,22222)
                # print([m for m in anss[0]])
                # print([m[0] + '.' for m in anss[1]])

                anss0 = maskrecon(anss,n=N)
                print(anss0)
                print(anss,33333)
                anss1 = maskrecon1(anss,N)
                print(anss1)

                print(anss,444444)
                anss2 = maskrecon2(anss,N)
                print(anss2)
                # print(res)
                # anss = tiaozheng(anss)
                # print(anss)
                # print([m[0] + '.' for m in anss[1]])
                # print()
            # except:
            #     pass



