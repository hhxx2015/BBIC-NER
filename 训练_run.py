from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# 标签
str_Labels_list = 'X [CLS] [SEP] O MASK'
for i in ['dis', 'sym', 'pro', 'equ', 'dru', 'ite', 'bod', 'dep', 'mic']:
    str_Labels_list+=' '+'B-' + i.upper()
    str_Labels_list+=' '+'I-' + i.upper()


def Config_train():
    import os
    args={
        'bert_config_file': 'bert_hrtp/bert_base/bert_config.json',
        'init_checkpoint': 'bert_hrtp/bert_base/bert_model.ckpt',
        'vocab_file': 'bert_hrtp/bert_base/vocab.txt',
        'data_dir': 'data/data_nosym',
        'output_dir': 'output',
        'labels_list': str_Labels_list,
        'device_map': '0',
        'batch_size':3,
        'cell':'lstm',
        'clean':True,
        'clip':0.5,
        'do_eval':False,
        'do_lower_case':True,
        'do_predict':False,
        'do_train':True,
        'dropout_rate':0.5,
        'filter_adam_var':False,
        'learning_rate':2e-05,
        'lstm_size':128,
        'max_seq_length':128,
        'ner':'ner',
        'num_layers':1,
        'num_train_epochs':3.0,
        'save_checkpoints_steps':500,
        'save_summary_steps':500,
        'verbose':False,
        'warmup_proportion':0.1

    }

    for i in args.items():
        print(i[0],'=',i[1])


    os.environ['CUDA_VISIBLE_DEVICES'] = args['device_map']
    from bert_hrtp.bert_lstm_ner import train
    train(args=args)


if __name__ == '__main__':
    Config_train()
