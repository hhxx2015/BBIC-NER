def chuliii(fromfile,tofile,n):
    with open(fromfile,'r',encoding='utf-8') as f:
        with open(tofile, 'a+', encoding='utf-8') as ff:
            for string in f:
# string="对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。|||3    9    bod|||19    24    dis|||"
                pf = 0
                position = []
                while True:
                    pf=(string.find("|||",pf))+3
                    if pf == -1+3: break
                    position.append(pf-3)
                # 得到“|”的位置
                print(position)

                # 标记队列
                biaoji= ['O'] * len(string[:position[0]])

                # 得到标签的位置
                for i in range(len(position)-1):
                    strr = string[position[i]+3:position[i+1]]
                    s_list = strr.split('    ')
                    a = int(s_list[0])
                    b = int(s_list[1])+1
                    label = s_list[2].upper()
                    print(a,b,label)
                    count=0
                    for p in range(a,b):
                        count += 1
                        if biaoji[p] == 'O':
                            if count ==1:
                                biaoji[p] = 'B-'+label
                            else:
                                biaoji[p] = 'I-'+label
                        else:
                            biaoji[p] = 'MASK'
                            gg=1

                # 句子和标签构造完毕了
                print(string[:position[0]], len(string[:position[0]]))
                ds=[]
                if len(string[:position[0]])>n-3:
                    for i in range(len(string[:position[0]])):
                        if string[:position[0]][i] in ['？','，','。','！','、']:
                            ds.append(i)
                ds.append(len(string[:position[0]])+10000000)
                print(ds)
                print(biaoji, len(biaoji))
                ans=[0]
                lst=0
                flg=1
                for i in range(len(ds)-1):
                    if ds[i]-lst<=n-3 and ds[i+1]-lst>n-3:
                        flg=0
                        ans.append(ds[i])
                        lst=ds[i]
                if flg:
                    ans.append(len(string[:position[0]])-1)
                print(ans,111)
                p=0
                for i in range(1,len(ans)):
                    q=ans[i]+1
                    strr=string[:position[0]]
                    print(strr[p:q],len(strr[p:q]))
                    print(biaoji[p:q],len(biaoji[p:q]))
                    for t in range(len(strr[p:q])):
                        ff.write(strr[p:q][t]+' '+biaoji[p:q][t]+'\n')
                    ff.write('\n')
                    p=q

                # print(ans)


def sym(fromfile, tofile, n):
    with open(fromfile, 'r', encoding='utf-8') as f:
        with open(tofile, 'a+', encoding='utf-8') as ff:
            for string in f:
                # string="对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。|||3    9    bod|||19    24    dis|||"
                pf = 0
                position = []
                while True:
                    pf = (string.find("|||", pf)) + 3
                    if pf == -1 + 3: break
                    position.append(pf - 3)
                # 得到“|”的位置
                print(position)

                # 标记队列
                biaoji = ['O'] * len(string[:position[0]])

                # 得到标签的位置
                for i in range(len(position) - 1):
                    strr = string[position[i] + 3:position[i + 1]]
                    s_list = strr.split('    ')
                    a = int(s_list[0])
                    b = int(s_list[1]) + 1
                    label = s_list[2].upper()
                    if label=='SYM':
                        print(a, b, label)
                        count = 0
                        for p in range(a, b):
                            count += 1
                            if count == 1:
                                biaoji[p] = 'B-' + label
                            else:
                                biaoji[p] = 'I-' + label


                # 句子和标签构造完毕了
                print(string[:position[0]], len(string[:position[0]]))
                ds = []
                if len(string[:position[0]]) > n - 3:
                    for i in range(len(string[:position[0]])):
                        if string[:position[0]][i] in ['？', '，', '。', '！', '、','；','：','.']:
                            ds.append(i)
                ds.append(len(string[:position[0]]) + 10000000)
                print(ds)
                print(biaoji, len(biaoji))
                ans = [0]
                lst = 0
                flg = 1
                for i in range(len(ds) - 1):
                    if ds[i] - lst <= n - 3 and ds[i + 1] - lst > n - 3:
                        flg = 0
                        ans.append(ds[i])
                        lst = ds[i]
                if flg:
                    ans.append(len(string[:position[0]]) - 1)
                print(ans, 111)
                p = 0
                for i in range(1, len(ans)):
                    q = ans[i] + 1
                    strr = string[:position[0]]
                    print(strr[p:q], len(strr[p:q]))
                    print(biaoji[p:q], len(biaoji[p:q]))
                    for t in range(len(strr[p:q])):
                        ff.write(strr[p:q][t] + ' ' + biaoji[p:q][t] + '\n')
                    ff.write('\n')
                    p = q

                # print(ans)


def budai_sym(fromfile, tofile, n):
    with open(fromfile, 'r', encoding='utf-8') as f:
        with open(tofile, 'a+', encoding='utf-8') as ff:
            for string in f:

                # string="对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。|||3    9    bod|||19    24    dis|||"
                pf = 0
                position = []
                while True:
                    pf = (string.find("|||", pf)) + 3
                    if pf == -1 + 3: break
                    position.append(pf - 3)
                # 得到“|”的位置
                print(position)

                # 标记队列
                biaoji = ['O'] * len(string[:position[0]])

                # 得到标签的位置
                for i in range(len(position) - 1):
                    strr = string[position[i] + 3:position[i + 1]]
                    s_list = strr.split('    ')
                    a = int(s_list[0])
                    b = int(s_list[1]) + 1
                    label = s_list[2].upper()
                    if label!='SYM':
                        print(a, b, label)
                        count = 0
                        for p in range(a, b):
                            count += 1
                            if count == 1:
                                biaoji[p] = 'B-' + label
                            else:
                                biaoji[p] = 'I-' + label


                # 句子和标签构造完毕了
                print(string[:position[0]], len(string[:position[0]]))
                ds = []
                if len(string[:position[0]]) > n - 3:
                    for i in range(len(string[:position[0]])):
                        if string[:position[0]][i] in ['？', '，', '。', '！', '、','；','：','.']:
                            ds.append(i)
                ds.append(len(string[:position[0]]) + 10000000)
                print(ds)
                print(biaoji, len(biaoji))
                ans = [0]
                lst = 0
                flg = 1
                for i in range(len(ds) - 1):
                    if ds[i] - lst <= n - 3 and ds[i + 1] - lst > n - 3:
                        flg = 0
                        ans.append(ds[i])
                        lst = ds[i]
                if flg:
                    ans.append(len(string[:position[0]]) - 1)
                print(ans, 111)
                p = 0
                for i in range(1, len(ans)):
                    q = ans[i] + 1
                    strr = string[:position[0]]
                    print(strr[p:q], len(strr[p:q]))
                    print(biaoji[p:q], len(biaoji[p:q]))
                    for t in range(len(strr[p:q])):
                        ff.write(strr[p:q][t] + ' ' + biaoji[p:q][t] + '\n')
                    ff.write('\n')
                    p = q

                # print(ans)

chuliii('train_data.txt','data_all/train.txt',128-3)
sym('train_data.txt','data_sym/train.txt',128-3)
# budai_sym('train_data.txt','trainnosym.txt',128-3)

