import math

from jieba import *
import networkx as nx
import numpy as np
import codecs

def split_sentences(full_text):
    """
    分离句子并计算句子的位置价值
    :param full_text:
    :return:seats
    """
    # 分离句子（除标题外）
    seats = re.split(u'[。?!？！]', full_text)
    seats = [sent for sent in seats if len(sent) > 0]
    # 分离标题,将标题放至列表末端保存
    title = re.split(u'[\n]', seats[0])
    seats[0] = title[1]
    seats.append(title[0])

    # 初始化位置计分数组
    length = len(seats)
    locatscore = [0 for x in range(0, length)]
    # 标题句子价值加3
    locatscore[length - 1] += 3

    # 段首句子价值加1
    t = 0
    for s in seats:
        if re.search(u'[\n]', s) != None:
            locatscore[t] += 1
        t += 1
    l_mark = locatscore  # 复制一个标记数组，记录回车符的位置

    # 首段句子价值+2
    t = 0
    while l_mark[t] < 1:
        locatscore[t] += 2
        t += 1

    # 尾段句子价值+1.5
    t = length - 2
    while l_mark[t] < 1:
        locatscore[t] += 1.5
        t -= 1
    locatscore[t] += 1.5

    for i in range(0, length - 1 + 1):
        seats[i] = seats[i].replace('\n', '').replace('\r', '')
    seats = [sent for sent in seats if len(sent) > 0]
    return (seats, locatscore)

def cal_sim(wordlist1, wordlist2):
    co_occur_sum = 0
    wordset1 = list(set(wordlist1))
    wordset2 = list(set(wordlist2))
    for word in wordset1:
        if word in wordset2:
            co_occur_sum += 1.0
    if co_occur_sum < 1e-10:
        return 0.0
    denominator = math.log(len(wordset1)) + math.log(len(wordset2))
    if abs(denominator) < 1e-10:
        return 0.0
    return co_occur_sum / denominator


def text_rank(sentences, num=10, pagerank_config={'alpha': 0.85, }):
    """
    对输入的句子进行重要度排序
    :param sentences: 句子的list
    :param num: 希望输出的句子数
    :param pagerank_config: pagerank相关设置，默认设置阻尼系数为0.85
    :return:
    """
    sorted_sentences = []
    sentences_num = len(sentences)
    wordlist = []  # 存储wordlist避免重复分词，其中wordlist的顺序与sentences对应
    for sent in sentences:
        tmp = []
        cur_res = cut(sent)
        for i in cur_res:
            tmp.append(i)
        wordlist.append(tmp)
    graph = np.zeros((sentences_num, sentences_num))
    for x in xrange(sentences_num):
        for y in xrange(x, sentences_num):
            similarity = cal_sim(wordlist[x], wordlist[y])
            graph[x, y] = similarity
            graph[y, x] = similarity
    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)  # this is a dict
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for index, score in sorted_scores:
        item = {"sent": sentences[index], 'score': score, 'index': index}
        sorted_sentences.append(item)
    return sorted_sentences
    #将结果全部输出


def extract_abstracts(full_text, sent_num=10, ratio = 0.5, t = 5):
    """
    摘要提取的入口函数，并根据textrank结果进行摘要组织
    :param full_text:
    :param sent_num:
    :param ratio
    :return:
    """
    (sents, locatscore) = split_sentences(full_text)
    trank_res = text_rank(sents, num=sent_num)
    a = ratio #Tankrank得分比例
    b = 0.01*t *(1 - a) #位置评价得分比例
    for i in range (0, len(trank_res)-1):
        trank_res[i]['score'] = a * trank_res[i]['score'] + b * locatscore[trank_res[i]['index']]
    sorted_res = sorted(trank_res, key=lambda x: x['score'], reverse=True)
    return sorted_res[:sent_num]


def human_summary():
    """
    读取人工摘要数据
    :return: ans
    """
    anwser = open("news/anwser.txt", 'r')
    line = anwser.readline()
    ans = []
    i = 1
    while line:
        s = re.split(u'[\n ]', line)
        s = [sent for sent in s if len(sent) > 0]
        ans.append(s)
        line = anwser.readline()
    anwser.close()
    return ans

def human_summary_test():
    """
    读取人工摘要数据（测试用）
    :return: ans(二维列表)
    """
    anwser = open("news_2/answer.txt", 'r')
    line = anwser.readline()
    ans = []
    i = 1
    while line:
        s = re.split(u'[\n ]', line)
        s = [sent for sent in s if len(sent) > 0]
        ans.append(s)
        line = anwser.readline()
    anwser.close()
    return ans

def main_procedure(parameter, zoom):
    """
    计算F-measure值
    :param parameter(a):
    :param zoom(t)
    :return: sum
    """
    ssc = []
    for i in range(1,20 +1):

        filename = "news/" + str(i) +".txt"
        textfile = codecs.open(filename, 'r', 'utf8').read()
        res = extract_abstracts(textfile, sent_num=5, ratio=parameter, t = zoom)

        summary = human_summary()
        m_summary = []
        for s in res:
           m_summary.append(s['index'])

        #计算F-measure
        sum = 0
        k = len(m_summary)
        n = len(summary[i-1])
        for i in summary[i-1]:
            for j in m_summary:
                if int(i) == int(j):
                    sum+=1
        p = sum/k
        r = sum/n
        if (p+r) == 0:
            f_measure = 0
        else:
            f_measure = (2*p*r)/(p+r)
        ssc.append(f_measure)
    sum = 0
    for s in ssc:
        sum += s
    sum = sum/20
    return sum


def graph_pre():
    """
    形成数据框
    :return: ssc
    """
    ssc = []
    for i in range(0, 100, 10):
        mid = []
        for j in range(0, 100, 10):
            s = i/100
            t = j/10
            mid.append(main_procedure(s, t))
            print(i,' ',j)
        ssc.append(list(mid))
    return ssc


def Ruarua():
    """
    测试参数&模型检验
    :return:None,直接屏幕输出结果
    """
    for i in range(0, 100, 1):
        s = i/100
        t = 3
        print(main_procedure(s,t))


def calculate_R_pre(parameter, zoom):
    """
    计算F-measure值，用于测试模型
    :param parameter:
    :param zoom:
    :return:
    """
    ssc = []
    for i in range(1, 20 + 1):

        filename = "news_2/" + str(i) + ".txt"
        textfile = codecs.open(filename, 'r', 'utf8').read()
        res = extract_abstracts(textfile, sent_num=5, ratio=parameter, t=zoom)

        summary = human_summary_test()
        m_summary = []
        for s in res:
            m_summary.append(s['index'])

        # 计算F-measure
        sum = 0
        k = len(m_summary)
        n = len(summary[i - 1])
        for i in summary[i - 1]:
            for j in m_summary:
                if int(i) == int(j):
                    sum += 1
        p = sum / k
        r = sum / n
        if (p + r) == 0:
            f_measure = 0
        else:
            f_measure = (2 * p * r) / (p + r)
        ssc.append(f_measure)
    sum = 0
    for s in ssc:
        sum += s
    sum = sum / 20
    return sum


def Automatic_summarization(File_adress, num=5):
    filename = File_adress
    textfile = codecs.open(filename, 'r', 'utf8').read()
    res = extract_abstracts(textfile, sent_num=num, ratio=0.492, t=4)
    for s in res:
        print(s['sent'])

Automatic_summarization("news/6.txt", num=4)




