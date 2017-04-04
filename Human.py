from jieba import *
import codecs

def split_sentences(full_text, num):
    #分离句子（除标题外）
    seats = re.split(u'[。?!？！]', full_text)
    seats = [sent for sent in seats if len(sent) > 0]
    #分离标题,将标题放至列表末端保存
    title = re.split(u'[\n]', seats[0])
    seats[0] = title[1]
    seats.append(title[0])
    length = len(seats)
    for i in range(0,length-1 + 1):
        seats[i] = seats[i].replace('\n', '').replace('\r', '')

    #处理后引号问题
    for i in range(0, length-1 +1):
        try:
            if(seats[i][0]) == '”':
                l = list(seats[i])
                l[0] = ''
                seats[i] = ''.join(l)
                seats[i-1] += "”"
        except:
            s = "Error:文章（" + str(num)+ "）第" +str(i) + "句处理后引号失败"
            print(s)
            continue
    seats = [sent for sent in seats if len(sent) > 0]
    return seats

def split(File_address):
    for i in range(1, 20 + 1):
        file_name = File_address
        textfile = codecs.open(file_name, 'r', 'utf8').read()
        text = split_sentences(textfile, i)

        file_name = file_name + "read" + str(i) + ".txt"
        file = open(file_name, 'w', encoding='utf8')
        for i in range(0, len(text) - 1 + 1):
            s = '[' + str(i) + "]  " + text[i] + "\n"
            file.write(s)
        file.close()

"""
for i in range(1, 20 +1):
    file_name ="news_2/" + str(i) + ".txt"
    textfile = codecs.open(file_name, 'r', 'utf8').read()
    text = split_sentences(textfile,i)

    file_name ="news_2/read_" + str(i) + ".txt"
    file = open(file_name, 'w', encoding='utf8')
    for i in range(0, len(text) - 1 + 1):
        s = '[' + str(i) + "]  " + text[i] + "\n"
        file.write(s)
    file.close()
"""
