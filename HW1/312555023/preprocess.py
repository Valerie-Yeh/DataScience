#-*- encoding:utf-8 -*-
from bs4 import BeautifulSoup as bs
import requests
import sys
import time
import json
from itertools import islice
import re
import splitfolders
import os
import random

payload = {
    'from': '/bbs/Beauty/index.html',
    'yes': 'yes'
}
rs = requests.session()
res = rs.post('https://www.ptt.cc/ask/over18', data=payload)

def crawl():
    for i in range(3634, 3923):
        url = r'https://www.ptt.cc/bbs/Beauty/index{}.html'.format(i)
        res = rs.get(url)
        content = res.text

        soup = bs(content, 'html.parser')
        entries = soup.find_all(class_='r-ent')

        for entry in entries:
            date = entry.find(class_='date').string
            date = date.replace('/', '').replace(' ', '0')
            if (i == 3922 and date == '0101') or (i == 3634 and date == '1231'): continue
            
            mark = entry.find(string = re.compile(u'[公告]'))
            mark_fw = entry.find(string = re.compile(u'Fw: [公告]'))
            if mark or mark_fw is not None: continue

            title = entry.find('a')
            if title is None: continue
            title = entry.find('a').string

            link = entry.find('a').get('href')
            if not link: continue
            link = 'https://www.ptt.cc' + link
            dict = {'date':date, 'title':title, 'url':link}
            out = json.dumps(dict, ensure_ascii=False)

            number = entry.find(class_='hl f3')
            pop_number = entry.find(class_='hl f1')
            single_num = entry.find(class_='hl f2')

            if number:
                number = number.string
                if number > '35':
                    with open('pop.jsonl', mode='a', encoding='utf-8') as popwriter:
                        popwriter.write(out + '\n')
                elif number <= '35':
                    with open('nonpop.jsonl', mode='a', encoding='utf-8') as writer:
                        writer.write(out + '\n')
            elif pop_number:
                with open('pop.jsonl', mode='a', encoding='utf-8') as popwriter:
                    popwriter.write(out + '\n')
            elif single_num:
                with open('nonpop.jsonl', mode='a', encoding='utf-8') as writer:
                    writer.write(out + '\n')

        if i % 10 == 0:
            time.sleep(0.5)

def popular():
    pop_article_url = []
    count = 0
    with open('pop.jsonl', 'r', encoding='utf-8') as reader:
        for line in reader:
            # deserialize into dict and return dict
            json_data = json.loads(line)

            url_value = json_data['url']
            pop_article_url.append(url_value)

    for i in range(len(pop_article_url)):
        url = pop_article_url[i]
        res = rs.get(url)
        content = res.text

        soup = bs(content, 'html.parser')
        entries = soup.find_all('img')
        for entry in entries:
            try:
                result = entry.get('src')
                r = requests.get(result, timeout=2)
                r.raise_for_status()

                with open(f"ptt_img/train_test/1/images{count+1}.jpg",'wb+') as f:
                    f.write(r.content)
                    count += 1
            except requests.exceptions.RequestException:
                pass

def nonpopular():
    nonpop_article_url = []
    count = 0
    with open('nonpop.jsonl', 'r', encoding='utf-8') as reader:
        for line in reader:
            # deserialize into dict and return dict
            json_data = json.loads(line)

            url_value = json_data['url']
            nonpop_article_url.append(url_value)

    for i in range(len(nonpop_article_url)):
        if count > 2850: break
        url = nonpop_article_url[i]
        res = rs.get(url)
        content = res.text

        soup = bs(content, 'html.parser')
        entries = soup.find_all('img')
        for entry in entries:
            try:
                result = entry.get('src')
                r = requests.get(result, timeout=2)
                r.raise_for_status()

                with open(f"ptt_img/train_test/0/images{count+1}.jpg",'wb+') as f:
                    f.write(r.content)
                    count += 1
            except requests.exceptions.RequestException:
                pass

def split():
    splitfolders.ratio('ptt_img', output='dataset', ratio=(0.8,0.1,0.1))

def get_img_path():
    dict0 = {}
    folder_0_path = 'ptt_img/train_test/0/'
    all_0_files = os.listdir(folder_0_path)
    for file in all_0_files:
        add_words = '_0'
        new_file = os.path.splitext(file)[0] + add_words + os.path.splitext(file)[1]
        os.rename(folder_0_path + file, folder_0_path + new_file)
        dict0[new_file] = 0
    #print(dict0)

    dict1 = {}
    folder_1_path = 'ptt_img/train_test/1/'
    all_1_files = os.listdir(folder_1_path)
    for file in all_1_files:
        add_words = '_1'
        new_file = os.path.splitext(file)[0] + add_words + os.path.splitext(file)[1]
        os.rename(folder_1_path + file, folder_1_path + new_file)
        dict1[new_file] = 1
    #print(dict1)
    dict = randomly_insert(dict0, dict1, len(dict0))
    dict = shuffle_dict(dict)
    #print(dict)
    image_paths = []
    image_groud_truths = []
    for key in dict.keys():
        image_paths.append(key)
    with open('image_paths.json', 'w') as file:
        json.dump({"image_paths": image_paths}, file, indent=4)

    for value in dict.values():
        image_groud_truths.append(value)
    with open('image_ground_truths.json', 'w') as f:
        json.dump({"image_ground_truths": image_groud_truths}, f, indent=4)

def randomly_insert(dict1, dict2, num_insertions):
    for _ in range(num_insertions):
        key, value = random.choice(list(dict1.items()))
        dict2[key] = value
    return dict2

def shuffle_dict(original_dict):
    items = list(original_dict.items())
    random.shuffle(items)
    shuffled_dict = dict(items)
    return shuffled_dict

if __name__ == '__main__':
    if sys.argv[1] == 'crawl':
        crawl()
    elif sys.argv[1] == 'popular':
        popular()
    elif sys.argv[1] == 'nonpopular':
        nonpopular()
    elif sys.argv[1] == 'split':
        split()
    elif sys.argv[1] == 'imgpath':
        get_img_path()