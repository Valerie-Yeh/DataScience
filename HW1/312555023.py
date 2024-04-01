#-*- encoding:utf-8 -*-
from bs4 import BeautifulSoup as bs
import requests
import sys
import time
import json
from itertools import islice
import re

payload = {
    'from': '/bbs/Beauty/index.html',
    'yes': 'yes'
}
rs = requests.session()
res = rs.post('https://www.ptt.cc/ask/over18', data=payload)

def crawl():
    latest_url = 'https://www.ptt.cc/bbs/Beauty/index.html'
    count = 4000
    while True:
        count -= 1
        if count < 3600:
            end_idx = 3923
            start_idx = 3605
            break
        tmp_res = rs.get(latest_url)
        tmp_content = tmp_res.text
        soup = bs(tmp_content, 'html.parser')
        
        titles_list = []
        tmp_entities = soup.find_all(class_='r-ent')
        for tmp_entity in tmp_entities:
            if tmp_entity.find('a') is None: continue
            tmp_title = tmp_entity.find('a').string
            titles_list.append(tmp_title)
        if '[正妹]今天過後，這一年出生的將全數邁入40大關' in titles_list:
            end_idx_url = soup.find_all(class_='btn wide')[2].get('href')
            pattern = '[0-9]+'
            end_idx = int(re.findall(pattern, end_idx_url)[0])
            start_idx = end_idx - 289
            #print("Fetch start and end idx!")
            break
        else:
            latest_url = soup.find_all(class_='btn wide')[1].get('href')
            latest_url = 'https://www.ptt.cc' + latest_url
        #print(latest_url)

    #print("Start parsing!")
    for i in range(start_idx, end_idx):
        url = r'https://www.ptt.cc/bbs/Beauty/index{}.html'.format(i)
        res = rs.get(url)
        content = res.text

        soup = bs(content, 'html.parser')
        entries = soup.find_all(class_='r-ent')

        for entry in entries:
            date = entry.find(class_='date').string
            date = date.replace('/', '').replace(' ', '0')
            if (i == (end_idx-1) and date == '0101') or (i == start_idx and date == '1231'): continue
            
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
            
            with open('articles.jsonl', mode='a', encoding='utf-8') as writer:
                out = json.dumps(dict, ensure_ascii=False)
                writer.write(out + '\n')
            
            pop = entry.find(class_='hl f1')
            if pop is not None:
                with open('popular_articles.jsonl', mode='a', encoding='utf-8') as popwriter:
                    popwriter.write(out + '\n')

        if i % 10 == 0:
            time.sleep(2)

def push(sd, ed):
    result_url = []
    with open('articles.jsonl', 'r', encoding='utf-8') as reader:
        for line in reader:
            # deserialize into dict and return dict
            json_data = json.loads(line)

            date_value = json_data['date']
            url_value = json_data['url']

            if sd <= date_value <= ed:
                result_url.append(url_value)

    push_userid_dict = {}
    boo_userid_dict = {}
    push_total = 0
    boo_total = 0
    for i in range(len(result_url)):
        url = result_url[i]
        res = rs.get(url)
        content = res.text

        soup = bs(content, 'html.parser')
        entries = soup.find_all(class_='push')

        for entry in entries:
            push_tag = entry.find(class_='hl push-tag')
            push_userid = entry.find(class_='f3 hl push-userid').string
            boo_tag = entry.find(class_='f1 hl push-tag')
            boo_userid = entry.find(class_='f3 hl push-userid').string

            if push_tag is not None:
                push_userid_dict[push_userid] = push_userid_dict.get(push_userid, 0) + 1
                push_total += 1
            elif (boo_tag is not None) and (boo_tag.string == '噓 '):
                boo_userid_dict[boo_userid] = boo_userid_dict.get(boo_userid, 0) + 1
                boo_total += 1
    sorted_push_userid_dict = dict(sorted(push_userid_dict.items(), key=lambda item:(item[1], item[0]), reverse=True))
    sorted_push_userid_dict = dict(islice(sorted_push_userid_dict.items(), 10))
    top10_sorted_push_userid_dict = []
    for user_id, count in sorted_push_userid_dict.items():
        top10_sorted_push_userid_dict.append({"user_id": user_id, "count": count})

    sorted_boo_userid_dict = dict(sorted(boo_userid_dict.items(), key=lambda item: (item[1], item[0]), reverse=True))
    sorted_boo_userid_dict = dict(islice(sorted_boo_userid_dict.items(), 10))
    top10_sorted_boo_userid_dict = []
    for user_id, count in sorted_boo_userid_dict.items():
        top10_sorted_boo_userid_dict.append({"user_id": user_id, "count": count})

    push_data = {
        "total": push_total,
        "top10": top10_sorted_push_userid_dict
    }

    boo_data = {
        "total": boo_total,
        "top10": top10_sorted_boo_userid_dict
    }

    with open(r'push_{}_{}.json'.format(sd, ed), 'w') as file:
        json.dump({"push": push_data, "boo": boo_data}, file, indent=4)

def popular(sd, ed):
    pop_article_url = []
    image_url_list = []
    article_num = 0
    with open('popular_articles.jsonl', 'r', encoding='utf-8') as reader:
        for line in reader:
            # deserialize into dict and return dict
            json_data = json.loads(line)

            date_value = json_data['date']
            url_value = json_data['url']
            article_num += 1
            if sd <= date_value <= ed:
                pop_article_url.append(url_value)

    for i in range(len(pop_article_url)):
        url = pop_article_url[i]
        res = rs.get(url)
        content = res.text

        soup = bs(content, 'html.parser')
        entries = soup.find_all('a')
        for entry in entries:
            image_url = entry.get('href')
            pattern = '(https|http)+(.//)+[A-Za-z0-9._/]+\.(jpg|jpeg|png|gif)'
            result = re.findall(pattern, image_url)
            if result:
                image_url_list.append(image_url)

    with open(r'popular_{}_{}.json'.format(sd, ed), 'w') as file:
        json.dump({"number_of_popular_articles": article_num, "image_urls": image_url_list}, file, indent=4)

def keyword(sd, ed, kw):
    article_url = []
    image_url_list = []
    with open('articles.jsonl', 'r', encoding='utf-8') as reader:
        for line in reader:
            # deserialize into dict and return dict
            json_data = json.loads(line)

            date_value = json_data['date']
            url_value = json_data['url']

            if sd <= date_value <= ed:
                article_url.append(url_value)

    for i in range(len(article_url)):
        url = article_url[i]
        res = rs.get(url)
        content = res.text
        soup = bs(content, 'html.parser')

        station = soup.find(string = re.compile(u'※ 發信站:'))
        if station is None: continue

        words  = soup.find(id='main-content').get_text().split()
        index_of_re_word = words.index('※')
        context = ' '.join(words[:index_of_re_word])

        if kw not in context: continue
        #print(context)
        entries = soup.find_all('a')
        for entry in entries:
            image_url = entry.get('href')
            pattern = 'https{0,1}://.*\.(?i:jpg|jpeg|png|gif)'
            result = re.findall(pattern, image_url)
            if result:
                image_url_list.append(image_url)

    with open(r'keyword_{}_{}_{}.json'.format(sd, ed, kw), 'w') as file:
        json.dump({"image_urls": image_url_list}, file, indent=4)

               
if __name__ == '__main__':
    if sys.argv[1] == 'crawl':
        crawl()
    elif sys.argv[1] == 'push':
        push(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'popular':
        popular(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'keyword':
        keyword(sys.argv[2], sys.argv[3], sys.argv[4])
       