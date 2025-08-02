

import bs4
import os
import csv
import re
from save_data import dump_pickle,save_dict_to_pickle
import requests

# 读取清洗后的HTML文件
input_path = r"input\\2025年5月份70个大中城市商品住宅销售价格变动情况2.html"
path = os.path.join(os.path.dirname(__file__), input_path)
all_data={}
each_dict={}

def read_data_pkl(s="data.pkl"):
    global all_data
    if os.path.exists(s):
        all_data = dump_pickle(s)
        # print(all_data)
    
def save_data_pkl(s="data.pkl"):
    global all_data
    save_dict_to_pickle(all_data, s)






def read_link_from_pickle(s ):
    ''' link.pkl 保存了所有 link '''
    global all_data

    links = dump_pickle(s)

    
    # print(type(links))
    # exit()
    if not links:
        raise ValueError

    for i in links:
        read_link_from_link(i)

def read_link_from_link(link:dict ):
    i = link
    date_str = i["date"]
    link_str = i["href"]
    # print(date_str,all_data.keys())

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    if date_str not in all_data.keys():
        content = requests.get(link_str)
        print(date_str,link_str)
    
        response = requests.get(link_str, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        with open(r'input/201801.html','w',encoding = 'utf8') as f:
            f.write(response.text)
        
        # 解析页面
        # soup = BeautifulSoup(response.text, 'html.parser')

        d = process_one_link(response.text)
    
        all_data[date_str] = d
    else:
        print(f"{i['date']} 已经存在")
        


def replace_empty(s:str):
    s = s.replace('\u3000', '').replace('\xa0', '').replace(' ', '').replace('\t', '').replace('\n', '').replace('\r', '')
    return s

def process_one_link(html:str ):
    ''' 传入的是内容'''
    # with open(html_file, 'r', encoding='utf-8') as f:
    #     html = f.read()

    soup = bs4.BeautifulSoup(html, 'html.parser')
    


    # 只处理第一个表格
    
    # 提取表格上方的时间信息
    # 例如：表1：2025年5月70个大中城市新建商品住宅销售价格指数
    # caption = soup.find('p')
    # time_str = ''
    # if caption:
    #     cap_text = caption.get_text()
    #     m = re.search(r'(\d{4})年(\d{1,2})月', cap_text)
    #     if m:
    #         time_str = f"{m.group(1)}{m.group(2).zfill(2)}"
    #     else:
    #         time_str = cap_text.strip()
    # else:
    #     time_str = '未知时间'
    #     print("fail")
    #     return

    # 提取所有表格
    tables = soup.find_all('table')
    tables_size = len(tables)
    print(f"共找到 {tables_size} 个表格")
    # tables = caption.find_all("table")
    if tables_size==6:
        table = tables[0]
    elif tables_size==10:
        table = tables[1]
    elif tables_size==12:
        table = tables[0]
    elif tables_size==14:
        table = tables[0]
    else:
        print("wrong talbe size")
        return
    # if tables:

    if True:

        # str_ = "月70个大中城市新建商品住宅销售价格指数"
        if True:
        # for t in tables:
            # tmp = t.find_previous("div").find_previous("div").text
            # if str_ in tmp:
            if True:
                # print("找到有效表格")

                # table = t
                rows = table.find_all('tr')
                city_list = []
                values = []
                for row in rows[2:]:  # 跳过前两行表头
                    cols = row.find_all(['td', 'th'])
                    # 每行有8列，前4列为城市1，后4列为城市2
                    if len(cols) == 8:
                        # 城市1
                        city1 = replace_empty(cols[0].get_text(strip=True))
                        factor1 = cols[1].get_text(strip=True)
                        # print(replace_empty(city1))
                        each_dict[city1] = factor1

                        # print(city1,factor1)
                        city1 = replace_empty(cols[4].get_text(strip=True))
                        factor1 = cols[5].get_text(strip=True)
                        # print(replace_empty(city1))
                        each_dict[city1] = factor1
                        # print(city1,factor1)
                    elif len(cols) == 6:
                        # 城市1
                        city1 = replace_empty(cols[0].get_text(strip=True))
                        factor1 = cols[1].get_text(strip=True)
                        # print(replace_empty(city1))
                        each_dict[city1] = factor1

                        # print(city1,factor1)
                        city1 = replace_empty(cols[3].get_text(strip=True))
                        factor1 = cols[4].get_text(strip=True)
                        # print(replace_empty(city1))
                        each_dict[city1] = factor1
                        # print(city1,factor1)    
                    else:
                        print(f"len cols={len(cols)}, != 8")    
                
                # break # 只处理第一个匹配的表格
    
        
    else:
        print('未找到表格')

    return each_dict.copy()
# TODO city1 有问题，2018头部多了一个表格，有问题；2 个问题。

if __name__=="__main__":
    read_data_pkl()
    read_link_from_pickle("link.pkl")
    # read_link_from_link({"date":"201801",
    #     "text":"2018年1月份70个大中城市商品住宅销售价格变动情况",
    #     "href":"https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1899851.html"})

    # read_link_from_link({"date":"201812",
    #     "text":"2018年12月份70个大中城市商品住宅销售价格变动情况",
    #     "href":"https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900198.html"})
    # del all_data["202401"]
    # read_link_from_link({"date":"202401",
    #     "text":"2024年1月份70个大中城市商品住宅销售价格变动情况",
    #     "href":"https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202402/t20240223_1947808.html"})
    # TODO ,奇怪得数据 6
    # 202401 https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202402/t20240223_1947808.html
    # 202501 https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202502/t20250219_1958761.htm
    save_data_pkl() 


    pass


