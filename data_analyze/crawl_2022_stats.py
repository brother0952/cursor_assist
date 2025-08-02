import os
import re
import time
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

SEARCH_URL = "https://www.stats.gov.cn/search/s?qt=2022%2070%E4%B8%AA%E5%A4%A7%E4%B8%AD%E5%9F%8E%E5%B8%82%E5%95%86%E5%93%81&siteCode=bm36000002&tab=all&toolsStatus=1"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
}
OUTPUT_DIR = "output_2022_stats"

# 匹配标题的正则
TITLE_PATTERN = re.compile(r"主动公开2022年(\d{1,2})月份70个大中城市商品住宅销售价格变动情况")

# 匹配表格标题
TABLE_TITLE_PATTERN = re.compile(r"表(\d)：")

def get_month_links():
    """用selenium获取2022年每月的目标链接"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_service = Service(r'D:\Users\xianyuchao\Downloads\chromedriver-win64\chromedriver.exe')
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    driver.get(SEARCH_URL)
    # 等待包含"主动公开2022年"的元素出现
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, "//*[contains(text(), '主动公开2022年')]"))
    )
    time.sleep(2)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for a in soup.find_all('a'):
        title = a.get('title') or a.text
        m = TITLE_PATTERN.match(title)
        if m:
            month = int(m.group(1))
            url = a.get('href')
            if not url.startswith('http'):
                url = 'https://www.stats.gov.cn' + url
            links.append((month, title, url))
    links.sort(key=lambda x: x[0])
    return links

def fetch_and_save_tables(month, title, url):
    """抓取页面表格并保存为csv"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = resp.apparent_encoding
        # 有些页面表格在iframe里
        soup = BeautifulSoup(resp.text, 'html.parser')
        iframes = soup.find_all('iframe')
        if iframes:
            iframe_url = iframes[0].get('src')
            if not iframe_url.startswith('http'):
                iframe_url = 'https://www.stats.gov.cn' + iframe_url
            resp = requests.get(iframe_url, headers=HEADERS, timeout=20)
            resp.encoding = resp.apparent_encoding
        # 解析表格
        tables = pd.read_html(resp.text)
        logging.info(f"{month}月: 共解析到{len(tables)}个表格")
        for idx, table in enumerate(tables, 1):
            filename = f"2022年{month}月_表{idx}.csv"
            filepath = os.path.join(OUTPUT_DIR, filename)
            table.to_csv(filepath, index=False, encoding='utf-8-sig')
            logging.info(f"保存: {filepath}")
    except Exception as e:
        logging.error(f"{month}月抓取或保存失败: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    logging.info("开始获取2022年每月数据链接...")
    links = get_month_links()
    logging.info(f"共获取到{len(links)}个月份数据链接")
    for i, (month, title, url) in enumerate(links, 1):
        logging.info(f"[{i}/{len(links)}] 处理: {title} -> {url}")
        fetch_and_save_tables(month, title, url)
        time.sleep(2)  # 避免请求过快被封
    logging.info("全部完成！")

if __name__ == "__main__":
    main() 