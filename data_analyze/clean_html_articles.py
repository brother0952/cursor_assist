import os
from pathlib import Path
from bs4 import BeautifulSoup
import re
import logging
import pandas as pd
from io import StringIO

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_main_content(html, title):
    """提取正文和标题，移除无关内容，返回简化后的HTML字符串"""
    soup = BeautifulSoup(html, 'html.parser')
    # 先移除常见无用区块
    useless_selectors = [
        {'name': 'div', 'class_': 'footer'},
        {'name': 'div', 'class_': 'header'},
        {'name': 'div', 'class_': 'nav'},
        {'name': 'div', 'class_': 'sidebar'},
        {'name': 'div', 'class_': 'listbox', 'id': 'newgwybmbox'},
        {'name': 'div', 'id': 'newgwybmbox'},
    ]
    for sel in useless_selectors:
        found = soup.find_all(sel.get('name'), class_=sel.get('class_'), id=sel.get('id'))
        for tag in found:
            tag.decompose()
    # 常见正文容器
    main_selectors = [
        {'name': 'div', 'class_': 'TRS_Editor'},
        {'name': 'div', 'class_': 'article'},
        {'name': 'div', 'id': 'content'},
        {'name': 'div', 'id': 'zoom'},
    ]
    main_content = None
    for sel in main_selectors:
        main_content = soup.find(sel.get('name'), class_=sel.get('class_'), id=sel.get('id'))
        if main_content:
            break
    if not main_content:
        # 兜底：取最大段落数的div
        divs = soup.find_all('div')
        if divs:
            main_content = max(divs, key=lambda d: len(d.find_all('p')))
        else:
            main_content = soup.body or soup
    head = f'<meta charset="utf-8"><title>{title}</title>'
    body = f'<h1>{title}</h1>\n' + str(main_content)
    simple_html = f'<html><head>{head}</head><body>{body}</body></html>'
    return simple_html

def get_title_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    # 优先用 h1，其次 title
    h1 = soup.find('h1')
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    title = soup.title.string if soup.title else None
    if title:
        return title.strip()
    return "无标题"

def clean_all_htmls(root_dir):
    root = Path(root_dir)
    html_files = list(root.glob('**/*.html'))
    logger.info(f"共找到 {len(html_files)} 个 html 文件，开始处理...")
    for html_file in html_files:
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html = f.read()
            title = get_title_from_html(html)
            cleaned_html = extract_main_content(html, title)
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_html)
            logger.info(f"处理完成: {html_file}")
        except Exception as e:
            logger.error(f"处理失败: {html_file}，原因: {e}")

def main():
    clean_all_htmls('downloaded_articles')
    logger.info("全部处理完成！")

def extract_tables_from_html():
    for root, dirs, files in os.walk('downloaded_articles'):
        for file in files:
            if file.endswith('.html'):
                html_path = os.path.join(root, file)
                with open(html_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f, 'html.parser')
                tables = soup.find_all('table')
                if not tables:
                    print(f"未找到表格: {html_path}")
                    continue
                for idx, table in enumerate(tables):
                    df = pd.read_html(StringIO(str(table)))[0]
                    # 构造输出csv路径
                    rel_dir = os.path.relpath(root, 'downloaded_articles')
                    csv_dir = os.path.join('csv_output', rel_dir)
                    os.makedirs(csv_dir, exist_ok=True)
                    base_name = os.path.splitext(file)[0]
                    csv_name = f"{base_name}_table{idx+1}.csv" if len(tables) > 1 else f"{base_name}.csv"
                    csv_path = os.path.join(csv_dir, csv_name)
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    print(f"已保存: {csv_path}")

if __name__ == '__main__':
    main()
    extract_tables_from_html() 