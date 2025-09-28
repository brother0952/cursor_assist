#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试页面链接
"""

import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_links():
    """调试页面上的所有链接"""
    
    try:
        logger.info("开始调试页面链接...")
        
        # 配置Chrome选项
        chrome_options = Options()
        # chrome_options.add_argument('--headless')  # 注释掉无头模式，方便观察
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # 使用本地ChromeDriver
        chromedriver_path = r"D:\Users\xianyuchao\Downloads\chromedriver-win64\chromedriver.exe"
        
        if not os.path.exists(chromedriver_path):
            chromedriver_path = r"C:\Python311\chromedriver.exe"
        
        if not os.path.exists(chromedriver_path):
            logger.error("找不到ChromeDriver，请检查路径")
            return False
        
        logger.info(f"使用ChromeDriver路径: {chromedriver_path}")
        service = Service(chromedriver_path)
        
        # 创建WebDriver
        logger.info("正在启动Chrome浏览器...")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # 访问搜索页面
        year = 2018
        search_url = f"https://www.stats.gov.cn/search/s?qt={year}%2070%E4%B8%AA%E5%A4%A7%E4%B8%AD%E5%9F%8E%E5%B8%82%E5%95%86%E5%93%81%E4%BD%8F&siteCode=bm36000002&tab=all&toolsStatus=1"
        driver.get(search_url)
        
        # 等待页面加载
        time.sleep(6)
        
        # 获取页面标题
        page_title = driver.title
        logger.info(f"搜索页面标题: {page_title}")
        
        # 导入并调用筛选按钮点击函数
        from main import click_stats_filter_button, click_title_search_option
        
        # 点击"统计数据"筛选按钮
        stats_success = click_stats_filter_button(driver)
        
        if stats_success:
            logger.info("✅ 统计数据筛选按钮点击成功！")
        else:
            logger.warning("❌ 统计数据筛选按钮点击失败")
        
        # 点击"标题"搜索位置选项
        title_success = click_title_search_option(driver)
        
        if title_success:
            logger.info("✅ 标题搜索位置选项点击成功！")
        else:
            logger.warning("❌ 标题搜索位置选项点击失败")
        
        # 等待页面更新
        time.sleep(3)
        
        # 获取所有链接
        links = driver.find_elements(By.TAG_NAME, "a")
        logger.info(f"找到 {len(links)} 个链接")
        
        # 详细分析每个链接
        logger.info("=== 详细分析所有链接 ===")
        for i, link in enumerate(links):
            try:
                text = link.text.strip()
                href = link.get_attribute('href')
                
                if text and len(text) > 5:  # 只显示有意义的文本
                    logger.info(f"链接 {i+1}:")
                    logger.info(f"  文本: '{text}'")
                    logger.info(f"  href: {href}")
                    
                    # 检查是否包含关键词
                    keywords = ['70个大中城市', '商品住宅', '销售价格', '房价']
                    found_keywords = [kw for kw in keywords if kw in text]
                    if found_keywords:
                        logger.info(f"  包含关键词: {found_keywords}")
                    
                    # 检查是否匹配正则表达式
                    from main import match_title
                    ret_date = match_title(text)
                    if ret_date:
                        logger.info(f"  匹配日期: {ret_date}")
                    
                    logger.info("  ---")
            except Exception as e:
                logger.warning(f"处理链接 {i+1} 时出错: {e}")
                continue
        
        # 保存页面源码
        page_source = driver.page_source
        with open('debug_page_source.html', 'w', encoding='utf-8') as f:
            f.write(page_source)
        logger.info("页面源码已保存到 debug_page_source.html")
        
        # 等待用户观察
        logger.info("等待20秒后关闭浏览器...")
        time.sleep(20)
        
        # 关闭浏览器
        driver.quit()
        
        return True
        
    except Exception as e:
        logger.error(f"调试失败: {e}")
        return False

if __name__ == "__main__":
    debug_links() 