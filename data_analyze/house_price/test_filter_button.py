#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试筛选按钮点击功能
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

def test_filter_button():
    """测试筛选按钮点击功能"""
    
    try:
        logger.info("开始测试筛选按钮点击功能...")
        
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
        year = 2023
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
        
        # 等待一下，然后检查页面是否发生了变化
        time.sleep(3)
        
        # 再次获取页面信息
        new_links = driver.find_elements(By.TAG_NAME, "a")
        logger.info(f"筛选后找到 {len(new_links)} 个链接")
        
        # 检查是否有房价相关的链接
        house_price_count = 0
        for link in new_links:
            try:
                text = link.text.strip()
                if '70个大中城市' in text and '商品住宅' in text:
                    house_price_count += 1
                    logger.info(f"找到房价相关链接: {text}")
            except Exception:
                continue
        
        logger.info(f"筛选后找到 {house_price_count} 个房价相关链接")
        
        success = stats_success or title_success
        
        # 等待用户观察
        logger.info("等待10秒后关闭浏览器...")
        time.sleep(10)
        
        # 关闭浏览器
        driver.quit()
        
        return success
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    test_filter_button() 