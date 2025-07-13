#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试爬虫功能
"""

import requests
from bs4 import BeautifulSoup
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_website_access():
    """测试网站访问"""
    url = "https://www.stats.gov.cn/sj/zxfb/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    try:
        logger.info("测试网站访问...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        logger.info(f"网站访问成功，状态码: {response.status_code}")
        logger.info(f"页面大小: {len(response.text)} 字符")
        
        # 解析页面
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找文章链接
        links = soup.find_all('a', href=True)
        article_links = []
        
        for link in links:
            href = link.get('href')
            title = link.get_text(strip=True)
            
            if (href and 
                ('t20' in href or 'html' in href) and 
                title and 
                len(title) > 5):
                article_links.append({
                    'url': href,
                    'title': title
                })
        
        logger.info(f"找到 {len(article_links)} 个可能的文章链接")
        
        # 显示前5个链接
        for i, link in enumerate(article_links[:5], 1):
            logger.info(f"{i}. {link['title']} -> {link['url']}")
        
        return True
        
    except Exception as e:
        logger.error(f"网站访问失败: {e}")
        return False

def test_single_article():
    """测试单篇文章下载"""
    article_url = "https://www.stats.gov.cn/sj/zxfb/202507/t20250710_1960379.html"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    try:
        logger.info("测试单篇文章下载...")
        response = requests.get(article_url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        logger.info(f"文章下载成功，状态码: {response.status_code}")
        
        # 解析文章内容
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找标题
        title_elem = soup.find(['h1', 'h2', '.title', '.headline'])
        if title_elem:
            title = title_elem.get_text(strip=True)
            logger.info(f"文章标题: {title}")
        
        # 查找附件
        attachments = []
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link.get('href')
            text = link.get_text(strip=True)
            
            # 检查是否是附件
            attachment_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar']
            attachment_keywords = ['附件', '下载', '表格', '数据']
            
            is_attachment = False
            for ext in attachment_extensions:
                if ext.lower() in href.lower():
                    is_attachment = True
                    break
            
            if not is_attachment:
                for keyword in attachment_keywords:
                    if keyword in text or keyword in href:
                        is_attachment = True
                        break
            
            if is_attachment:
                attachments.append({
                    'url': href,
                    'text': text
                })
        
        logger.info(f"找到 {len(attachments)} 个附件")
        for i, attachment in enumerate(attachments, 1):
            logger.info(f"附件 {i}: {attachment['text']} -> {attachment['url']}")
        
        return True
        
    except Exception as e:
        logger.error(f"文章下载失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始测试爬虫功能...")
    
    # 测试网站访问
    if test_website_access():
        logger.info("网站访问测试通过")
    else:
        logger.error("网站访问测试失败")
        return
    
    # 测试单篇文章下载
    if test_single_article():
        logger.info("单篇文章下载测试通过")
    else:
        logger.error("单篇文章下载测试失败")
        return
    
    logger.info("所有测试完成！")

if __name__ == "__main__":
    main() 