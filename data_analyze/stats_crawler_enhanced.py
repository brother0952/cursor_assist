import requests
from bs4 import BeautifulSoup
import os
import re
import time
from urllib.parse import urljoin, urlparse
import logging
from pathlib import Path
import json
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedStatsCrawler:
    def __init__(self, base_url="https://www.stats.gov.cn/sj/zxfb/", output_dir="downloaded_articles"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        
        # 设置请求头，模拟浏览器
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        })
        
        # 统计信息
        self.stats = {
            'total_articles': 0,
            'skipped_articles': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_attachments': 0,
            'successful_attachments': 0,
            'failed_attachments': 0
        }
        
        # 下载记录文件
        self.download_record_file = self.output_dir / "download_record.json"
        self.download_record = self.load_download_record()
        
    def load_download_record(self):
        """加载下载记录"""
        if self.download_record_file.exists():
            try:
                with open(self.download_record_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载下载记录失败: {e}")
                return {}
        return {}
    
    def save_download_record(self):
        """保存下载记录"""
        try:
            with open(self.download_record_file, 'w', encoding='utf-8') as f:
                json.dump(self.download_record, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存下载记录失败: {e}")
    
    def is_article_downloaded(self, url, title):
        """检查文章是否已下载"""
        safe_title = self.clean_filename(title)
        article_dir = self.output_dir / safe_title
        
        # 检查目录是否存在
        if not article_dir.exists():
            return False
        
        # 检查HTML文件是否存在
        html_file = article_dir / f"{safe_title}.html"
        if not html_file.exists():
            return False
        
        # 检查下载记录
        if url in self.download_record:
            record = self.download_record[url]
            if record.get('status') == 'success' and record.get('title') == title:
                return True
        
        return False
    
    def mark_article_downloaded(self, url, title, status='success'):
        """标记文章下载状态"""
        self.download_record[url] = {
            'title': title,
            'status': status,
            'download_time': datetime.now().isoformat()
        }
        self.save_download_record()
        
    def clean_filename(self, filename):
        """清理文件名，移除非法字符"""
        # 移除或替换非法字符
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\s+', '_', filename)
        # 移除开头和结尾的点和下划线
        filename = filename.strip('._')
        return filename[:150]  # 限制长度
    
    def get_page_content(self, url, retries=3):
        """获取页面内容，带重试机制"""
        for attempt in range(retries):
            try:
                logger.info(f"正在获取页面: {url} (尝试 {attempt + 1}/{retries})")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                response.encoding = 'utf-8'
                return response.text
            except requests.exceptions.RequestException as e:
                logger.warning(f"获取页面失败 (尝试 {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(5)  # 等待5秒后重试
                else:
                    logger.error(f"获取页面最终失败: {url}")
                    return None
        return None
    
    def get_article_links(self, page_url):
        """获取页面上的所有文章链接"""
        html_content = self.get_page_content(page_url)
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        article_links = []
        
        # 针对国家统计局网站的具体结构
        # 查找文章列表容器
        article_containers = soup.find_all(['ul', 'div'], class_=re.compile(r'list|news|article'))
        
        for container in article_containers:
            links = container.find_all('a', href=True)
            for link in links:
                href = link.get('href')
                title = link.get_text(strip=True)
                
                # 检查是否是文章链接
                if (href and 
                    ('t20' in href or 'html' in href) and 
                    title and 
                    len(title) > 5):  # 过滤掉太短的标题
                    
                    # 构建完整URL
                    full_url = urljoin(page_url, href)
                    
                    # 避免重复
                    if not any(article['url'] == full_url for article in article_links):
                        article_links.append({
                            'url': full_url,
                            'title': title
                        })
        
        # 如果没有找到特定结构的链接，尝试通用方法
        if not article_links:
            links = soup.find_all('a', href=True)
            for link in links:
                href = link.get('href')
                title = link.get_text(strip=True)
                
                if (href and 
                    't20' in href and 
                    href.endswith('.html') and 
                    title and 
                    len(title) > 5):
                    
                    full_url = urljoin(page_url, href)
                    if not any(article['url'] == full_url for article in article_links):
                        article_links.append({
                            'url': full_url,
                            'title': title
                        })
        
        logger.info(f"找到 {len(article_links)} 篇文章")
        return article_links
    
    def extract_article_info(self, soup, url):
        """提取文章的详细信息"""
        article_info = {
            'title': '',
            'publish_date': '',
            'content': '',
            'attachments': []
        }
        
        # 提取标题
        title_selectors = [
            'h1', 'h2', '.title', '.headline', '.article-title',
            'div[class*="title"]', 'div[class*="head"]'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                article_info['title'] = title_elem.get_text(strip=True)
                break
        
        # 提取发布日期
        date_selectors = [
            '.date', '.time', '.publish-date', '.article-date',
            'span[class*="date"]', 'div[class*="date"]'
        ]
        
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                article_info['publish_date'] = date_elem.get_text(strip=True)
                break
        
        # 提取正文内容
        content_selectors = [
            '.content', '.article-content', '.text', '.body',
            'div[class*="content"]', 'div[class*="text"]'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                article_info['content'] = content_elem.get_text(strip=True)
                break
        
        return article_info
    
    def find_attachments(self, soup, base_url):
        """查找文章中的附件"""
        attachments = []
        
        # 附件文件扩展名
        attachment_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar', '.txt']
        
        # 附件关键词
        attachment_keywords = ['附件', '下载', '表格', '数据', '文件', 'document', 'download']
        
        links = soup.find_all('a', href=True)
        for link in links:
            href = link.get('href')
            text = link.get_text(strip=True)
            
            # 检查是否是附件
            is_attachment = False
            
            # 检查文件扩展名
            for ext in attachment_extensions:
                if ext.lower() in href.lower():
                    is_attachment = True
                    break
            
            # 检查关键词
            if not is_attachment:
                for keyword in attachment_keywords:
                    if keyword in text or keyword in href:
                        is_attachment = True
                        break
            
            if is_attachment:
                attachment_url = urljoin(base_url, href)
                filename = text if text else os.path.basename(href)
                
                attachments.append({
                    'url': attachment_url,
                    'filename': filename,
                    'text': text
                })
        
        return attachments
    
    def download_article(self, article_info):
        """下载单篇文章及其附件"""
        url = article_info['url']
        title = article_info['title']
        
        # 检查是否已下载
        if self.is_article_downloaded(url, title):
            logger.info(f"文章已存在，跳过: {title}")
            self.stats['skipped_articles'] += 1
            return True
        
        # 创建文章目录
        safe_title = self.clean_filename(title)
        article_dir = self.output_dir / safe_title
        article_dir.mkdir(exist_ok=True)
        
        try:
            logger.info(f"正在下载文章: {title}")
            
            # 获取文章内容
            html_content = self.get_page_content(url)
            if not html_content:
                return False
            
            # 解析HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取文章信息
            article_data = self.extract_article_info(soup, url)
            
            # 保存HTML文件
            html_file = article_dir / f"{safe_title}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # 保存文章信息为JSON
            info_file = article_dir / "article_info.json"
            article_data['url'] = url
            article_data['download_time'] = datetime.now().isoformat()
            
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, ensure_ascii=False, indent=2)
            
            # 查找并下载附件
            attachments = self.find_attachments(soup, url)
            self.stats['total_attachments'] += len(attachments)
            
            for attachment in attachments:
                try:
                    attachment_url = attachment['url']
                    filename = self.clean_filename(attachment['filename'])
                    
                    # 如果没有扩展名，尝试从URL获取
                    if not os.path.splitext(filename)[1]:
                        parsed_url = urlparse(attachment_url)
                        ext = os.path.splitext(parsed_url.path)[1]
                        if ext:
                            filename += ext
                    
                    attachment_path = article_dir / filename
                    
                    logger.info(f"正在下载附件: {filename}")
                    attachment_response = self.session.get(attachment_url, timeout=60)
                    attachment_response.raise_for_status()
                    
                    with open(attachment_path, 'wb') as f:
                        f.write(attachment_response.content)
                    
                    logger.info(f"附件下载完成: {filename}")
                    self.stats['successful_attachments'] += 1
                    
                except Exception as e:
                    logger.error(f"下载附件失败 {attachment['url']}: {e}")
                    self.stats['failed_attachments'] += 1
            
            # 标记下载成功
            self.mark_article_downloaded(url, title, 'success')
            logger.info(f"文章下载完成: {title}")
            self.stats['successful_downloads'] += 1
            return True
            
        except Exception as e:
            logger.error(f"下载文章失败 {url}: {e}")
            # 标记下载失败
            self.mark_article_downloaded(url, title, 'failed')
            self.stats['failed_downloads'] += 1
            return False
    
    def retry_failed_articles(self):
        """重试之前失败的下载"""
        failed_articles = []
        for url, record in self.download_record.items():
            if record.get('status') == 'failed':
                failed_articles.append({
                    'url': url,
                    'title': record.get('title', '未知标题')
                })
        
        if failed_articles:
            logger.info(f"发现 {len(failed_articles)} 个之前失败的下载，开始重试...")
            for article_info in failed_articles:
                logger.info(f"重试下载: {article_info['title']}")
                self.download_article(article_info)
                time.sleep(3)  # 重试时也添加延迟
        else:
            logger.info("没有发现之前失败的下载")
    
    def crawl_all_articles(self, max_pages=10, retry_failed=True):
        """爬取所有页面的文章"""
        if retry_failed:
            self.retry_failed_articles()
        
        for page in range(1, max_pages + 1):
            page_url = f"{self.base_url}index_{page}.html" if page > 1 else self.base_url
            
            try:
                article_links = self.get_article_links(page_url)
                
                if not article_links:
                    logger.info(f"第 {page} 页没有找到文章，停止爬取")
                    break
                
                self.stats['total_articles'] += len(article_links)
                
                for i, article_info in enumerate(article_links, 1):
                    logger.info(f"处理第 {page} 页第 {i}/{len(article_links)} 篇文章")
                    self.download_article(article_info)
                    
                    # 添加延迟，避免请求过于频繁
                    time.sleep(3)
                
                logger.info(f"第 {page} 页处理完成")
                
            except Exception as e:
                logger.error(f"处理第 {page} 页失败: {e}")
                break
        
        self.print_stats()
    
    def print_stats(self):
        """打印统计信息"""
        logger.info("=" * 50)
        logger.info("爬取统计信息:")
        logger.info(f"总文章数: {self.stats['total_articles']}")
        logger.info(f"跳过文章: {self.stats['skipped_articles']}")
        logger.info(f"成功下载: {self.stats['successful_downloads']}")
        logger.info(f"下载失败: {self.stats['failed_downloads']}")
        logger.info(f"总附件数: {self.stats['total_attachments']}")
        logger.info(f"附件成功: {self.stats['successful_attachments']}")
        logger.info(f"附件失败: {self.stats['failed_attachments']}")
        logger.info("=" * 50)

def main():
    """主函数"""
    crawler = EnhancedStatsCrawler()
    
    # 创建输出目录
    output_dir = Path("downloaded_articles")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("开始爬取国家统计局数据发布页面...")
    crawler.crawl_all_articles(max_pages=3, retry_failed=True)  # 爬取前3页，并重试失败的
    logger.info("爬取完成！")

if __name__ == "__main__":
    main() 