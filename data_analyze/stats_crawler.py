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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StatsCrawler:
    def __init__(self, base_url="https://www.stats.gov.cn/sj/zxfb/", output_dir="downloaded_articles"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        # 设置请求头，模拟浏览器
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
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
        return filename[:200]  # 限制长度
    
    def get_article_links(self, page_url):
        """获取页面上的所有文章链接"""
        try:
            logger.info(f"正在获取页面: {page_url}")
            response = self.session.get(page_url, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            article_links = []
            
            # 查找文章链接，通常在列表项中
            links = soup.find_all('a', href=True)
            for link in links:
                href = link.get('href')
                if href and 't20' in href and href.endswith('.html'):
                    # 构建完整URL
                    full_url = urljoin(page_url, href)
                    title = link.get_text(strip=True)
                    if title:
                        article_links.append({
                            'url': full_url,
                            'title': title
                        })
            
            logger.info(f"找到 {len(article_links)} 篇文章")
            return article_links
            
        except Exception as e:
            logger.error(f"获取文章链接失败: {e}")
            return []
    
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
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            # 保存HTML文件
            html_file = article_dir / f"{safe_title}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # 解析HTML查找附件
            soup = BeautifulSoup(response.text, 'html.parser')
            attachments = []
            
            # 查找附件链接（常见的附件格式）
            attachment_patterns = [
                r'\.(doc|docx|pdf|xls|xlsx|zip|rar)$',
                r'附件',
                r'下载'
            ]
            
            links = soup.find_all('a', href=True)
            for link in links:
                href = link.get('href')
                text = link.get_text(strip=True)
                
                # 检查是否是附件
                is_attachment = False
                for pattern in attachment_patterns:
                    if re.search(pattern, href, re.IGNORECASE) or re.search(pattern, text, re.IGNORECASE):
                        is_attachment = True
                        break
                
                if is_attachment:
                    attachments.append({
                        'url': urljoin(url, href),
                        'filename': text or os.path.basename(href)
                    })
            
            # 下载附件
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
                time.sleep(2)  # 重试时也添加延迟
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
                
                for article_info in article_links:
                    if self.download_article(article_info):
                        pass  # 成功或跳过
                    else:
                        pass  # 失败已在download_article中处理
                    
                    # 添加延迟，避免请求过于频繁
                    time.sleep(2)
                
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
    crawler = StatsCrawler()
    
    # 创建输出目录
    output_dir = Path("downloaded_articles")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("开始爬取国家统计局数据发布页面...")
    crawler.crawl_all_articles(max_pages=5, retry_failed=True)  # 爬取前5页，并重试失败的
    logger.info("爬取完成！")

if __name__ == "__main__":
    main() 