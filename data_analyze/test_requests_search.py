import requests
import time
import logging
from bs4 import BeautifulSoup
from urllib.parse import quote

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_stats_search():
    """测试国家统计局搜索功能"""
    try:
        logger.info("开始测试国家统计局搜索功能...")
        
        # 创建session
        session = requests.Session()
        
        # 设置请求头，模拟浏览器
        session.headers.update({
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
        
        # 首先访问搜索页面
        search_base_url = "https://www.stats.gov.cn/search/s"
        logger.info(f"正在访问搜索页面: {search_base_url}")
        
        response = session.get(search_base_url, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        logger.info(f"搜索页面状态码: {response.status_code}")
        logger.info(f"搜索页面标题: {BeautifulSoup(response.text, 'html.parser').title.string if BeautifulSoup(response.text, 'html.parser').title else '无标题'}")
        
        # 保存搜索页面
        with open('stats_search_base.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        logger.info("搜索页面已保存到 stats_search_base.html")
        
        # 构建搜索URL
        search_query = "2024 70个大中城市商品住房"
        encoded_query = quote(search_query)
        search_url = f"{search_base_url}?qt={encoded_query}"
        
        logger.info(f"正在搜索: {search_query}")
        logger.info(f"搜索URL: {search_url}")
        
        # 执行搜索
        response = session.get(search_url, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        logger.info(f"搜索结果页面状态码: {response.status_code}")
        
        # 保存搜索结果页面
        with open('stats_search_results.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        logger.info("搜索结果页面已保存到 stats_search_results.html")
        
        # 解析搜索结果
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找所有链接
        links = soup.find_all('a', href=True)
        logger.info(f"找到 {len(links)} 个链接")
        
        # 查找包含"主动公开"的链接
        active_public_links = []
        for link in links:
            text = link.get_text(strip=True)
            if '主动公开' in text:
                active_public_links.append({
                    'text': text,
                    'href': link.get('href')
                })
                logger.info(f"找到主动公开链接: {text}")
        
        logger.info(f"总共找到 {len(active_public_links)} 个主动公开链接")
        
        # 查找房价相关的链接
        house_price_links = []
        keywords = ['70个大中城市', '商品住房', '房价', '住宅销售价格', '商品住宅']
        
        for link in links:
            text = link.get_text(strip=True)
            href = link.get('href')
            
            # 检查是否包含房价相关关键词
            is_house_price = any(keyword in text for keyword in keywords)
            
            if is_house_price and href and 't20' in href and href.endswith('.html'):
                house_price_links.append({
                    'text': text,
                    'href': href
                })
                logger.info(f"找到房价相关链接: {text}")
        
        logger.info(f"总共找到 {len(house_price_links)} 个房价相关链接")
        
        return len(active_public_links) > 0 or len(house_price_links) > 0
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False

def test_data_page():
    """测试数据发布页面"""
    try:
        logger.info("开始测试数据发布页面...")
        
        # 创建session
        session = requests.Session()
        
        # 设置请求头
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # 访问数据发布页面
        data_url = "https://www.stats.gov.cn/sj/zxfb/"
        logger.info(f"正在访问数据发布页面: {data_url}")
        
        response = session.get(data_url, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        logger.info(f"数据发布页面状态码: {response.status_code}")
        
        # 保存页面
        with open('stats_data_page.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        logger.info("数据发布页面已保存到 stats_data_page.html")
        
        # 解析页面
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找所有链接
        links = soup.find_all('a', href=True)
        logger.info(f"找到 {len(links)} 个链接")
        
        # 查找房价相关的链接
        house_price_links = []
        keywords = ['70个大中城市', '商品住房', '房价', '住宅销售价格', '商品住宅']
        
        for link in links:
            text = link.get_text(strip=True)
            href = link.get('href')
            
            # 检查是否包含房价相关关键词
            is_house_price = any(keyword in text for keyword in keywords)
            
            if is_house_price and href and 't20' in href and href.endswith('.html'):
                house_price_links.append({
                    'text': text,
                    'href': href
                })
                logger.info(f"找到房价相关链接: {text}")
        
        logger.info(f"总共找到 {len(house_price_links)} 个房价相关链接")
        
        return len(house_price_links) > 0
        
    except Exception as e:
        logger.error(f"测试数据发布页面失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始测试国家统计局网站...")
    
    # 测试搜索功能
    if test_stats_search():
        logger.info("搜索功能测试成功！")
    else:
        logger.warning("搜索功能测试失败")
    
    # 测试数据发布页面
    if test_data_page():
        logger.info("数据发布页面测试成功！")
    else:
        logger.warning("数据发布页面测试失败")
    
    logger.info("测试完成！")

if __name__ == "__main__":
    main() 