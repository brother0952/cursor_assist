import time
import logging
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_selenium_local():
    """使用本地ChromeDriver测试Selenium"""
    try:
        logger.info("开始测试Selenium（使用本地ChromeDriver）...")
        
        # 配置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # 无头模式
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # 使用本地ChromeDriver
        chromedriver_path = r"D:\Users\xianyuchao\Downloads\chromedriver-win64\chromedriver.exe"
        
        if not os.path.exists(chromedriver_path):
            # 尝试Python目录下的ChromeDriver
            chromedriver_path = r"C:\Python311\chromedriver.exe"
        
        if not os.path.exists(chromedriver_path):
            logger.error("找不到ChromeDriver，请检查路径")
            return False
        
        logger.info(f"使用ChromeDriver路径: {chromedriver_path}")
        service = Service(chromedriver_path)
        
        # 创建WebDriver
        logger.info("正在启动Chrome浏览器...")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # # 测试访问百度
        # logger.info("正在访问测试页面...")
        # driver.get("https://www.baidu.com")
        
        # # 等待页面加载
        # wait = WebDriverWait(driver, 10)
        # title = wait.until(EC.presence_of_element_located((By.TAG_NAME, "title")))
        
        # logger.info(f"页面标题: {title.get_attribute('textContent')}")
        # logger.info("Selenium基本设置测试成功！")
        
        # 测试访问国家统计局搜索页面
        logger.info("正在访问国家统计局搜索页面...")
        # search_url = "https://www.stats.gov.cn/search/s?qt=2023%2070%E4%B8%AA%E5%A4%A7%E4%B8%AD%E5%9F%8E%E5%B8%82%E5%95%86%E5%93%81%E4%BD%8F%E6%88%BF"
        search_url = "https://www.stats.gov.cn/search/s?qt=2023%2070%E4%B8%AA%E5%A4%A7%E4%B8%AD%E5%9F%8E%E5%B8%82%E5%95%86%E5%93%81%E4%BD%8F&siteCode=bm36000002&tab=all&toolsStatus=1"
        driver.get(search_url)
        
        # 等待页面加载
        time.sleep(10)
        
        # 获取页面标题
        page_title = driver.title
        logger.info(f"搜索页面标题: {page_title}")
        
        # 查找页面中的链接
        links = driver.find_elements(By.TAG_NAME, "a")
        logger.info(f"找到 {len(links)} 个链接")
        
        # 查找包含"主动公开"的链接
        active_public_links = []
        for link in links:
            try:
                text = link.text.strip()
                # if '主动公开' in text:
                if text.endswith('变动情况'):
                    active_public_links.append({
                        'text': text,
                        'href': link.get_attribute('href')
                    })
                    logger.info(f"找到主动公开链接: {text}")
                else:
                    # logger.info(f"非主动公开链接: {text}")    
                    pass
            except Exception as e:
                continue
        
        logger.info(f"总共找到 {len(active_public_links)} 个主动公开链接")
        
        if False: # TODO

            # 查找房价相关的链接
            house_price_links = []
            keywords = ['70个大中城市', '商品住房', '房价', '住宅销售价格', '商品住宅']
            
            for link in links:
                try:
                    text = link.text.strip()
                    href = link.get_attribute('href')
                    
                    # 检查是否包含房价相关关键词
                    is_house_price = any(keyword in text for keyword in keywords)
                    
                    if is_house_price and href and 't20' in href and href.endswith('.html'):
                        house_price_links.append({
                            'text': text,
                            'href': href
                        })
                        logger.info(f"找到房价相关链接: {text}")
                except Exception as e:
                    continue
            
            logger.info(f"总共找到 {len(house_price_links)} 个房价相关链接")
            
            # 保存页面源码用于调试
            page_source = driver.page_source
            with open('stats_search_selenium_local_debug.html', 'w', encoding='utf-8') as f:
                f.write(page_source)
            logger.info("页面源码已保存到 stats_search_selenium_local_debug.html")
        
        # 关闭浏览器
        driver.quit()
        
        return len(active_public_links) > 0 or len(house_price_links) > 0
        
    except WebDriverException as e:
        logger.error(f"WebDriver错误: {e}")
        return False
    except Exception as e:
        logger.error(f"Selenium测试失败: {e}")
        return False

def test_data_page_selenium():
    """使用Selenium测试数据发布页面"""
    try:
        logger.info("开始使用Selenium测试数据发布页面...")
        
        # 配置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # 使用本地ChromeDriver
        chromedriver_path = r"D:\Users\xianyuchao\Downloads\chromedriver-win64\chromedriver.exe"
        
        if not os.path.exists(chromedriver_path):
            chromedriver_path = r"C:\Python311\chromedriver.exe"
        
        service = Service(chromedriver_path)
        
        # 创建WebDriver
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # 访问数据发布页面
        data_url = "https://www.stats.gov.cn/sj/zxfb/"
        logger.info(f"正在访问数据发布页面: {data_url}")
        driver.get(data_url)
        
        # 等待页面加载
        time.sleep(5)
        
        # 查找页面中的链接
        links = driver.find_elements(By.TAG_NAME, "a")
        logger.info(f"找到 {len(links)} 个链接")
        
        # 查找房价相关的链接
        house_price_links = []
        keywords = ['70个大中城市', '商品住房', '房价', '住宅销售价格', '商品住宅']
        
        for link in links:
            try:
                text = link.text.strip()
                href = link.get_attribute('href')
                
                # 检查是否包含房价相关关键词
                is_house_price = any(keyword in text for keyword in keywords)
                
                if is_house_price and href and 't20' in href and href.endswith('.html'):
                    house_price_links.append({
                        'text': text,
                        'href': href
                    })
                    logger.info(f"找到房价相关链接: {text}")
            except Exception as e:
                continue
        
        logger.info(f"总共找到 {len(house_price_links)} 个房价相关链接")
        
        # 保存页面源码
        page_source = driver.page_source
        with open('stats_data_selenium_local_debug.html', 'w', encoding='utf-8') as f:
            f.write(page_source)
        logger.info("页面源码已保存到 stats_data_selenium_local_debug.html")
        
        # 关闭浏览器
        driver.quit()
        
        return len(house_price_links) > 0
        
    except Exception as e:
        logger.error(f"测试数据发布页面失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始Selenium本地测试...")
    
    # 测试搜索功能
    if test_selenium_local():
        logger.info("Selenium搜索功能测试成功！")
    else:
        logger.warning("Selenium搜索功能测试失败")
    
    # 测试数据发布页面
    # if test_data_page_selenium():
    #     logger.info("Selenium数据发布页面测试成功！")
    # else:
    #     logger.warning("Selenium数据发布页面测试失败")
    
    logger.info("Selenium本地测试完成！")

if __name__ == "__main__":
    main() 