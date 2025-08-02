import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, WebDriverException

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_selenium_with_manager():
    """使用webdriver-manager测试Selenium"""
    try:
        logger.info("开始测试Selenium（使用webdriver-manager）...")
        
        # 配置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # 无头模式
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # 使用webdriver-manager自动下载和管理ChromeDriver
        logger.info("正在下载ChromeDriver...")
        service = Service(ChromeDriverManager().install())
        
        # 创建WebDriver - 使用正确的API
        logger.info("正在启动Chrome浏览器...")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # 测试访问百度
        logger.info("正在访问测试页面...")
        driver.get("https://www.baidu.com")
        
        # 等待页面加载
        wait = WebDriverWait(driver, 10)
        title = wait.until(EC.presence_of_element_located((By.TAG_NAME, "title")))
        
        logger.info(f"页面标题: {title.get_attribute('textContent')}")
        logger.info("Selenium基本设置测试成功！")
        
        # 测试访问国家统计局搜索页面
        logger.info("正在访问国家统计局搜索页面...")
        search_url = "https://www.stats.gov.cn/search/s?qt=2024%2070%E4%B8%AA%E5%A4%A7%E4%B8%AD%E5%9F%8E%E5%B8%82%E5%95%86%E5%93%81%E4%BD%8F%E6%88%BF"
        driver.get(search_url)
        
        # 等待页面加载
        time.sleep(5)
        
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
                if '主动公开' in text:
                    active_public_links.append({
                        'text': text,
                        'href': link.get_attribute('href')
                    })
                    logger.info(f"找到主动公开链接: {text}")
            except Exception as e:
                continue
        
        logger.info(f"总共找到 {len(active_public_links)} 个主动公开链接")
        
        # 保存页面源码用于调试
        page_source = driver.page_source
        with open('stats_search_selenium_debug.html', 'w', encoding='utf-8') as f:
            f.write(page_source)
        logger.info("页面源码已保存到 stats_search_selenium_debug.html")
        
        # 关闭浏览器
        driver.quit()
        
        return len(active_public_links) > 0
        
    except WebDriverException as e:
        logger.error(f"WebDriver错误: {e}")
        return False
    except Exception as e:
        logger.error(f"Selenium测试失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始Selenium测试...")
    
    if test_selenium_with_manager():
        logger.info("Selenium测试成功！")
    else:
        logger.error("Selenium测试失败！")
    
    logger.info("测试完成！")

if __name__ == "__main__":
    main() 