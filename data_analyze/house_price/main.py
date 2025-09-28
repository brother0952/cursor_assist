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

import re
from save_data import dump_pickle,save_list_to_pickle

pattern = r'(\d{4})年(\d{1,2})月份70个大中城市商品住宅销售价格变动情况'


all_links=[]

def match_title(s):
    match = re.fullmatch(pattern, s)
    year = ""
    month = ""
    if match:
        year = match.group(1)
        month = match.group(2).zfill(2)
        # print(f"匹配成功: '{s}' → 年份: {year}, 月份: {month}")
        return year+month
    else:
        # print(f"不匹配: '{s}'")
        pass
        return None
    
    

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def click_title_search_option(driver):
    """点击搜索位置中的'标题'选项"""
    logger.info("正在尝试点击'标题'搜索位置选项...")
    
    # 首先检查当前是否已经是"标题"模式
    try:
        current_title_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '标题') and contains(@class, 'active') or contains(@style, 'color: red')]")
        if current_title_elements:
            logger.info("当前已经是'标题'搜索模式，无需切换")
            return True
    except Exception:
        pass
    
    # 调试：打印搜索位置相关的元素
    try:
        logger.info("=== 调试信息：搜索位置相关元素 ===")
        search_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '搜索位置') or contains(text(), '标题') or contains(text(), '全文')]")
        for i, element in enumerate(search_elements[:10]):
            try:
                text = element.text.strip()
                if text and len(text) < 50:
                    logger.info(f"搜索位置元素 {i+1}: '{text}' (标签: {element.tag_name})")
            except Exception:
                continue
        logger.info("=== 搜索位置调试信息结束 ===")
    except Exception as e:
        logger.warning(f"搜索位置调试信息获取失败: {e}")
    
    # 方法1: 直接查找"标题"文本
    try:
        title_elements = driver.find_elements(By.XPATH, "//*[text()='标题']")
        for element in title_elements:
            if element.is_displayed() and element.is_enabled():
                driver.execute_script("arguments[0].click();", element)
                logger.info("成功点击'标题'选项")
                time.sleep(2)
                return True
    except Exception as e:
        logger.warning(f"方法1失败: {e}")
    
    # 方法2: 查找包含"标题"的元素
    try:
        title_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '标题')]")
        for element in title_elements:
            if element.is_displayed() and element.is_enabled():
                driver.execute_script("arguments[0].click();", element)
                logger.info("通过文本匹配成功点击'标题'选项")
                time.sleep(2)
                return True
    except Exception as e:
        logger.warning(f"方法2失败: {e}")
    
    # 方法3: 查找搜索位置区域中的选项
    try:
        # 查找可能包含搜索位置选项的元素
        search_options = driver.find_elements(By.XPATH, "//a | //span | //button | //div[@role='button'] | //label")
        for element in search_options:
            try:
                text = element.text.strip()
                if text == '标题':
                    if element.is_displayed() and element.is_enabled():
                        driver.execute_script("arguments[0].click();", element)
                        logger.info(f"成功点击搜索位置选项: {text}")
                        time.sleep(2)
                        return True
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"方法3失败: {e}")
    
    # 方法4: 查找下拉菜单或选择器中的"标题"选项
    try:
        # 查找可能的下拉菜单或选择器
        select_elements = driver.find_elements(By.XPATH, "//select | //div[contains(@class, 'select')] | //div[contains(@class, 'dropdown')] | //div[contains(@class, 'search')]")
        for select_element in select_elements:
            try:
                options = select_element.find_elements(By.XPATH, ".//option | .//div[contains(@class, 'option')] | .//span[contains(@class, 'option')] | .//a | .//span")
                for option in options:
                    text = option.text.strip()
                    if text == '标题':
                        if option.is_displayed() and option.is_enabled():
                            driver.execute_script("arguments[0].click();", option)
                            logger.info(f"通过下拉选项成功点击: {text}")
                            time.sleep(2)
                            return True
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"方法4失败: {e}")
    
    # 方法5: 查找搜索位置标签附近的"标题"选项
    try:
        # 查找包含"搜索位置"的元素，然后在其附近查找"标题"
        search_position_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '搜索位置')]")
        for search_element in search_position_elements:
            try:
                # 查找父元素或兄弟元素中的"标题"
                parent = search_element.find_element(By.XPATH, "./..")
                title_options = parent.find_elements(By.XPATH, ".//*[contains(text(), '标题')]")
                for option in title_options:
                    if option.is_displayed() and option.is_enabled():
                        driver.execute_script("arguments[0].click();", option)
                        logger.info("在搜索位置附近成功点击'标题'选项")
                        time.sleep(2)
                        return True
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"方法5失败: {e}")
    
    # 方法6: 查找所有包含"标题"的可点击元素
    try:
        all_title_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '标题')]")
        for element in all_title_elements:
            try:
                # 检查元素是否在搜索相关区域
                parent_text = element.find_element(By.XPATH, "./..").text
                if '搜索位置' in parent_text or '搜索' in parent_text:
                    if element.is_displayed() and element.is_enabled():
                        driver.execute_script("arguments[0].click();", element)
                        logger.info("在搜索区域成功点击'标题'选项")
                        time.sleep(2)
                        return True
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"方法6失败: {e}")
    
    # 方法7: 查找搜索位置区域中的链接或按钮
    try:
        # 查找搜索位置区域中的所有可点击元素
        search_area_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'search')]//a | //div[contains(@class, 'search')]//span | //div[contains(@class, 'filter')]//a | //div[contains(@class, 'filter')]//span")
        for element in search_area_elements:
            try:
                text = element.text.strip()
                if text == '标题':
                    if element.is_displayed() and element.is_enabled():
                        driver.execute_script("arguments[0].click();", element)
                        logger.info(f"在搜索区域成功点击'标题'选项: {text}")
                        time.sleep(2)
                        return True
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"方法7失败: {e}")
    
    # 方法8: 查找页面中所有包含"标题"的链接
    try:
        title_links = driver.find_elements(By.XPATH, "//a[contains(text(), '标题')]")
        for link in title_links:
            if link.is_displayed() and link.is_enabled():
                driver.execute_script("arguments[0].click();", link)
                logger.info("成功点击'标题'链接")
                time.sleep(2)
                return True
    except Exception as e:
        logger.warning(f"方法8失败: {e}")
    
    # 方法9: 查找所有包含"标题"的span元素
    try:
        title_spans = driver.find_elements(By.XPATH, "//span[contains(text(), '标题')]")
        for span in title_spans:
            if span.is_displayed() and span.is_enabled():
                driver.execute_script("arguments[0].click();", span)
                logger.info("成功点击'标题'span元素")
                time.sleep(2)
                return True
    except Exception as e:
        logger.warning(f"方法9失败: {e}")
    
    logger.warning("所有方法都未能成功点击'标题'搜索位置选项")
    return False

def click_stats_filter_button(driver):
    """点击'统计数据'筛选按钮的专用函数"""
    logger.info("正在尝试点击'统计数据'筛选按钮...")
    
    # 调试：打印页面上所有可见的按钮和文本
    try:
        logger.info("=== 调试信息：页面上的按钮和文本 ===")
        all_elements = driver.find_elements(By.XPATH, "//button | //a | //span | //div[@role='button'] | //div[contains(@class, 'btn')] | //div[contains(@class, 'button')]")
        for i, element in enumerate(all_elements[:20]):  # 只显示前20个元素
            try:
                text = element.text.strip()
                if text and len(text) < 50:  # 只显示短文本
                    logger.info(f"元素 {i+1}: '{text}' (标签: {element.tag_name})")
            except Exception:
                continue
        logger.info("=== 调试信息结束 ===")
    except Exception as e:
        logger.warning(f"调试信息获取失败: {e}")
    
    # 方法1: 直接通过文本查找
    try:
        stats_buttons = driver.find_elements(By.XPATH, "//*[text()='统计数据']")
        for button in stats_buttons:
            if button.is_displayed() and button.is_enabled():
                driver.execute_script("arguments[0].click();", button)
                logger.info("成功点击'统计数据'按钮")
                time.sleep(2)
                return True
    except Exception as e:
        logger.warning(f"方法1失败: {e}")
    
    # 方法2: 通过部分文本匹配
    try:
        stats_buttons = driver.find_elements(By.XPATH, "//*[contains(text(), '统计数据')]")
        for button in stats_buttons:
            if button.is_displayed() and button.is_enabled():
                driver.execute_script("arguments[0].click();", button)
                logger.info("通过部分文本匹配成功点击'统计数据'按钮")
                time.sleep(2)
                return True
    except Exception as e:
        logger.warning(f"方法2失败: {e}")
    
    # 方法3: 查找筛选区域中的按钮
    try:
        # 查找所有可能的筛选按钮
        filter_elements = driver.find_elements(By.XPATH, "//button | //a | //span | //div[@role='button']")
        for element in filter_elements:
            try:
                text = element.text.strip()
                if '统计数据' in text or ('统计' in text and '数据' in text):
                    if element.is_displayed() and element.is_enabled():
                        driver.execute_script("arguments[0].click();", element)
                        logger.info(f"成功点击筛选按钮: {text}")
                        time.sleep(2)
                        return True
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"方法3失败: {e}")
    
    # 方法4: 通过CSS类名查找
    try:
        filter_buttons = driver.find_elements(By.CSS_SELECTOR, ".filter-item, .category-item, .btn, .button")
        for button in filter_buttons:
            try:
                text = button.text.strip()
                if '统计数据' in text:
                    if button.is_displayed() and button.is_enabled():
                        driver.execute_script("arguments[0].click();", button)
                        logger.info(f"通过CSS选择器成功点击: {text}")
                        time.sleep(2)
                        return True
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"方法4失败: {e}")
    
    # 方法5: 查找筛选区域中的标签或分类按钮
    try:
        # 查找可能包含"统计数据"的标签元素
        tag_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'tag')] | //span[contains(@class, 'tag')] | //div[contains(@class, 'label')] | //span[contains(@class, 'label')]")
        for element in tag_elements:
            try:
                text = element.text.strip()
                if '统计数据' in text:
                    if element.is_displayed() and element.is_enabled():
                        driver.execute_script("arguments[0].click();", element)
                        logger.info(f"通过标签元素成功点击: {text}")
                        time.sleep(2)
                        return True
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"方法5失败: {e}")
    
    # 方法6: 查找下拉菜单或选择器中的选项
    try:
        # 查找可能的下拉菜单或选择器
        select_elements = driver.find_elements(By.XPATH, "//select | //div[contains(@class, 'select')] | //div[contains(@class, 'dropdown')]")
        for select_element in select_elements:
            try:
                options = select_element.find_elements(By.XPATH, ".//option | .//div[contains(@class, 'option')] | .//span[contains(@class, 'option')]")
                for option in options:
                    text = option.text.strip()
                    if '统计数据' in text:
                        if option.is_displayed() and option.is_enabled():
                            driver.execute_script("arguments[0].click();", option)
                            logger.info(f"通过下拉选项成功点击: {text}")
                            time.sleep(2)
                            return True
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"方法6失败: {e}")
    
    # 方法7: 最后尝试，查找所有包含"统计"或"数据"的元素
    try:
        all_text_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '统计') or contains(text(), '数据')]")
        for element in all_text_elements:
            try:
                text = element.text.strip()
                if '统计' in text and '数据' in text:  # 必须同时包含"统计"和"数据"
                    if element.is_displayed() and element.is_enabled():
                        driver.execute_script("arguments[0].click();", element)
                        logger.info(f"通过文本匹配成功点击: {text}")
                        time.sleep(2)
                        return True
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"方法7失败: {e}")
    
    logger.warning("所有方法都未能成功点击'统计数据'按钮")
    return False

def selenium_search_year(year:int):
    """使用本地ChromeDriver测试Selenium"""

    global all_links

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
        
        
        # 测试访问国家统计局搜索页面
        logger.info("正在访问国家统计局搜索页面...")
        # search_url = "https://www.stats.gov.cn/search/s?qt=2023%2070%E4%B8%AA%E5%A4%A7%E4%B8%AD%E5%9F%8E%E5%B8%82%E5%95%86%E5%93%81%E4%BD%8F%E6%88%BF"
        search_url = f"https://www.stats.gov.cn/search/s?qt={year}%2070%E4%B8%AA%E5%A4%A7%E4%B8%AD%E5%9F%8E%E5%B8%82%E5%95%86%E5%93%81%E4%BD%8F&siteCode=bm36000002&tab=all&toolsStatus=1"
        driver.get(search_url)
        
        # 等待页面加载
        time.sleep(6)
        
        # 获取页面标题
        page_title = driver.title
        logger.info(f"搜索页面标题: {page_title}")
        
        # 点击"统计数据"筛选按钮
        click_stats_filter_button(driver)
        
        # 点击"标题"搜索位置选项
        click_title_search_option(driver)
        
        # 查找页面中的链接
        links = driver.find_elements(By.TAG_NAME, "a")
        logger.info(f"找到 {len(links)} 个链接")
        
        # 查找包含"主动公开"的链接
        active_public_links = []

        # https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202310/t20231019_1943727.html
        # https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900151.html
        for link in links:
            try:
                text = link.text.strip()
                href = link.get_attribute('href')
                
                # 调试：打印所有链接文本
                if text and len(text) > 10:
                    logger.info(f"链接文本: '{text}'")
                
                # 尝试匹配标题
                ret_date = match_title(text)
                if ret_date:
                    if not check_if_link_in_list(ret_date):
                        active_public_links.append({
                            'date': ret_date,
                            'text': text,
                            'href': href
                        })
                        logger.info(f"找到主动公开链接: {text}")
                    else:
                        logger.info(f"主动公开链接: {text} 已存在")
                else:
                    pass
            except Exception as e:
                logger.warning(f"处理链接时出错: {e}")
                continue
        save_list_to_pickle(active_public_links,pickle_file='link.pkl')    
        
        logger.info(f"总共找到 {len(active_public_links)} 个主动公开链接")
        
        # 查找房价相关的链接
        # house_price_links = []
        # keywords = ['70个大中城市', '商品住房', '房价', '住宅销售价格', '商品住宅']
        
        # for link in links:
        #     try:
        #         text = link.text.strip()
        #         href = link.get_attribute('href')
                
        #         # 检查是否包含房价相关关键词
        #         is_house_price = any(keyword in text for keyword in keywords)
                
        #         if is_house_price and href and 't20' in href and href.endswith('.html'):
        #             house_price_links.append({
        #                 'text': text,
        #                 'href': href
        #             })
        #             logger.info(f"找到房价相关链接: {text}")
        #     except Exception as e:
        #         continue
        
        # logger.info(f"总共找到 {len(house_price_links)} 个房价相关链接")
        
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


def get_all_links():
    global all_links
    all_links = dump_pickle("link.pkl")
    # print(all_links)


def check_if_link_in_list(date:str)->bool:
    ''' 检查链接是否已经保存过'''
    global all_links
    for i in all_links:
        if i["date"]==date:
            return True

    return False            

def main():
    """主函数"""
    get_all_links()
    
    # 测试搜索功能
    for i in range(2023,2024):
    # for i in range(2018,2026):
        if selenium_search_year(i):
            logger.info("Selenium搜索功能测试成功！")
        else:
            logger.warning("Selenium搜索功能测试失败")
    


if __name__ == "__main__":
    main() 
    pass