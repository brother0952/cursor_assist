from selenium import webdriver

chromedriver_path=r"D:\Users\xianyuchao\Downloads\chromedriver-win64\chromedriver.exe"

driver = webdriver.Chrome(chromedriver_path)

def main():
    driver.get("https://www.baidu.com")


if __name__ == "__main__":
    main()