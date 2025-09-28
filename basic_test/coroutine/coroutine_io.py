import asyncio
import aiohttp
import time

async def fetch_url(session, url):
    """异步获取URL内容"""
    async with session.get(url) as response:
        return await response.text()

async def main():
    # 要访问的URL列表
    urls = [
        'http://example.com',
        'http://example.org',
        'http://example.net'
    ]
    
    # 记录开始时间
    start_time = time.time()
    
    # 创建异步HTTP会话
    async with aiohttp.ClientSession() as session:
        # 创建任务列表
        tasks = [fetch_url(session, url) for url in urls]
        
        # 并发执行所有任务
        responses = await asyncio.gather(*tasks)
        
        # 打印结果
        for url, response in zip(urls, responses):
            print(f"URL: {url}, 响应长度: {len(response)}")
    
    # 计算总耗时
    print(f"总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    # 需要先安装 aiohttp: pip install aiohttp
    asyncio.run(main()) 