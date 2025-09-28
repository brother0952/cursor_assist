import asyncio

async def async_task(name, sleep_time):
    """异步任务函数"""
    print(f"{name} 开始执行")
    
    # await 用于等待异步操作完成
    await asyncio.sleep(sleep_time)
    
    print(f"{name} 执行完成")
    return f"{name} 的结果"

async def main():
    # 创建多个协程任务
    tasks = [
        async_task("任务1", 2),
        async_task("任务2", 1),
        async_task("任务3", 3)
    ]
    
    # 并发执行所有任务
    results = await asyncio.gather(*tasks)
    
    # 打印结果
    for result in results:
        print(result)

if __name__ == "__main__":
    # 运行协程主函数
    asyncio.run(main()) 