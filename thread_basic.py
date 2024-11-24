import threading
import time

def worker(thread_name, sleep_time):
    """工作线程函数"""
    # 打印线程开始的信息
    print(f"{thread_name} 开始工作")
    
    # 模拟工作过程
    for i in range(3):
        print(f"{thread_name} 正在工作 - 第{i+1}次")
        time.sleep(sleep_time)
    
    # 打印线程结束的信息
    print(f"{thread_name} 工作结束")

def main():
    # 创建两个线程
    thread1 = threading.Thread(target=worker, args=("线程1", 1))
    thread2 = threading.Thread(target=worker, args=("线程2", 2))
    
    # 启动线程
    thread1.start()
    thread2.start()
    
    # 等待两个线程都结束
    thread1.join()
    thread2.join()
    
    print("所有线程已完成")

if __name__ == "__main__":
    main() 