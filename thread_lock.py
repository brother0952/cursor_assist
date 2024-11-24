import threading
import time

class BankAccount:
    def __init__(self):
        self.balance = 0
        # 创建一个线程锁
        self.lock = threading.Lock()
    
    def deposit(self, amount):
        # 获取锁
        with self.lock:
            # 模拟网络延迟
            time.sleep(0.1)
            # 获取当前余额
            current_balance = self.balance
            # 更新余额
            self.balance = current_balance + amount
            print(f"存款: {amount}，当前余额: {self.balance}")

def main():
    # 创建账户实例
    account = BankAccount()
    
    # 创建多个存款线程
    threads = []
    for i in range(5):
        t = threading.Thread(target=account.deposit, args=(100,))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    print(f"最终余额: {account.balance}")

if __name__ == "__main__":
    main() 