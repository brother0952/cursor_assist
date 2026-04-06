def max_match(money, coins:list[int], max_count):
    """ money 是目标金额 ，越接近目标金额越好
        coins是硬币面值，例如 [3,5] 表示有3元合5元的硬币
        max_count 是硬币最大使用次数，不能超过最大次数
    """
    # 使用动态规划方法
    # dp[i][j] 表示使用 i 个硬币能达到金额 j
    # 初始化为 -1 表示无法达到该状态
    dp = [[-1 for _ in range(money + 1)] for _ in range(max_count + 1)]
    
    # 基础情况：使用 0 个硬币，金额为 0
    for i in range(max_count + 1):
        dp[i][0] = 0
    
    # 记录最佳结果
    best_sum = 0
    best_count = 0
    
    # 填充 dp 表
    for i in range(1, max_count + 1):  # 硬币数量
        for j in range(1, money + 1):  # 目标金额
            for coin in coins:
                if j >= coin and dp[i-1][j-coin] != -1:
                    dp[i][j] = max(dp[i][j], dp[i-1][j-coin] + coin)
            
            # 更新最佳结果
            if dp[i][j] != -1 and dp[i][j] > best_sum:
                best_sum = dp[i][j]
                best_count = i
    
    # 回溯找出使用的硬币组合
    combination = []
    remaining_money = best_sum
    remaining_count = best_count
    
    while remaining_count > 0 and remaining_money > 0:
        for coin in coins:
            if (remaining_money >= coin and 
                remaining_count > 0 and 
                dp[remaining_count-1][remaining_money-coin] != -1 and
                dp[remaining_count-1][remaining_money-coin] + coin == remaining_money):
                
                combination.append(coin)
                remaining_money -= coin
                remaining_count -= 1
                break
    
    return best_sum, combination


# 更简单的回溯实现
def max_match_simple(money, coins:list[int], max_count):
    """ money 是目标金额 ，越接近目标金额越好
        coins是硬币面值，例如 [3,5] 表示有3元合5元的硬币
        max_count 是硬币最大使用次数，不能超过最大次数
    """
    best_sum = 0
    best_combination = []
    
    def backtrack(current_sum, current_combination, start_index):
        nonlocal best_sum, best_combination
        
        # 更新最佳结果
        if current_sum > best_sum:
            best_sum = current_sum
            best_combination = current_combination[:]
        
        # 终止条件
        if len(current_combination) >= max_count or current_sum >= money:
            return
        
        # 继续尝试硬币
        for i in range(start_index, len(coins)):
            coin = coins[i]
            if current_sum + coin <= money:
                current_combination.append(coin)
                backtrack(current_sum + coin, current_combination, i)
                current_combination.pop()
    
    # 排序硬币以优化搜索
    coins.sort(reverse=True)
    backtrack(0, [], 0)
    
    return best_sum, best_combination


# 测试函数
if __name__ == "__main__":
    print("测试 max_match_simple 函数:")
    # 示例：目标金额为18，硬币面值为[3, 5]，最多使用5个硬币
    result_sum, result_comb = max_match_simple(18, [3, 5], 5)
    print(f"最接近的金额: {result_sum}")
    print(f"使用的硬币组合: {result_comb}")
    
    # 示例：目标金额为11，硬币面值为[1, 3, 4]，最多使用4个硬币
    result_sum, result_comb = max_match_simple(11, [1, 3, 4], 4)
    print(f"最接近的金额: {result_sum}")
    print(f"使用的硬币组合: {result_comb}")
    
    # 示例：目标金额为22，硬币面值为[3, 7]，最多使用5个硬币
    result_sum, result_comb = max_match_simple(106, [11, 15], 15)
    print(f"最接近的金额: {result_sum}")
    print(f"使用的硬币组合: {result_comb}")