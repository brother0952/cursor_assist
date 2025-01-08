import unittest
import numpy as np

class LampCurrentTest(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        # 采样间隔2ms
        self.sample_interval = 0.002  
        
    def test_current_boundary(self):
        """测试1：电流边界测试
        检查电流是否在规定范围内
        """
        # 示例数据
        current_data = [1.5, 1.8, 1.6, 1.7, 1.4]
        min_current = 1.0  # 最小允许电流
        max_current = 2.0  # 最大允许电流
        
        for current in current_data:
            with self.subTest(current=current):
                self.assertGreaterEqual(current, min_current, f"电流值 {current}A 低于最小允许值 {min_current}A")
                self.assertLessEqual(current, max_current, f"电流值 {current}A 超过最大允许值 {max_current}A")

    def test_current_descent(self):
        """测试2：电流下降特性测试
        检查电流是否按预期下降并稳定
        1. 检查是否在规定时间内完成下降
        2. 检查下降过程是否平滑
        3. 检查最终是否稳定在目标范围内
        """
        # 示例数据：每2ms一个采样点
        current_data = [2.0, 1.8, 1.6, 1.4, 1.2, 1.1, 1.05, 1.02, 1.01, 1.0]
        
        # 测试参数
        max_descent_time = 0.02  # 最大下降时间20ms
        stable_range = (0.95, 1.05)  # 稳定范围
        max_step_change = 0.3  # 最大允许的相邻点变化量
        
        # 计算实际下降时间
        descent_time = len(current_data) * self.sample_interval
        
        # 1. 检查下降时间
        self.assertLessEqual(descent_time, max_descent_time, 
                           f"下降时间 {descent_time*1000}ms 超过允许值 {max_descent_time*1000}ms")
        
        # 2. 检查下降过程是否平滑
        for i in range(1, len(current_data)):
            change = abs(current_data[i] - current_data[i-1])
            self.assertLessEqual(change, max_step_change, 
                               f"时间点 {i*self.sample_interval*1000}ms 的电流变化 {change}A 超过允许值 {max_step_change}A")
        
        # 3. 检查最终稳定值
        final_current = current_data[-1]
        self.assertTrue(stable_range[0] <= final_current <= stable_range[1],
                       f"最终电流值 {final_current}A 不在稳定范围 {stable_range}A 内")

    def test_current_peak(self):
        """测试3：电流峰值测试
        检查特定时间段内是否达到预期的峰值范围
        """
        # 示例数据
        current_data = [1.0, 1.5, 1.8, 2.0, 1.9, 1.7, 1.5, 1.2, 1.1, 1.0]
        expected_peak_range = (1.9, 2.1)  # 预期峰值范围
        peak_time_range = (0.004, 0.008)  # 预期达到峰值的时间范围（4-8ms）
        
        # 找到峰值及其时间点
        peak_current = max(current_data)
        peak_index = current_data.index(peak_current)
        peak_time = peak_index * self.sample_interval
        
        # 检查峰值是否在预期范围内
        self.assertTrue(expected_peak_range[0] <= peak_current <= expected_peak_range[1],
                       f"峰值电流 {peak_current}A 不在预期范围 {expected_peak_range}A 内")
        
        # 检查达到峰值的时间是否在预期范围内
        self.assertTrue(peak_time_range[0] <= peak_time <= peak_time_range[1],
                       f"达到峰值的时间 {peak_time*1000}ms 不在预期范围 {peak_time_range[0]*1000}-{peak_time_range[1]*1000}ms 内")

    def test_current_stability(self):
        """测试4：电流稳定性测试
        检查稳定状态下的电流波动
        """
        # 示例数据：稳定状态的采样点
        current_data = [1.02, 1.01, 1.03, 1.02, 1.01, 1.02, 1.03, 1.02, 1.01, 1.02]
        
        # 测试参数
        allowed_deviation = 0.05  # 允许的最大偏差（相对于平均值）
        
        # 计算统计值
        mean_current = np.mean(current_data)
        std_current = np.std(current_data)
        max_deviation = max(abs(x - mean_current) for x in current_data)
        
        # 检查最大偏差
        self.assertLessEqual(max_deviation, allowed_deviation,
                           f"电流波动 {max_deviation}A 超过允许值 {allowed_deviation}A")
        
        # 检查标准差（用于评估稳定性）
        self.assertLessEqual(std_current, allowed_deviation/3,  # 3σ原则
                           f"电流标准差 {std_current}A 表明稳定性不足")

    def test_current_response_time(self):
        """测试5：电流响应时间测试
        检查从触发到电流变化的响应时间
        """
        # 示例数据
        current_data = [0.1, 0.1, 0.1, 0.5, 1.0, 1.5, 1.8, 2.0, 1.9, 1.8]
        trigger_threshold = 0.3  # 触发阈值
        max_response_time = 0.006  # 最大允许响应时间（6ms）
        
        # 找到首次超过触发阈值的时间点
        response_index = next(i for i, current in enumerate(current_data) 
                            if current > trigger_threshold)
        response_time = response_index * self.sample_interval
        
        # 检查响应时间
        self.assertLessEqual(response_time, max_response_time,
                           f"响应时间 {response_time*1000}ms 超过允许值 {max_response_time*1000}ms")

    def test_current_pattern(self):
        """测试6：电流模式测试
        检查电流变化是否符合预期模式
        """
        # 示例数据
        current_data = [0.1, 0.5, 1.0, 1.8, 2.0, 1.5, 1.2, 1.1, 1.0, 1.0]
        
        # 定义预期的模式特征
        expected_patterns = [
            ('启动阶段', lambda x: all(i < j for i, j in zip(x[:4], x[1:5]))),  # 持续上升
            ('峰值阶段', lambda x: max(x[4:6]) >= 1.9),  # 达到峰值
            ('稳定阶段', lambda x: all(0.95 <= i <= 1.05 for i in x[-3:]))  # 最后稳定
        ]
        
        # 检查每个阶段的模式
        for pattern_name, check_func in expected_patterns:
            self.assertTrue(check_func(current_data),
                          f"{pattern_name}的电流模式不符合预期")

def main():
    unittest.main()

if __name__ == '__main__':
    main() 