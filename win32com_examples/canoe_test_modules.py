import win32com.client
import time

class CANoeTestEnvironment:
    def __init__(self):
        self.application = None
        self.measurement = None
        self.test_setup = None
        self.test_modules = None
    
    def connect(self):
        """连接到CANoe并初始化测试环境"""
        try:
            self.application = win32com.client.Dispatch("CANoe.Application")
            self.measurement = self.application.Measurement
            self.test_setup = self.application.Test.TestSetup
            self.test_modules = self.test_setup.TestModules
            print("成功连接到CANoe测试环境")
            return True
        except Exception as e:
            print(f"连接CANoe测试环境失败: {str(e)}")
            return False
    
    def get_test_module(self, module_name):
        """获取指定的测试模块"""
        try:
            return self.test_modules.Item(module_name)
        except Exception as e:
            print(f"获取测试模块失败: {str(e)}")
            return None
    
    def start_test_module(self, module_name):
        """启动指定的测试模块"""
        try:
            test_module = self.get_test_module(module_name)
            if test_module:
                test_module.Start()
                print(f"测试模块 {module_name} 已启动")
                return True
            return False
        except Exception as e:
            print(f"启动测试模块失败: {str(e)}")
            return False
    
    def stop_test_module(self, module_name):
        """停止指定的测试模块"""
        try:
            test_module = self.get_test_module(module_name)
            if test_module:
                test_module.Stop()
                print(f"测试模块 {module_name} 已停止")
                return True
            return False
        except Exception as e:
            print(f"停止测试模块失败: {str(e)}")
            return False
    
    def get_test_result(self, module_name):
        """获取测试模块的结果"""
        try:
            test_module = self.get_test_module(module_name)
            if test_module:
                result = test_module.TestResult
                status = {
                    0: "未执行",
                    1: "通过",
                    2: "失败",
                    3: "错误",
                    4: "运行中"
                }.get(result, "未知状态")
                return status
            return None
        except Exception as e:
            print(f"获取测试结果失败: {str(e)}")
            return None
    
    def execute_capl_function(self, node_name, function_name, *args):
        """执行CAPL函数"""
        try:
            node = self.application.CAPL.GetNode(node_name)
            if node:
                # 将参数转换为COM对象
                com_args = []
                for arg in args:
                    com_args.append(win32com.client.VARIANT(arg))
                
                # 执行CAPL函数
                result = node.Execute(function_name, *com_args)
                print(f"CAPL函数 {function_name} 执行成功")
                return result
            return None
        except Exception as e:
            print(f"执行CAPL函数失败: {str(e)}")
            return None
    
    def wait_for_test_completion(self, module_name, timeout=60):
        """等待测试完成"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.get_test_result(module_name)
            if result not in ["运行中", "未执行"]:
                return result
            time.sleep(1)
        return "超时"

def main():
    # 创建示例
    canoe = CANoeTestEnvironment()
    
    # 连接到CANoe
    if not canoe.connect():
        return
    
    try:
        # 示例1：启动测试模块
        print("\n示例1：启动测试模块")
        test_module = "TestModule1"
        if canoe.start_test_module(test_module):
            # 等待测试完成
            result = canoe.wait_for_test_completion(test_module)
            print(f"测试结果: {result}")
        
        # 示例2：执行CAPL函数
        print("\n示例2：执行CAPL函数")
        node_name = "TestNode"
        function_name = "TestFunction"
        result = canoe.execute_capl_function(node_name, function_name, 123, "test")
        if result is not None:
            print(f"CAPL函数返回值: {result}")
        
        # 示例3：获取所有测试模块的状态
        print("\n示例3：获取测试模块状态")
        test_modules = ["Module1", "Module2", "Module3"]
        for module in test_modules:
            result = canoe.get_test_result(module)
            print(f"模块 {module} 状态: {result}")
        
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 