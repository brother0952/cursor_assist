import win32com.client
import time

class CANoeDiagnostics:
    def __init__(self):
        self.application = None
        self.measurement = None
        self.diag_modules = {}
    
    def connect(self):
        """连接到CANoe并初始化诊断环境"""
        try:
            self.application = win32com.client.Dispatch("CANoe.Application")
            self.measurement = self.application.Measurement
            print("成功连接到CANoe")
            return True
        except Exception as e:
            print(f"连接CANoe失败: {str(e)}")
            return False
    
    def initialize_diagnostic_module(self, module_name, ecu_name):
        """初始化诊断模块"""
        try:
            diag_module = self.application.Diagnostics.CreateModule(module_name, ecu_name)
            self.diag_modules[module_name] = diag_module
            print(f"诊断模块已初始化: {module_name} -> {ecu_name}")
            return True
        except Exception as e:
            print(f"初始化诊断模块失败: {str(e)}")
            return False
    
    def send_diagnostic_request(self, module_name, service_id, data=None, timeout=5000):
        """发送诊断请求"""
        try:
            if module_name not in self.diag_modules:
                print(f"诊断模块未找到: {module_name}")
                return None
            
            diag_module = self.diag_modules[module_name]
            request = diag_module.CreateRequest()
            
            # 设置服务ID
            request.ServiceId = service_id
            
            # 设置数据（如果有）
            if data:
                if isinstance(data, (bytes, bytearray)):
                    request.DataBytes = data
                elif isinstance(data, list):
                    request.DataBytes = bytes(data)
                else:
                    print("不支持的数据类型")
                    return None
            
            # 发送请求
            response = request.Send(timeout)
            
            # 解析响应
            if response.Valid:
                return {
                    'valid': True,
                    'service_id': response.ServiceId,
                    'data': list(response.DataBytes),
                    'positive': response.Positive
                }
            else:
                return {
                    'valid': False,
                    'error': "无效响应"
                }
            
        except Exception as e:
            print(f"发送诊断请求失败: {str(e)}")
            return None
    
    def read_dtc(self, module_name):
        """读取故障码"""
        try:
            if module_name not in self.diag_modules:
                print(f"诊断模块未找到: {module_name}")
                return None
            
            diag_module = self.diag_modules[module_name]
            dtcs = diag_module.DTCs
            
            result = []
            for dtc in dtcs:
                result.append({
                    'code': dtc.Code,
                    'description': dtc.Description,
                    'status': dtc.Status,
                    'severity': dtc.Severity
                })
            
            return result
        except Exception as e:
            print(f"读取故障码失败: {str(e)}")
            return None
    
    def clear_dtc(self, module_name):
        """清除故障码"""
        try:
            if module_name not in self.diag_modules:
                print(f"诊断模块未找到: {module_name}")
                return False
            
            diag_module = self.diag_modules[module_name]
            diag_module.ClearDTCs()
            print(f"已清除模块 {module_name} 的故障码")
            return True
        except Exception as e:
            print(f"清除故障码失败: {str(e)}")
            return False
    
    def read_data_by_identifier(self, module_name, identifier):
        """通过标识符读取数据"""
        try:
            # 使用服务ID 0x22 (Read Data By Identifier)
            data = [identifier >> 8, identifier & 0xFF]  # 拆分标识符为高字节和低字节
            response = self.send_diagnostic_request(module_name, 0x22, data)
            
            if response and response['valid'] and response['positive']:
                return response['data'][2:]  # 跳过服务ID和标识符
            return None
        except Exception as e:
            print(f"读取数据失败: {str(e)}")
            return None
    
    def write_data_by_identifier(self, module_name, identifier, data):
        """通过标识符写入数据"""
        try:
            # 使用服务ID 0x2E (Write Data By Identifier)
            request_data = [identifier >> 8, identifier & 0xFF] + data
            response = self.send_diagnostic_request(module_name, 0x2E, request_data)
            
            return response and response['valid'] and response['positive']
        except Exception as e:
            print(f"写入数据失败: {str(e)}")
            return False

def main():
    # 创建示例
    canoe = CANoeDiagnostics()
    
    # 连接到CANoe
    if not canoe.connect():
        return
    
    try:
        # 示例1：初始化诊断模块
        print("\n示例1：初始化诊断模块")
        module_name = "Engine_ECU"
        ecu_name = "Engine_Control_Unit"
        if not canoe.initialize_diagnostic_module(module_name, ecu_name):
            return
        
        # 示例2：读取VIN码（示例标识符：0xF190）
        print("\n示例2：读取VIN码")
        vin_data = canoe.read_data_by_identifier(module_name, 0xF190)
        if vin_data:
            # 假设VIN码是ASCII编码
            vin = bytes(vin_data).decode('ascii')
            print(f"VIN码: {vin}")
        
        # 示例3：读取故障码
        print("\n示例3：读取故障码")
        dtcs = canoe.read_dtc(module_name)
        if dtcs:
            for dtc in dtcs:
                print(f"故障码: {dtc['code']}")
                print(f"描述: {dtc['description']}")
                print(f"状态: {dtc['status']}")
                print(f"严重程度: {dtc['severity']}")
                print("---")
        
        # 示例4：清除故障码
        print("\n示例4：清除故障码")
        canoe.clear_dtc(module_name)
        
        # 示例5：发送自定义诊断请求
        print("\n示例5：发送自定义诊断请求")
        response = canoe.send_diagnostic_request(module_name, 0x10, [0x01])  # 诊断会话控制
        if response:
            print(f"响应有效: {response['valid']}")
            print(f"正响应: {response['positive']}")
            print(f"数据: {response['data']}")
        
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 