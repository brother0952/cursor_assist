#include "Rte_Receiver.h"

// 接收者软件组件实现
void Receiver_Run(void) {
    VehicleData_Type receivedData;
    
    // 通过RTE读取数据
    if (Rte_Read_VehicleData(&receivedData) == RTE_E_OK) {
        // 处理接收到的数据
        ProcessVehicleData(receivedData.speed, receivedData.rpm);
    }
}

static void ProcessVehicleData(uint8 speed, uint16 rpm) {
    // 实现数据处理逻辑
    if (speed > 100) {
        // 触发超速警告
        Rte_Call_SpeedWarning();
    }
} 