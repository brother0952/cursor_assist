#ifndef RTE_RECEIVER_H
#define RTE_RECEIVER_H

#include "Rte_Type.h"

// RTE 状态定义
#define RTE_E_OK 0
#define RTE_E_NOK 1

// RTE API 声明
Std_ReturnType Rte_Read_VehicleData(VehicleData_Type* data);
void Rte_Call_SpeedWarning(void);

#endif 