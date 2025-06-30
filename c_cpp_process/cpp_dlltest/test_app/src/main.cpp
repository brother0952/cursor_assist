#include <iostream>
#include <windows.h>
#include "../include/my_dll.h"

bool is_dll_loaded()
{
    HMODULE hDll = GetModuleHandle(TEXT("libMyDLL.dll"));
    if (hDll == NULL) {
        std::cerr << "Error: Failed to load DLL" << std::endl;
        return false;
    }
    return true;
}

int main()
{
    if (!is_dll_loaded()) {
        return 1;
    }

    std::cout << "Testing DLL function..." << std::endl;
    
    int a = 5;
    int b = 7;
    int result = add_numbers(a, b);
    
    std::cout << a << " + " << b << " = " << result << std::endl;
    
    if (result == (a + b)) {
        std::cout << "Test passed!" << std::endl;
        return 0;
    } else {
        std::cerr << "Test failed!" << std::endl;
        return 1;
    }
}
