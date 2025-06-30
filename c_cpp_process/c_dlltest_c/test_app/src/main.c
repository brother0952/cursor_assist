#include <stdio.h>
#include <windows.h>
#include "../../include/my_dll.h"

typedef int (*AddNumbersFunc)(int, int);

int main()
{
    HMODULE hDll = LoadLibrary(TEXT("libMyDLL_C.dll"));
    if (hDll == NULL) {
        printf("Error: Failed to load DLL\n");
        return 1;
    }

    AddNumbersFunc add_numbers = (AddNumbersFunc)GetProcAddress(hDll, "add_numbers");
    if (add_numbers == NULL) {
        printf("Error: Failed to get function address\n");
        FreeLibrary(hDll);
        return 1;
    }

    printf("Testing DLL function...\n");
    
    int a = 5;
    int b = 7;
    int result = add_numbers(a, b);
    
    printf("%d + %d = %d\n", a, b, result);
    
    if (result == (a + b)) {
        printf("Test passed!\n");
    } else {
        printf("Test failed!\n");
    }

    FreeLibrary(hDll);
    return 0;
}
