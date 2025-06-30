#ifndef MY_DLL_H
#define MY_DLL_H

#ifdef _WIN32
    #ifdef MYDLL_EXPORTS
        #define MYDLL_API __declspec(dllexport)
    #else
        #define MYDLL_API __declspec(dllimport)
    #endif
#else
    #define MYDLL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

MYDLL_API int add_numbers(int a, int b);

#ifdef __cplusplus
}
#endif

#endif // MY_DLL_H
