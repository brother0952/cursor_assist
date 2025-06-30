#ifndef MY_DLL_H
#define MY_DLL_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MYDLL_EXPORTS
#define MYDLL_API __declspec(dllexport)
#else
#define MYDLL_API __declspec(dllimport)
#endif

MYDLL_API int add_numbers(int a, int b);

#ifdef __cplusplus
}
#endif

#endif // MY_DLL_H
