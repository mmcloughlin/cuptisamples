#include <stdio.h>
#include <stdlib.h>

//
// Linux: dlsym must be exported as a strong symbol
//        __attribute__((visibility("default"))) ensures this
//        Ex:
//        $ nm libinjection_1.so | grep dlsym
//        00000000000006ea T dlsym
//

#include "dlfcn.h" // dlsym, RTLD_NEXT
extern "C"
{
    extern decltype(dlsym) __libc_dlsym;
    extern decltype(dlopen) __libc_dlopen_mode;
}

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define HIDDEN
#else
#define DLLEXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))
#endif

#include <iostream>
using ::std::cerr;
using ::std::cout;
using ::std::endl;

extern "C" DLLEXPORT int InitializeInjection()
{
    static bool initialized = false;

    if (initialized == false)
    {
        initialized = true;

        cout << "Hello world from the injection library" << endl;
    }

    return 1;
}

extern "C" DLLEXPORT void * dlsym(void * handle, char const * symbol)
{
    InitializeInjection();

    typedef void * (*dlsym_fn)(void *, char const *);
    static dlsym_fn real_dlsym = NULL;
    if (real_dlsym == NULL)
    {
        // Use libc internal names to avoid recursive call
        real_dlsym = (dlsym_fn)(__libc_dlsym(__libc_dlopen_mode("libdl.so", RTLD_LAZY), "dlsym"));
    }
    if (real_dlsym == NULL)
    {
        cerr << "Error finding real dlsym symbol" << endl;
        return NULL;
    }
    return real_dlsym(handle, symbol);
}
