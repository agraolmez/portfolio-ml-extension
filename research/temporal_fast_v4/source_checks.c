/* The same canonical-path and complete file-stamp checks, batched in C. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
int kernel_abi(void) { return 1; }
int verify_paths(size_t n,const char **paths,const uint64_t *expected,const char *root) {
    size_t r=strlen(root);
    for (size_t i=0;i<n;i++) {
        char *resolved=realpath(paths[i],NULL);
        if (!resolved) return (int)i+1;
        int within=strncmp(root,resolved,r)==0 && resolved[r]=='/';
        struct stat st;
        int status=within?stat(resolved,&st):-1;
        free(resolved);
        if (status || expected[i*5]!=(uint64_t)st.st_dev || expected[i*5+1]!=(uint64_t)st.st_ino
            || expected[i*5+2]!=(uint64_t)st.st_size
            || expected[i*5+3]!=(uint64_t)st.st_mtimespec.tv_sec*1000000000ULL+(uint64_t)st.st_mtimespec.tv_nsec
            || expected[i*5+4]!=(uint64_t)st.st_ctimespec.tv_sec*1000000000ULL+(uint64_t)st.st_ctimespec.tv_nsec)
            return (int)i+1;
    }
    return 0;
}
