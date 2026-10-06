/* IEEE double kernels and byte-identical SHA-256. No numerical approximations. */
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <CommonCrypto/CommonDigest.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#ifdef __FAST_MATH__
#error "Fast math is forbidden"
#endif
_Static_assert(sizeof(double)==8 && DBL_MANT_DIG==53 && FLT_RADIX==2,"IEEE binary64 required");
int kernel_abi(void) { return 1; }

int sha_segments(size_t count,const void **data,const size_t *length,unsigned char *out) {
    CC_SHA256_CTX context;
    if (!CC_SHA256_Init(&context)) return 1;
    for (size_t i=0;i<count;i++) {
        const unsigned char *p=data[i];size_t left=length[i];
        while (left) {
            CC_LONG part=left>UINT32_MAX?UINT32_MAX:(CC_LONG)left;
            if (!CC_SHA256_Update(&context,p,part)) return 2;
            p+=part;left-=part;
        }
    }
    return CC_SHA256_Final(out,&context)?0:3;
}

int solve_lu(size_t n,size_t columns,const double *lu,const int64_t *pivots,double *b,int trans) {
    if (!n || !columns || n>INT_MAX || columns>INT_MAX || (trans!=0 && trans!=1)) return 1;
    for (size_t i=0;i<n;i++) if (pivots[i]<(int64_t)i || pivots[i]>=(int64_t)n) return 2;
    if (!trans) {
        for (size_t i=0;i<n;i++) if (pivots[i]!=(int64_t)i)
            for (size_t j=0;j<columns;j++) {
                double x=b[i+j*n];b[i+j*n]=b[pivots[i]+j*n];b[pivots[i]+j*n]=x;
            }
        cblas_dtrsm(CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasUnit,(int)n,(int)columns,1.,lu,(int)n,b,(int)n);
        cblas_dtrsm(CblasColMajor,CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit,(int)n,(int)columns,1.,lu,(int)n,b,(int)n);
    } else {
        cblas_dtrsm(CblasColMajor,CblasLeft,CblasUpper,CblasTrans,CblasNonUnit,(int)n,(int)columns,1.,lu,(int)n,b,(int)n);
        cblas_dtrsm(CblasColMajor,CblasLeft,CblasLower,CblasTrans,CblasUnit,(int)n,(int)columns,1.,lu,(int)n,b,(int)n);
        for (size_t ii=n;ii>0;ii--) {
            size_t i=ii-1;
            if (pivots[i]!=(int64_t)i) for (size_t j=0;j<columns;j++) {
                double x=b[i+j*n];b[i+j*n]=b[pivots[i]+j*n];b[pivots[i]+j*n]=x;
            }
        }
    }
    return 0;
}

int gemm_add(size_t n,size_t p,const double *a,const double *b,double *out) {
    if (!n || !p || n>INT_MAX || p>INT_MAX) return 1;
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,(int)n,(int)p,(int)n,1.,a,(int)n,b,(int)p,1.,out,(int)p);
    return 0;
}

int normalized_from_stats(size_t n,size_t k,const double *cosine,const double *sine,
                          const double *mean,const double *norm,double *out) {
    if (n<2 || k==0) return 1;
    for (size_t j=0;j<2*k;j++) if (!isfinite(mean[j]) || !(norm[j]>0.) || !isfinite(norm[j])) return 2;
    for (size_t i=0;i<n;i++) {
        for (size_t j=0;j<k;j++) out[i*2*k+j]=(cosine[i*k+j]-mean[j])/norm[j];
        for (size_t j=0;j<k;j++) out[i*2*k+k+j]=(sine[i*k+j]-mean[k+j])/norm[k+j];
    }
    return 0;
}
