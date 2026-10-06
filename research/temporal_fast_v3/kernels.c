/* Fuse temporary-array operations only. No fast math, parallel reductions,
   reduced precision, matrix approximation, or persistent/global scratch. */
#include <math.h>
#include <stddef.h>
#include <float.h>
#ifdef __FAST_MATH__
#error "Fast math is forbidden for these kernels"
#endif
_Static_assert(sizeof(double)==8 && DBL_MANT_DIG==53 && FLT_RADIX==2,
               "IEEE binary64 is required");

int kernel_abi(void) { return 1; }

int normalize(size_t n,size_t k,const double *cosine,const double *sine,
              double *normalized,double *norm,double *mean) {
    if (n<2 || k==0) return 1;
    size_t p=2*k;
    for (size_t j=0;j<p;j++) { mean[j]=0.;norm[j]=0.; }
    for (size_t i=0;i<n;i++) {
        for (size_t j=0;j<k;j++) mean[j]+=cosine[i*k+j];
        for (size_t j=0;j<k;j++) mean[k+j]+=sine[i*k+j];
    }
    for (size_t j=0;j<p;j++) mean[j]/=(double)n;
    for (size_t i=0;i<n;i++) {
        for (size_t j=0;j<k;j++) normalized[i*p+j]=cosine[i*k+j]-mean[j];
        for (size_t j=0;j<k;j++) normalized[i*p+k+j]=sine[i*k+j]-mean[k+j];
        for (size_t j=0;j<p;j++) {
            double x=normalized[i*p+j];norm[j]+=x*x;
        }
    }
    for (size_t j=0;j<p;j++) {
        norm[j]=sqrt(norm[j]);
        if (!(norm[j]>0.) || !isfinite(norm[j])) return 2;
    }
    for (size_t i=0;i<n;i++)
        for (size_t j=0;j<p;j++) normalized[i*p+j]/=norm[j];
    return 0;
}

int signals(size_t n,size_t p,const double *normalized,const double *vol,
            double constant,double *out) {
    for (size_t i=0;i<n;i++) {
        if (!(vol[i]>0.) || !isfinite(vol[i])) return 1;
        out[i*(p+1)]=constant/vol[i];
        for (size_t j=0;j<p;j++) out[i*(p+1)+j+1]=normalized[i*p+j]/vol[i];
    }
    return 0;
}

int angle_adjoint(size_t n,size_t k,const double *a,double a_mean,
                  const double *normalized,const double *projection,
                  const double *coefficient,const double *cosine,const double *sine,
                  double *out) {
    size_t p=2*k;
    for (size_t i=0;i<n;i++) {
        double centered=a[i]-a_mean;
        for (size_t j=0;j<k;j++) {
            double c=(centered-normalized[i*p+j]*projection[j])*coefficient[j];
            double s=(centered-normalized[i*p+k+j]*projection[k+j])*coefficient[k+j];
            out[i*k+j]=-sine[i*k+j]*c+cosine[i*k+j]*s;
        }
    }
    return 0;
}

int characteristic_adjoint(size_t n,size_t d,const double *gradient,
                           const double *scale,const double *derivative,double *out) {
    for (size_t j=0;j<d;j++) out[j]=0.;
    for (size_t i=0;i<n;i++)
        for (size_t j=0;j<d;j++) out[j]+=(gradient[i*d+j]*scale[j])*derivative[i*d+j];
    for (size_t j=0;j<d;j++) if (!isfinite(out[j])) return 1;
    return 0;
}
