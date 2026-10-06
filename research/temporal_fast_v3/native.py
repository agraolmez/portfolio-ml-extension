"""Versioned local C build and checked array interfaces; no financial execution."""
import ctypes
import fcntl
import os
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import threading
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FLAGS=('-dynamiclib','-O3','-std=c11','-fno-fast-math','-ffp-contract=off')
_LOCK=threading.RLock()
_LIB=None
_IDENTITY=None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library():
    global _LIB,_IDENTITY
    with _LOCK:
        if _LIB is not None:return _LIB
        if platform.system()!='Darwin':raise RuntimeError('This verified native build requires macOS')
        compiler=subprocess.check_output(['xcrun','--find','clang'],text=True).strip()
        version=subprocess.check_output([compiler,'--version'],text=True)
        # Use this compiler installation's SDK, not xcrun's newer standalone SDK.
        sdk=str((Path(compiler).parents[2]/'SDKs/MacOSX.sdk').resolve(strict=True))
        flags=[*FLAGS,'-isysroot',sdk]
        source=ROOT/'temporal_fast_v3/kernels.c'
        recipe=dict(source_sha256=digest(source),compiler=compiler,compiler_version=version,
                    flags=flags,system=platform.platform(),machine=platform.machine(),abi=1)
        key=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
        directory=ROOT/'env/extension3_native'/key
        binary,record=directory/'kernels.dylib',directory/'build.json'
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'build.lock').open('a+') as build_lock:
            fcntl.flock(build_lock,fcntl.LOCK_EX)
            if not binary.exists() and not record.exists():
                directory.mkdir(parents=True,exist_ok=True)
                pending=directory/'kernels.pending.dylib'
                if pending.exists():raise FileExistsError('An unfinished native build requires review')
                try:
                    subprocess.run([compiler,*flags,str(source),'-o',str(pending)],check=True,capture_output=True,text=True)
                    binary_sha=digest(pending)
                    os.link(pending,binary)
                    with record.open('x') as f:json.dump(dict(recipe=recipe,binary_sha256=binary_sha),f,indent=2)
                finally:
                    pending.unlink(missing_ok=True)
            if not binary.is_file() or not record.is_file():raise ValueError('Incomplete native build')
            metadata=json.loads(record.read_text())
            if metadata['recipe']!=recipe or metadata['binary_sha256']!=digest(binary):
                raise ValueError('Native source, compiler recipe or binary differs')
        lib=ctypes.CDLL(str(binary))
        size=ctypes.c_size_t;real=ctypes.c_double
        pointer=np.ctypeslib.ndpointer(dtype=np.float64,flags=('C_CONTIGUOUS','ALIGNED'))
        lib.kernel_abi.argtypes=[];lib.kernel_abi.restype=ctypes.c_int
        if lib.kernel_abi()!=1:raise ValueError('Native ABI differs')
        signatures=dict(normalize=[size,size,pointer,pointer,pointer,pointer,pointer],
            signals=[size,size,pointer,pointer,real,pointer],
            angle_adjoint=[size,size,pointer,real,pointer,pointer,pointer,pointer,pointer,pointer],
            characteristic_adjoint=[size,size,pointer,pointer,pointer,pointer])
        for name,types in signatures.items():
            function=getattr(lib,name);function.argtypes=types;function.restype=ctypes.c_int
        _IDENTITY=dict(recipe=recipe,artifact_sha256={str(p.relative_to(ROOT)):digest(p) for p in (binary,record)})
        _LIB=lib
        return lib


def identity():
    library()
    result=json.loads(json.dumps(_IDENTITY))
    for name,expected in result['artifact_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Loaded native artifact changed')
    if digest(ROOT/'temporal_fast_v3/kernels.c')!=result['recipe']['source_sha256']:
        raise ValueError('Loaded native source changed')
    return result


def contiguous(value,shape=None):
    result=np.ascontiguousarray(value,dtype=np.float64)
    if shape is not None and result.shape!=shape:raise ValueError('Native array shape mismatch')
    return result


def normalize(cosine,sine):
    cosine=contiguous(cosine)
    if cosine.ndim!=2 or min(cosine.shape)<1:raise ValueError('Aligned matrix trigonometric inputs required')
    sine=contiguous(sine,cosine.shape);n,k=cosine.shape
    normalized=np.empty((n,2*k));norm=np.empty(2*k);mean=np.empty(2*k)
    if library().normalize(n,k,cosine,sine,normalized,norm,mean):
        raise ValueError('Zero or invalid cross-sectional Fourier-feature norm')
    return normalized,norm


def signals(normalized,vol):
    normalized=contiguous(normalized)
    if normalized.ndim!=2 or normalized.shape[0]<2:raise ValueError('Normalized feature matrix required')
    n,p=normalized.shape;vol=contiguous(vol,(n,));out=np.empty((n,p+1))
    if library().signals(n,p,normalized,vol,float(1/np.sqrt(n)),out):raise ValueError('Invalid volatility')
    return out


def angle_adjoint(a,normalized,projection,coefficient,cosine,sine):
    cosine=contiguous(cosine)
    if cosine.ndim!=2 or min(cosine.shape)<1:raise ValueError('Matrix trigonometric inputs required')
    n,k=cosine.shape;sine=contiguous(sine,(n,k));a=contiguous(a,(n,))
    normalized=contiguous(normalized,(n,2*k));projection=contiguous(projection,(2*k,))
    coefficient=contiguous(coefficient,(2*k,));out=np.empty((n,k))
    if library().angle_adjoint(n,k,a,float(a.mean()),normalized,projection,coefficient,cosine,sine,out):
        raise ValueError('Native angle adjoint failed')
    return out


def characteristic_adjoint(gradient,scale,derivative):
    gradient=contiguous(gradient)
    if gradient.ndim!=2 or min(gradient.shape)<1:raise ValueError('Matrix characteristic adjoint required')
    n,d=gradient.shape;scale=contiguous(scale,(d,));derivative=contiguous(derivative,(n,d));out=np.empty(d)
    if library().characteristic_adjoint(n,d,gradient,scale,derivative,out):raise ValueError('Nonfinite characteristic adjoint')
    return out
