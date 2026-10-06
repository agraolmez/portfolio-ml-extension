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
FLAGS=('-dynamiclib','-O3','-std=c11','-fno-fast-math','-ffp-contract=off','-framework','Accelerate')
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
        source=ROOT/'temporal_fast_v4/kernels.c'
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
        any_pointer=ctypes.c_void_p
        signatures=dict(sha_segments=[size,ctypes.POINTER(any_pointer),ctypes.POINTER(size),any_pointer],
            solve_lu=[size,size,any_pointer,any_pointer,any_pointer,ctypes.c_int],
            gemm_add=[size,size,pointer,pointer,pointer],
            normalized_from_stats=[size,size,pointer,pointer,pointer,pointer,pointer])
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
    if digest(ROOT/'temporal_fast_v4/kernels.c')!=result['recipe']['source_sha256']:
        raise ValueError('Loaded native source changed')
    return result


def contiguous(value,shape=None):
    result=np.ascontiguousarray(value,dtype=np.float64)
    if shape is not None and result.shape!=shape:raise ValueError('Native array shape mismatch')
    return result



def sha_parts(parts):
    # Keep all byte/array owners alive across the C call. Arrays already contiguous.
    buffers=[];pointers=[];lengths=[]
    for part in parts:
        if isinstance(part,bytes):
            buffer=ctypes.c_char_p(part);buffers.append((part,buffer));pointers.append(ctypes.cast(buffer,ctypes.c_void_p));lengths.append(len(part))
        else:
            a=np.ascontiguousarray(part);buffers.append(a);pointers.append(ctypes.c_void_p(a.ctypes.data));lengths.append(a.nbytes)
    output=(ctypes.c_ubyte*32)();n=len(buffers)
    if library().sha_segments(n,(ctypes.c_void_p*n)(*pointers),(ctypes.c_size_t*n)(*lengths),output):
        raise ValueError('Native SHA-256 failed')
    return bytes(output).hex()


def solve_lu(factors,rhs,trans=0):
    lu,pivots=factors;lu=np.asfortranarray(lu,dtype=np.float64);pivots=np.asarray(pivots)
    rhs=np.asarray(rhs,dtype=np.float64)
    if (lu.ndim!=2 or lu.shape[0]!=lu.shape[1] or pivots.shape!=(len(lu),)
            or pivots.dtype.kind not in 'iu' or rhs.ndim not in (1,2) or len(rhs)!=len(lu)
            or trans not in (0,1)):
        raise ValueError('Aligned LU factors, pivots and right-hand sides required')
    pivots=np.ascontiguousarray(pivots,dtype=np.int64)
    out=np.array(rhs,dtype=np.float64,order='F',copy=True)
    columns=1 if out.ndim==1 else out.shape[1]
    ptr=lambda a:ctypes.c_void_p(a.ctypes.data)
    if library().solve_lu(len(lu),columns,ptr(lu),ptr(pivots),ptr(out),int(trans)):
        raise ValueError('Invalid LU dimensions or pivot sequence')
    return out


def gemm_add(matrix,values,base):
    matrix=contiguous(matrix);values=contiguous(values)
    if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or values.ndim!=2 or len(values)!=len(matrix):
        raise ValueError('Aligned matrix multiply required')
    out=np.array(base,dtype=np.float64,order='C',copy=True)
    if out.shape!=values.shape:raise ValueError('Aligned matrix sum required')
    if library().gemm_add(len(matrix),values.shape[1],matrix,values,out):raise ValueError('Matrix dimensions out of range')
    return out


def normalized_from_stats(cosine,sine,mean,norm):
    cosine=contiguous(cosine)
    if cosine.ndim!=2 or min(cosine.shape)<1:raise ValueError('Trigonometric matrix required')
    n,k=cosine.shape;sine=contiguous(sine,(n,k));mean=contiguous(mean,(2*k,));norm=contiguous(norm,(2*k,))
    out=np.empty((n,2*k))
    if library().normalized_from_stats(n,k,cosine,sine,mean,norm,out):raise ValueError('Invalid normalization statistics')
    return out
