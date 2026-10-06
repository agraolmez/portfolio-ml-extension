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
        source=ROOT/'temporal_fast_v4/source_checks.c'
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
        signatures=dict(verify_paths=[size,ctypes.POINTER(ctypes.c_char_p),any_pointer,ctypes.c_char_p])
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
    if digest(ROOT/'temporal_fast_v4/source_checks.c')!=result['recipe']['source_sha256']:
        raise ValueError('Loaded native source changed')
    return result



class AcceptedBatchVerifier:
    """Retain first-use hashes and every later path/stat/expected-hash check."""
    def __init__(self,files):
        self.files=files
        self._lock=threading.RLock()
        library()

    def __call__(self,expected):
        if not isinstance(expected,dict) or not expected:raise ValueError('Complete accepted input inventory required')
        paths=[];stamps=[];names=[]
        for name,wanted in expected.items():
            with self._lock:
                if name not in self.files._stats:self.files.path(name)
                if self.files.consumed.get(name)!=wanted:
                    raise ValueError('Restored cache input is not in the accepted source inventory')
                stamps.append(self.files._stats[name])
            names.append(name);paths.append(os.fsencode(self.files.root/name))
        records=np.asarray(stamps,dtype=np.uint64,order='C')
        result=library().verify_paths(len(paths),(ctypes.c_char_p*len(paths))(*paths),
            ctypes.c_void_p(records.ctypes.data),os.fsencode(self.files.root))
        if result:raise ValueError('Accepted source changed during use: '+names[result-1])
