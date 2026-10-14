"""Adapt Triton metadata query to the DTK HIP property ABI; no compute path changes.
Triton 3.8 ships newer HIP headers than DTK's libgalaxyhip. Compile only a
host metadata query against DTK headers, and retain Triton's compiler/launcher.
Enable explicitly with H3_DTK_TRITON_COMPAT=1 before querying the target.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess


def install():
    if os.environ.get('H3_DTK_TRITON_COMPAT') != '1':
        return
    import triton
    driver=triton.runtime.driver.active
    if getattr(driver, '_h3_dtk_compatible', False):
        return
    source=Path(__file__).with_name('dtk_device_properties.cpp')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache=Path(os.environ.get('WORK_HOME','/tmp'))/'h3-z200'/'host-compat'
    cache.mkdir(parents=True,exist_ok=True)
    library=cache/f'dtk-properties-{digest}.so'
    if not library.exists():
        temporary=library.with_suffix(f'.{os.getpid()}.so')
        subprocess.run(['g++','-shared','-fPIC','-O2','-D__HIP_PLATFORM_AMD__',
            '-I/opt/dtk/include',str(source),'-L/opt/dtk/lib','-lamdhip64',
            '-Wl,-rpath,/opt/dtk/lib','-o',str(temporary)],check=True)
        temporary.replace(library)
    fields=['max_shared_mem','max_num_regs','multiprocessor_count','sm_clock_rate',
            'mem_clock_rate','mem_bus_width','warpSize','max_threads_per_sm','cooperativeLaunch']
    class Properties(ctypes.Structure):
        _fields_=[(f,ctypes.c_int) for f in fields]+[('arch',ctypes.c_char*256)]
    lib=ctypes.CDLL(str(library))
    lib.h3_dtk_properties.argtypes=[ctypes.c_int,ctypes.POINTER(Properties)]
    lib.h3_dtk_properties.restype=ctypes.c_int
    def query(device):
        props=Properties()
        status=lib.h3_dtk_properties(device,ctypes.byref(props))
        if status:raise RuntimeError(f'DTK hipGetDeviceProperties failed: {status}')
        values={f:getattr(props,f) for f in fields}
        values['arch']=props.arch.decode()
        return values
    driver.utils.get_device_properties=query
    driver._h3_dtk_compatible=True
