"""Add the missing libpython dependency to RoboStack's librospack.dylib.

librospack uses CPython symbols (_Py_NoneStruct, ...) but does not link
libpython, so every C++ ROS binary that loads it (rviz, gzserver plugins,
rospack, our own nodes) aborts on macOS with:
    dyld: symbol not found in flat namespace '__Py_NoneStruct'

Usage: python fix_rospack.py <env-lib-dir>
"""
import glob
import os
import shutil
import subprocess
import sys

import lief

lib_dir = sys.argv[1]
target = os.path.join(lib_dir, "librospack.dylib")
pylibs = sorted(glob.glob(os.path.join(lib_dir, "libpython3.*.dylib")))
if not pylibs:
    sys.exit(f"no libpython3.*.dylib in {lib_dir}")
dep = "@loader_path/" + os.path.basename(pylibs[0])

fat = lief.MachO.parse(target)
binary = fat.at(0)
if any(lib.name == dep for lib in binary.libraries):
    print(f"{target}: already patched")
    sys.exit(0)

binary.add_library(dep)
tmp = target + ".patched"
binary.write(tmp)
shutil.copymode(target, tmp)
subprocess.run(["codesign", "--force", "--sign", "-", tmp], check=True)
# Replace instead of editing in place: the file is a hardlink into pixi's cache.
os.replace(tmp, target)
print(f"{target}: added {dep}")
