# Windows host environment setup

Run these steps directly on the physical Windows laptop -- I can't execute
anything there myself, only guide you through it.

## 1. Check what hardware you actually have first

Before deciding on an acceleration strategy, find out what's actually in this
machine -- no point planning around a GPU/NPU delegate that doesn't exist here:

- **GPU**: Device Manager -> Display adapters, or run `dxdiag` -> Display tab.
  Note the vendor (NVIDIA/AMD/Intel) -- this determines which acceleration
  path is even possible (CUDA only works on NVIDIA; DirectML works on any of
  the three).
- **NPU**: Task Manager -> Performance tab. Windows 11 24H2+ shows a dedicated
  "NPU" entry there IF the machine has one (only recent "Copilot+ PC" laptops
  do -- most corporate laptops from a couple years ago won't have one at all).
  Also check Settings -> System -> About for "Copilot+ PC" branding.

Report back what you find -- it changes which of the options below are even
worth pursuing.

## 2. Install Python

Get Python 3.11 (64-bit) from python.org. During install, check "Add
python.exe to PATH."

## 3. Install GStreamer + PyGObject via conda-forge (recommended route)

Plain `pip install PyGObject` on Windows either fails outright or installs
without a working GStreamer typelib behind it -- gi.repository.Gst needs the
actual GStreamer runtime + introspection data, not just the Python bindings.
conda-forge is the one route that reliably gives you both together as
prebuilt binaries, instead of fighting a manual GStreamer-installer +
environment-variable setup.

```
# Install Miniforge (conda-forge's minimal installer) from
# https://github.com/conda-forge/miniforge -- pick the Windows x86_64 installer

conda create -n vfi-env python=3.11
conda activate vfi-env
conda install -c conda-forge pygobject gst-python gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad
pip install -r requirements.txt
```

## 4. Verify GStreamer bindings actually work

```
python -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print('OK')"
```

## 5. Check TFLite delegate/acceleration -- don't assume, verify

Run `check_tflite_delegate.py` (in this same folder) once the environment is
up. It loads the toy U-Net model, tries the available delegates in order, and
reports which one actually loaded plus real inference timing -- so you have a
concrete number instead of guessing whether GPU/NPU is actually in use.
