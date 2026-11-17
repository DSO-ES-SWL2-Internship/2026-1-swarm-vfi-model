# The Yocto image

The i.MX8 board runs a custom Linux image that we build with **Yocto**. This page explains what Yocto is, how we build
the image, and, as two worked examples, how to add ready-made packages (easy) and your own custom recipes (the
Wi-Fi setup).

The build files live in a separate repository,
[2026-1-swarm-imx8-docker](https://github.com/DSO-ES-SWL2-Internship/2026-1-swarm-imx8-docker). This page describes
what is in it.

## Yocto in five minutes

Yocto does not give you a Linux image. It gives you a tool that builds one, for the board to your specifications. Some important keywords:

| Word | What it is |
| --- | --- |
| **Recipe** (`.bb`) | Instructions to fetch, build and package *one* program or library |
| **Layer** (`meta-...`) | A folder of recipes and settings. NXP provides several (`meta-imx-bsp`, `meta-imx-ml`, ...). We add our own, `meta-dso` |
| **bbappend** (`.bbappend`) | A small file that **changes an existing recipe** without copying it. This is how we customise NXP's recipes |
| **Image** | The finished system: a list of packages, written as a disk image |
| **BitBake** | The tool that reads all of this and does the build |

Two files decide what your image contains:

- `local.conf`: the settings, and which packages go into the image
- `bblayers.conf`: which layers are used

## The build environment

NXP's builds are fussy about the host system, so we build **inside a Docker container**, using NXP's `imx-docker`
setup. The build targets:

- machine `imx8mpevk` (the i.MX 8M Plus EVK)
- distro `fsl-imx-xwayland`
- NXP release `6.18.20-2.0.0` ("wrynose")
- image `core-image-base`, with our additions on top

In the `imx-docker` repository:

| File | What it does |
| --- | --- |
| `mybuild.sh` | Builds the Docker image |
| `myrun.sh` | Starts the container and runs the Yocto build (`yocto-build.sh` inside it) |
| `yocto-builds/myconf/local.conf` | **Our settings and package list.** Copied into the build's `conf/` on every build |
| `yocto-builds/myconf/bblayers.conf` | Our layer list, copied the same way |
| `yocto-builds/sources/meta-dso/` | **Our custom layer** |

!!! danger "Check that `meta-dso` is in `myconf/bblayers.conf`"
    `yocto-build.sh` copies `myconf/*` over the build's `conf/` each time. If `myconf/bblayers.conf` does not list
    `meta-dso`, a fresh build **silently drops our layer**, and with it the Wi-Fi setup. Make sure it contains:

    ```text
    BBLAYERS += "${BSPDIR}/sources/meta-dso"
    ```

    The other layers in that file use `${BSPDIR}` too. Do not use an absolute path from your own machine.

## Build the image

```bash
cd imx-docker
./mybuild.sh      # once: builds the Docker image
./myrun.sh        # runs the Yocto build inside it; the first build takes hours
```

Later builds are fast, because BitBake reuses everything that did not change (its *sstate cache*).

To get a shell **inside** the build environment, for example to run `bitbake` yourself:

```bash
./docker-run.sh 6.18.20-2.0.0
source sources/base/setup-environment build_fsl-imx-xwayland
bitbake core-image-base
```

The result is in `tmp/deploy/images/imx8mpevk/`: the file you want is
`core-image-base-imx8mpevk.rootfs.wic.zst`, with a `.wic.bmap` next to it.

### Flash it

Write the image to an SD card. **Check the device name first** with `lsblk`: writing to the wrong disk destroys it.

```bash
sudo bmaptool copy core-image-base-imx8mpevk.rootfs.wic.zst /dev/sdX
```

Then set the boot switches on the board to **0011** (boot from the SD card), insert the card and power on.

## Part 1: add ready-made packages

This is the easy case, and most of what you will do. NXP and the Yocto community already have recipes for almost
everything. You only have to **say which ones you want in the image**. In `local.conf`:

```text
IMAGE_INSTALL:append = " tensorflow-lite tensorflow-lite-vx-delegate"
```

Mind the **space after the opening quote**. `append` joins the strings with no separator.

This is what we add, grouped by why:

| Purpose | Packages |
| --- | --- |
| Run the model on the NPU | `tensorflow-lite`, `tensorflow-lite-vx-delegate` |
| Python for the relay | `python3-pygobject`, `gstreamer1.0-python`, `python3-numpy`, `python3-pillow` |
| Debugging | `htop`, `perf`, `sysstat`, `gdbserver`, `vim`, `git`, `rsync` |
| Video pipeline | `gstreamer1.0`, `-plugins-base`, `-plugins-good`, `-plugins-bad`, `gstreamer1.0-libav`, `imx-gst1.0-plugin` (the VPU and GPU plugins), `gstreamer1.0-plugins-ugly-x264` |
| GPU (OpenCL) | `libopencl-imx`, `clinfo`, `imx-gpu-viv-tools` |
| Networking | `openssh` (via `EXTRA_IMAGE_FEATURES`), `net-tools`, `avahi-daemon`, `iperf3` |

**To add a package:** find its recipe name (`bitbake -s | grep <name>`), add it to `IMAGE_INSTALL:append`, run the
build, reflash.

## Part 2: a custom recipe: Wi-Fi

NXP's image has no idea how *our* board should join *our* network. That needs changes no ready-made package can
make: a kernel option, a boot setting, and some configuration files. This is a good example of customising with a
layer of your own. Everything below is in `meta-dso`.

### The pattern

Every customisation in `meta-dso` is a **bbappend** that does the same three things:

```text
FILESEXTRAPATHS:prepend := "${THISDIR}/files:"     # 1. look for files next to this recipe
SRC_URI += "file://mlan0.network"                   # 2. add our file to the recipe's sources
do_install:append() {                               # 3. after the normal install, also put it in the image
    install -D -m0644 ${UNPACKDIR}/mlan0.network ${D}${systemd_unitdir}/network/25-mlan0.network
}
FILES:${PN} += "${systemd_unitdir}/network/25-mlan0.network"    # 4. tell the packager it belongs here
```

(The last line is easy to forget. Without it the file is built but never ends up in the package.)

### What makes the Wi-Fi work

The board's Wi-Fi is an NXP 88W8997 chip on a **PCIe** card, driven by `mwifiex`. Getting it to connect on its own
at boot needs five pieces:

| Piece | Where | What it does |
| --- | --- | --- |
| **Kernel driver** | `recipes-kernel/linux/linux-imx_%.bbappend` + `files/mwifiex.cfg` | A *config fragment*: two lines, `CONFIG_MWIFIEX=m` and `CONFIG_MWIFIEX_PCIE=m`, merged into the kernel's configuration so the driver is built as a module |
| **Packages** | `local.conf` | `kernel-module-mwifiex`, `kernel-module-mwifiex-pcie`, the firmware (`linux-firmware-nxp8997-sdio`, `linux-firmware-pcie8997`), `wpa-supplicant`, and `KERNEL_MODULE_AUTOLOAD += "mwifiex_pcie"` so the driver loads at boot |
| **Firmware conflict** | `recipes-bsp/firmware-imx/firmware-nxp-wifi_%.bbappend` | Two packages provide the same firmware files. `RCONFLICTS` and `RREPLACES` tell the package manager which one wins |
| **Join the network** | `recipes-connectivity/wpa-supplicant/` | Installs `/etc/wpa_supplicant/wpa_supplicant-mlan0.conf` (mode 0600) and enables `wpa_supplicant@mlan0.service` at boot |
| **Get an address** | `recipes-core/systemd/` | A `systemd-networkd` file, `25-mlan0.network`, that asks for an address by DHCP when `mlan0` appears |

!!! warning "The Wi-Fi password"
    `wpa_supplicant-mlan0.conf` holds your network's name and **password**. Never commit a real password to a public
    repository. Keep a template with placeholders in git, and put the real values in only when you build. Anything
    already committed stays in the git history.

### Configuring for the PCIe controller

The PCIe controller is **disabled** in the board's default device tree. A separate variant, `imx8mp-evk-pcie.dtb`,
turns it on. U-Boot's default is baked into the U-Boot binary, so fixing it by hand with `setenv` and `saveenv` is
undone on every fresh flash.

The fix, in `recipes-bsp/u-boot/u-boot-imx_%.bbappend`, patches the source before it is built:

```text
do_configure:prepend() {
    sed -i 's/CONFIG_DEFAULT_FDT_FILE="imx8mp-evk.dtb"/CONFIG_DEFAULT_FDT_FILE="imx8mp-evk-pcie.dtb"/' \
        ${S}/configs/imx8mp_evk_defconfig
}
```

That is the general lesson for a bbappend: **patch the cause in the recipe**, so the fix survives every rebuild and
reflash, instead of fixing the symptom on the board.

### Smaller customisations

| Recipe | What it does |
| --- | --- |
| `base-files_%.bbappend` | Installs a `.bashrc` and `.profile` for `root`. It uses `${ROOT_HOME}` and not `/etc/skel`, because `skel` is only copied when a *new* account is created and `root` already exists |
| `nativesdk-packagegroup-sdk-host.bbappend` | Removes `nativesdk-mxsldr`, an i.MX23/28 tool we do not need. Its git server was down and broke the SDK build |

## Write your own bbappend

1. Find the recipe you want to change: `bitbake-layers show-recipes '<name>*'`.
2. In `meta-dso`, create the same path with the name followed by `_%.bbappend` (the `%` matches any version).
3. Put your files in a `files/` folder next to it, and use the pattern above.
4. Make sure the layer is in `bblayers.conf` (see the warning above).
5. Build the recipe alone first: `bitbake <recipe>`. Then build the image.
6. To see which appends apply to a recipe: `bitbake-layers show-appends`.

## Next: build programs for the board

To compile your own C++ for the board **without** rebuilding the image, build the SDK. See
[Developing on the board](develop-on-board.md#method-2-c-with-the-sdk).
