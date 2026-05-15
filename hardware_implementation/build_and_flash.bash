#!/bin/bash
# build CMSIS-NN
# cmake .. -DCMAKE_TOOLCHAIN_FILE=../ethos-u-core-platform/cmake/toolchain/arm-none-eabi-gcc.cmake -DTARGET_CPU=cortex-m4 -DCMSIS_NN_USE_SINGLE_ROUNDING

# GPI_ARCH_BOARD_TUDNES_DPP2COM
# GPI_ARCH_BOARD_nRF_PCA10056

for ((DEVICE_ID=1; DEVICE_ID<=16;DEVICE_ID++));
do
    echo ""
    echo ""
    echo ""
    echo "=============================="
    /opt/SEGGER/segger_embedded_studio_8.22a/bin/emBuild -rebuild -verbose -config "Debug" -D EXT_THIS_NODE_ID=$DEVICE_ID -D EXT_BOARD=GPI_ARCH_BOARD_nRF_PCA10056 firmware/firmware.emProject 
    read -p "Please attatch device $DEVICE_ID and press enter" temp
    echo "Flashing firmware for device $DEVICE_ID"
    /usr/bin/JLinkExe -commanderscript erase.jlink
    /usr/bin/JLinkExe -commanderscript flash.jlink
done