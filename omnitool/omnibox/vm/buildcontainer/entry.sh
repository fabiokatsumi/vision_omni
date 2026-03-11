#!/usr/bin/env bash
set -Eeuo pipefail

: "${BOOT_MODE:="windows"}"
: "${PLATFORM:="x64"}"
: "${VM_NET_HOST:="OmniParser"}"

APP="OmniParser Windows"
SUPPORT="https://github.com/microsoft/OmniParser"

cd /run

. start.sh      # Startup hook
. utils.sh      # Load helper functions
. reset.sh      # Initialize system
. server.sh     # Start webserver
. define.sh     # Define versions
. install.sh    # Run installation
. disk.sh       # Initialize disks
. display.sh    # Initialize graphics
. network.sh    # Initialize network
. samba.sh      # Configure samba
. boot.sh       # Configure boot
. proc.sh       # Initialize processor
. memory.sh     # Check available memory
. power.sh      # Configure shutdown
. config.sh     # Configure arguments
. finish.sh     # Finish initialization

trap - ERR

version=$(qemu-system-x86_64 --version | head -n 1 | cut -d '(' -f 1 | awk '{ print $NF }')
info "Booting ${APP}${BOOT_DESC} using QEMU v$version..."

{ qemu-system-x86_64 ${ARGS:+ $ARGS} >"$QEMU_OUT" 2>"$QEMU_LOG"; rc=$?; } || :
(( rc != 0 )) && error "$(<"$QEMU_LOG")" && exit 15

terminal
( sleep 30; boot ) &
tail -fn +0 "$QEMU_LOG" 2>/dev/null &
cat "$QEMU_TERM" 2> /dev/null | tee "$QEMU_PTY" &
wait $! || :

sleep 1 & wait $!
[ ! -f "$QEMU_END" ] && finish 0
