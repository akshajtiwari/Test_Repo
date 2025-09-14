#!/bin/bash
set -e

REPO_URL="https://github.com/akshajtiwari/basics-meta"
PKG_DIR="basics-meta"

echo ">>> Checking prerequisites..."
sudo pacman -Sy --needed --noconfirm git base-devel

if [ -d "$PKG_DIR" ]; then
    echo ">>> Repo already exists, pulling latest..."
    cd "$PKG_DIR"
    git pull
else
    echo ">>> Cloning basics-meta package repo..."
    git clone "$REPO_URL"
    cd "$PKG_DIR"
fi

echo ">>> Building and installing basics package..."
makepkg -si --noconfirm

echo ">>> Basics installation finished!"
echo ">>> Reboot your system and select KDE Plasma or Hyprland from SDDM."
