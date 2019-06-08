#!/bin/bash


echo "========================================"
echo "This script assumes that the qucochemistry package is already installed."
echo "========================================"
read -p "Press [Enter] to continue..."


export PATH="${PATH}:${HOME}/.local/bin"

if ! [ -x "$(command -v sphinx-build)" ]; then
    echo -e 'Installing sphinx for the current user \n' >&2
    python3 -m pip install --user sphinx
    python3 -m pip install --user sphinx_rtd_theme
fi

make html

echo "========================================"
echo "Now you can see the docs by opening _build/html/index.html with your favorite browser"
echo "========================================"
