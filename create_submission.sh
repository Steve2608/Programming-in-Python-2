#!/usr/bin/env bash

if [[ "$1" == "--help" || $# != 4 ]]; then
  echo "Usage: $0 FILE YOUR_NAME MATR_NR EX_NR"
  exit 0
fi

TARGET="submissions"
mkdir -p $TARGET

cat - $1 <<EOF | tee "$TARGET/ex$4.py"
"""ex$4.py
Author: $2
Matr.Nr.: $3
Exercise $4
"""
EOF
