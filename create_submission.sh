#!/usr/bin/env bash

if [[ "$1" == "--help" ]]; then
  echo "Usage: $(basename $0) FILE YOUR_NAME MATR_NR EX_NR"
  exit 0
fi

mkdir submissions

PREFIX="\"\"\"$(basename $1)
Author: $2
Matr.Nr.: $3
Exercise $4
\"\"\"
"
PYTHON_CONTENT=$(cat $1)
CONTENT="$PREFIX$PYTHON_CONTENT"

echo "$CONTENT"
echo "$CONTENT" > "submissions/$(basename $1)"