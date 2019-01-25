#!/bin/bash

shopt -s extglob

dir="$1"

for f in "$dir"/*; do
  base="`basename "$f"`"
  newbase="`echo $base | sed -e 's:_0*:_:'`"
  mv "$f" "$dir/$newbase"
done
