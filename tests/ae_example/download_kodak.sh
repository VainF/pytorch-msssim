#!/bin/bash

mkdir -p datasets/data/kodak

for i in {01..24..1}; do
  echo ${i}
  wget http://r0k.us/graphics/kodak/kodak/kodim${i}.png -O datasets/data/kodak/kodim${i}.png
done