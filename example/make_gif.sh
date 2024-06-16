#!/bin/bash

# Run this monster line to reproduce the gif shown in the repo

ffmpeg -i output/grow-images/%d.png -vf "[0]split=2[bg][fg];[bg]drawbox=c=white@1:replace=1:t=fill[bg];[bg][fg]overlay=format=auto,fps=10,scale=128:128:flags=neighbor,split[s0][s1];[s0]palettegen=reserve_transparent=0[p];[s1][p]paletteuse" -loop 0 grow.gif