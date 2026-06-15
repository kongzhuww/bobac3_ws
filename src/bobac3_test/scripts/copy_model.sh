#!/bin/bash
# Copy best.pt from Windows Downloads to WSL models directory
cp /mnt/c/Users/33583/Downloads/best.pt /home/kongzhu/bobac3_ws/src/bobac3_test/models/best.pt
echo "Copy result: $?"
ls -lh /home/kongzhu/bobac3_ws/src/bobac3_test/models/best.pt
