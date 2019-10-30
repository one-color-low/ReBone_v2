#!/bin/bash

echo 'Downloading from googledrive...'
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Yu8BMJkFTv8AR26R0yNk7KPNNUdosT42" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Yu8BMJkFTv8AR26R0yNk7KPNNUdosT42" -o pretrain_data.zip

echo 'Done'
