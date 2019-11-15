#!/bin/bash

echo 'Downloading from googledrive...'
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1NnBiuK7PH33NGx-HlD6vbHepZ6NS0oFS" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1NnBiuK7PH33NGx-HlD6vbHepZ6NS0oFS" -o pretrain_data.zip

echo 'Done'