#!/bin/bash

echo 'Downloading from googledrive...'
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1TDJCFZ5xtTAZcxkKFrXeStDkIzaQq-x4" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1TDJCFZ5xtTAZcxkKFrXeStDkIzaQq-x4" -o pose_est_mod_data.zip

echo 'Done'