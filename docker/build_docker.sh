#!/usr/bin/env bash

set -e

USERNAME=maventree
IMAGE=modulation

docker build --no-cache -t ${USERNAME}/${IMAGE}:$1 .
docker push ${USERNAME}/${IMAGE}:$1
