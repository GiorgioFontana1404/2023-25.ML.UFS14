#!/bin/bash

echo "######### HELLO FROM inference-entrypoint.sh $(date)"

flask run --host 0.0.0.0 --port 8080