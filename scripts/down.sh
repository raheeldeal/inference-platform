#!/usr/bin/env bash
pkill -f "mockllm --name" || true
pkill -f "inference_platform.router" || pkill -f "uv run router" || true
echo stopped
