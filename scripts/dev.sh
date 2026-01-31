#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r backend/requirements.txt
npm install
npm --prefix frontend install
npm run dev
