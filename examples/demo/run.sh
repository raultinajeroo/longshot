#!/bin/sh
# Regenerate all demo artifacts offline from bundled data + fixtures.
set -e
cd "$(dirname "$0")/../.."
longshot demo --outdir examples/demo 2>&1 | tee examples/demo/run_output.txt
