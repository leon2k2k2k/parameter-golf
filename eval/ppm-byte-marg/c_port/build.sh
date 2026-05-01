#!/bin/bash
set -e
cd "$(dirname "$0")"
cc -O3 -fPIC -shared -Wall -o libppm_byte_mixer.so ppm_byte_mixer.c -lm
echo "built libppm_byte_mixer.so"
# self-test (optional)
cc -O3 -DPPM_BYTE_MIXER_MAIN -o ppm_byte_mixer_test ppm_byte_mixer.c -lm
./ppm_byte_mixer_test
echo "self-test passed"
