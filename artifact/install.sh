#!/bin/bash
set -e

bazel build //libspu/mpc/cheetah/cpsi:anongbdt_main --config=linux --config=avx --config=native --compilation_mode=opt --verbose_failures
cp ./bazel-bin/libspu/mpc/cheetah/cpsi/anongbdt_main /bin/
bazel build //libspu/mpc/cheetah/cpsi:basegbdt_main --config=linux --config=avx --config=native --compilation_mode=opt --verbose_failures
cp ./bazel-bin/libspu/mpc/cheetah/cpsi/basegbdt_main /bin/
bazel build //libspu/mpc/cheetah/cpsi:benchmark --config=linux --config=avx --config=native --compilation_mode=opt --verbose_failures
cp ./bazel-bin/libspu/mpc/cheetah/cpsi/benchmark /bin/
bazel build //libspu/mpc/cheetah/cpsi:anongbdt_inference --config=linux --config=avx --config=native --compilation_mode=opt --verbose_failures
cp ./bazel-bin/libspu/mpc/cheetah/cpsi/anongbdt_inference /bin/
