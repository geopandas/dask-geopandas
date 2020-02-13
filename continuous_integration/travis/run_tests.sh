#!/usr/bin/env bash

set -e

if [[ $COVERAGE == 'true' ]]; then
    echo "coverage run `which py.test`"
    coverage run `which py.test`
else
    echo "py.test tests -v"
    py.test tests -v
fi

set +e
