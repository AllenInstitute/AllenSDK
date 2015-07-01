#!/bin/bash
FILES=./z*py
for f in $FILES
    do
        echo $f
        ./$f
    done

