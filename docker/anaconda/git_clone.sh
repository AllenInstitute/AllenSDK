#!/bin/bash
echo REPO $REPO && \
echo BRANCH $BRANCH && \
git clone --depth 1 $REPO -b $BRANCH allensdk
