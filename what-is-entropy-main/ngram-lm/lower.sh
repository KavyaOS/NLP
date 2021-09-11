#!/usr/bin/env bash
cat $1 | tr '[:upper:]' '[:lower:]' > $2
