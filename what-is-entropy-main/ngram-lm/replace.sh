#!/bin/bash
# ',' -> ' ,'
# '.' -> ' .'
# ':' -> ' :'
# '"' -> ' "'
cat $1 | perl -pe 's/,/ ,/g' | \
	perl -pe 's/:/ :/g' | \
	perl -pe 's/\.[ ]*$/ ./g' | \
	perl -pe 's/"/ " /g' \
	> $2
