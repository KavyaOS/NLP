#!/bin/bash
files="/Users/yangxu/nltk_data/corpora/treebank/raw/wsj_*"
# cat $files | sed '/^$/d' | sed '/^.START/d' > wsj.txt # sed: RE error: illegal byte sequence
cat $files | grep -v '^$' | grep -v '^.START' > wsj.txt
