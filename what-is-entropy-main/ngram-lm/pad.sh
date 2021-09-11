# Pad each line in a text file with "<s>" string at the beginning
# and "</s>" string at the end
awk '$0="<s> "$0' $1 > $2
