awk -F'[: \t]+' '{gsub(/^[: \t]+|[: \t]+$/, ""); print NF}'
