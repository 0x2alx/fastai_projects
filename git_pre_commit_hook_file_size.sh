#!/bin/bash

# this file should be placed at .git/hooks/pre-commit

hard_limit=$(git config hooks.filesizehardlimit)
soft_limit=$(git config hooks.filesizesoftlimit)
: ${hard_limit:=52428800}  # 50 MB
: ${soft_limit:=49408000}  # ~47 MB

status=0

bytesToHuman() {
  b=${1:-0}; d=''; s=0; S=({,K,M,G,T,P,E,Z,Y}B)
  while ((b > 1000)); do
    d="$(printf ".%02d" $((b % 1000 * 100 / 1000)))"
    b=$((b / 1000))
    let s++
  done
  echo "$b$d${S[$s]}"
}

# Iterate over the zero-delimited list of staged files.
while IFS= read -r -d '' file; do
  [ -d "$file" ] && continue 
  hash=$(git ls-files -s "$file" | cut -d ' ' -f 2)
  size=$(git cat-file -s "$hash" 2>/dev/null)

  if [ -z "$size" ] || [ "$size" -eq 0 ]; then
    echo "Warning: Unable to determine size for '$file'."
    continue
  fi

  if (( size > hard_limit )); then
    echo "Error: '$file' is $(bytesToHuman $size), which exceeds the hard size limit of $(bytesToHuman $hard_limit). Removing from commit and adding to .gitignore."
    
    # Remove file from staging area
    git rm --cached "$file"
    
    # Add file path to .gitignore if not already present
    if ! grep -q "^$(echo "$file" | sed 's/[].[^$\\*]/\\&/g')$" .gitignore; then
      echo "$file" >> .gitignore
    fi
    
    status=1
  elif (( size > soft_limit )); then
    echo "Warning: '$file' is $(bytesToHuman $size), which exceeds the soft size limit of $(bytesToHuman $soft_limit). Please double check that you intended to commit this file."
  fi
done < <(git diff -z --staged --name-only --diff-filter=d)


# exit $status 
#because file is removed from cache we dont need to stop commit anymore 

exit 0
