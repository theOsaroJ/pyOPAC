#!/bin/bash
while IFS= read -r i
do
    cat "$i" >> test.xyz
    echo "" >> test.xyz  # Adds a blank line if needed
done < test.txt
