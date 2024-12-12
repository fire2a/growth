#/usr/env bash
echo "Deleting local *.csv and *.txt files"
rm -rf *csv *txt

echo "Running simulator"
python ../simulator.py -m ../tabla.csv ../config.toml > log.txt


# for every file comparto a file with the same name in the reference folder
for file in $(ls *.csv *txt); do
    diff $file results/$file
    if [ $? -eq 0 ]; then
        echo "Test passed for $file"
    else
        echo "Test failed for $file"
        echo "Aborting..."
        exit 1
    fi
done


