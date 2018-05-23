#!/bin/bash

REL_SCRIPT_DIR=$(dirname "$0")
INITIAL_DIR=$(pwd)
cd $REL_SCRIPT_DIR
ABS_SCRIPT_DIR=$(pwd)

if [ $# -lt 2 ]; then
    echo 1>&2 "Usage: $0 <exercise_num> <username>"
    echo 1>&2 "e.g. $0 1 s1111"
    exit 1
fi

cd exercise_$1
chmod -R a+r exercise_code
chmod a+x exercise_code/classifiers
echo "Enter the password for user $2 to upload your model files and exercise_code directory:"

rsync --delete-before -rlv -e 'ssh -x -p 58022' --exclude '*.pyc' --exclude 'output.*' --exclude '.gitignore' --exclude "__pycache__/" models/ exercise_code $2@filecremers1.informatik.tu-muenchen.de:submit/EX$1/

cd $INITIAL_DIR

