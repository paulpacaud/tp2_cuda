#!/bin/bash

# Backup the original main.h file
cp main.h main.h.bak

> results.txt

# Loop for X and Y as per your requirement
for ((X=1; X<=4; X*=2)); do  # Assuming you want to double X in each iteration for quicker traversal
    for ((Y=1; Y<=X; Y*=2)); do  # Assuming you want to double Y in each iteration for quicker traversal

        # Output X and Y values
        echo "X: $X, Y: $Y" >> results.txt

        # Modify main.h with new BLOCK_SIZE_X_K1 and BLOCK_SIZE_Y_K1 values
        sed -i.bak "s/#define BLOCK_SIZE_X_K1[[:space:]]*[0-9]*/#define BLOCK_SIZE_X_K1     $X/" main.h
        sed -i.bak "s/#define BLOCK_SIZE_Y_K1[[:space:]]*[0-9]*/#define BLOCK_SIZE_Y_K1     $Y/" main.h

        # Build the project
        make

        # Run the executable
        ./MatrixProduct -gpu-k 1 >> results.txt

    done
done

# Restore the original main.h file
mv main.h.bak main.h