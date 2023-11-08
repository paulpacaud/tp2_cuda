#!/bin/bash

RESULTS_G_MP="results_G_MP.txt"
RESULTS_G_K="results_G_K.txt"
RESULTS_T_MP="results_T_MP.txt"

> $RESULTS_G_K
> $RESULTS_G_MP
> $RESULTS_T_MP

# Backup the original main.h file
cp main.h main.h.bak

# Loop for X and Y as per your requirement
for ((Y=1; Y<=1024; Y*=2)); do  # Assuming you want to double X in each iteration for quicker traversal
    for ((X=1; X*Y<=1024; X*=2)); do  # Assuming you want to double Y in each iteration for quicker traversal

        # Output X and Y values
        echo "X: $X, Y: $Y"

        # Modify main.h with new BLOCK_SIZE_X_K1 and BLOCK_SIZE_Y_K1 values
        sed -i.bak "s/#define BLOCK_SIZE_X_K1[[:space:]]*[0-9]*/#define BLOCK_SIZE_X_K1     $X/" main.h
        sed -i.bak "s/#define BLOCK_SIZE_Y_K1[[:space:]]*[0-9]*/#define BLOCK_SIZE_Y_K1     $Y/" main.h

        # Build the project
        make

        # Run the executable
        ./MatrixProduct -gpu-k 1 > raw.txt

        cat raw.txt | grep Elapsed | awk '{ print $5 }' > temp.txt
        T_MP=$(sed '2d;3d' temp.txt)

        cat raw.txt | grep Gflops | awk '{ print $4 }' > temp.txt
        G_MP=$(sed '2d' temp.txt)
        G_K=$(sed '1d' temp.txt)

        rm temp.txt
        rm raw.txt

        echo -n $T_MP, >> $RESULTS_T_MP
        echo -n $G_MP, >> $RESULTS_G_MP
        echo -n $G_K, >> $RESULTS_G_K
        
      
    done
    echo -ne "\n" >> $RESULTS_T_MP
    echo -ne "\n" >> $RESULTS_G_MP
    echo -ne "\n" >> $RESULTS_G_K
done

# Restore the original main.h file
mv main.h.bak main.h