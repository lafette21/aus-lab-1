#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('tkAgg')

def plot_coordinates(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract coordinates
    x = []
    y = []
    z = []
    colors = []
    flag = False
    for line in lines[:-1]:
        if "end_header\n" in line:
            flag = True
        elif flag and line.strip():  # Skip empty lines
            coordinates = line.split()
            x.append(float(coordinates[0]))
            y.append(float(coordinates[1]))
            z.append(float(coordinates[2]))
            # Append color values, normalized between 0 and 1
            r = int(coordinates[3]) / 255.0
            g = int(coordinates[4]) / 255.0
            b = int(coordinates[5]) / 255.0
            colors.append([r, g, b])

    # Create the scatter plot using x, y coordinates and corresponding colors
    plt.scatter(x, y, c=colors, marker='o')
    plt.show()

file_path = 'raw.ply'
plot_coordinates(file_path)
