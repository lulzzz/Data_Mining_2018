#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Program used to calculate snow accumulation and melt for a point.
Bernt Viggo Matheussen
Marchc 2018.

"""

# ---------------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------


# ---------------------------------------------------
def main(argv):
    infile = "montana.txt"

    # First check if the file exists.
    try:
        with open(infile):
            pass
    except IOError:
        print
        "Oh dear! - the file doesnt exist"
        sys.exit(0)

    # Now read in the data
    f = open(infile)

    line = f.readline()  # Read (scip) the header - one line
    #print(line)
    stps = 0

    # Model parameters. Default values are set, but is also read from textfile
    degDay = 3.0  # Degree day snowmelt factor  [mm/day*C]
    threshRainSnow = 1.0  # Threshold airtemperature seperating snow and rain [C]
    correctPrcp = 1.2  # Gauge precipitation undercatch scaling factor

    swe_t = 0.0  # Snow Water Equivalent (mm), initial conditions to be zero snow
    SWE = []  # List of all the results
    OBS = []

    # Read the data
    while line:
        line = f.readline()

        if len(line) > 1:
            cols = line.split() #column 
            date = cols[0]
            prcp = float(cols[1]) #precipation
            airt = float(cols[2]) #air temperature
            swe_obs = float(cols[3]) #SnowWaterEquivalent observations

            # This is the super simple SNOW model
            if airt < threshRainSnow:
                swe_t += prcp * correctPrcp
            else:
                # melt = degDay*airt
                # swe_t -= melt
                swe_t -= degDay * airt
                if swe_t < 0.0:
                    swe_t = 0.0

            SWE.append(swe_t)  # Save data in a list for later use
            OBS.append(swe_obs)

            stps += 1
    f.close()

    snow = np.array(SWE)
    plt.plot(snow)
    plt.plot(OBS)

    plt.show()

    print
    "THE END"

    return '0'


# ---------------------------------------------------


# ---------------------------------------------------
# MAIN STARTS HERE
if __name__ == "__main__":
    main(sys.argv)
# ---------------------------------------------------
