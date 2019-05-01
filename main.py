#!/usr/bin/env python3
"""Sandpile Lab
============

This is the main file from which we run all the various routines in the
sandpile lab.

"""
import pathlib
import time

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot

from sandpile import SandPile




if __name__ == "__main__":
    # Make sure that the output/ directory exists, or create it otherwise.
    output_dir = pathlib.Path.cwd() / "output"
    if not output_dir.is_dir():
        output_dir.mkdir()

    # This is the main part of the code.  Since this lab has quite a few
    # distinct parts, it will be a good idea to define functions above that do
    # particular parts of the lab.

    # Example Plotting
    ########################################
    # Here is a simple example of a plotting routine.  We create a standard
    # line plot, and also a histogram.

    # First, we create an array of x values, and compute the corresponding y
    # values.
    x = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
    y = np.sin(x) / x
    # We create the new figure with 1 subplot (the default), and store the
    # Figure and Axis object in `fig` and `ax` (allowing for their properties
    # to be changed).
    fig, ax = pyplot.subplots()
    ax.plot(x, y)
    ax.set_title("Plot of sin(x) / x")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Matplotlib by default does a good job of figuring out the limits of the
    # axes; however, it can fail sometimes.  This allows you to set them
    # manually.
    ax.set_ylim([-0.5, 1.1])
    fig.savefig("output/example.pdf")
    pyplot.close(fig)

    # Now for the histogram.  We generate some random data
    data = np.random.randn(1000000)
    fig, ax = pyplot.subplots()
    ax.hist(data)
    ax.set_title("Histogram of random data")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    fig.savefig("output/example_histogram.pdf")
    pyplot.close(fig)

