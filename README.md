# BUTIful

Companion code for analyzing data collected from BUTI (Bi-Image Uniaxial Testing Instrument) device.

Use with BUTI data to obtain Stress-Stretch mechanical data.

# Installation

### Recommended Setup

Install <a href="https://anaconda.org/anaconda/spyder">Anaconda Spyder</a>

### Manual Setup

Requires ```python3 64-bit```

Developed on <a href="https://www.python.org/downloads/release/python-3119/">python v3.11</a>

#### Package Requirements

* matplotlib
* numpy
* pandas
* openpyxl
* opencv-python
* tqdm
* scikit-learn

```
pip install -r requirements.txt
```

# Usage

### Run ```butiful-graph.py```

    Instructions: 1. Verify that the Pixels-per-mm variable (PPMM) in the program settings is correct
                  2. Run the script
                  3. Select the video file of the final loading
                  4. Create a region of interest around the top view of the ring for the first image
                      4a. Press enter to submit
                      4b. Verify that both (top and side) views of the ring are visible in the windows
                      4c. Press enter to submit, c to redo the ROI
                      4d. Repeat for the final image
                 5. Select the Excel file containing the corresponding force data
                 6. Select an output folder for the results

Video and Excel data must be for the last pull cycle only.

    Outputs: [GRAPH] Length
             [GRAPH] Width
             [GRAPH] Thickness
             [GRAPH] Cross Sectional Area (W*T)
             [GRAPH] Stress-Stretch with low- and high-stiffness regions
             [EXCEL] Stress, Stretch, Length, Width, Thickness, Force
             [EXCEL] Low Stiffness Linear Fitting
                     High Stiffness Linear Fitting
                     Low-High Intersection Point
                     Local slope (E) and Stress around Stretch = 1.2
                     Maximum stress
             [VIDEO] Original Video vs Threshold side by side

# Features

* Import video and Excel data
* Auto-threshold samples from video
    * Goal seek minimum standard deviation of the gradient
* Calculation of the sample length and width from top view
  * Thickness obtained from the bottom view
  * Length is the maximum number of binarized pixels in the x direction
  * Width/thickness is the mean number of binarized pixels in the y direction
* Data smoothing
* Mechanical modeling
    * Sample material stress and stretch
    * Low and High stiffness regions located
* Generates side-by-side video of the original and thresholded sample to verify results

# Contact

broemere