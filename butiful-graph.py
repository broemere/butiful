import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

"""
    BUTIful Graph Script

    Description: Use with BUTI data to obtain Stress-Stretch mechanical data

    Instructions: 1. Verify that the Pixels-per-mm variable (PPMM) in the program setting below is correct
                  2. Run the script
                  3. Select the video file of the final loading
                  4. Create a region of interest around the top view of the ring for the first image
                      4a. Press enter to submit
                      4b. Verify that both (top and side) views of the ring are visible in the windows
                      4c. Press enter to submit, c to redo the ROI
                      4d. Repeat for the final image
                 5. Select the Excel file containing the corresponding force data
                 6. Select an output folder for the results

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

    Authors: Eli Broemer, Dillon McClintock

    Date modified: 10/3/2024

"""


# %% ############################################## PROGRAM SETTINGS ##################################################

# PIXELS PER MM
PPMM = 15.8

# SHIFT number of pixels to shift the top window for the side view window
SHIFT_X = 15
SHIFT_Y = 270

# PAD number of vertical pixels to make side view window bigger (to allow room for ring circumference)
PAD_Y = 50

# THRESH_RANGE thresholding values to analyze in auto-thresholding
THRESH_RANGE = np.arange(15, 200)

# KERNEL recommend 7 for videos 1000px, 15 for videos 2000px
KERNEL_SIZE = 7
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))

# DPI for graphs
FIG_DPI = 300

plt.style.use('bmh')

# %% ############################################# IMAGE PROCESSING ###################################################

def popup_msg(msg):
    """Pop up message for information on what is happening and instructions"""
    popup = tk.Tk()
    popup.attributes("-topmost", True)
    popup.wm_title("Instructions")
    label = ttk.Label(popup, text=msg, font=("Verdana", 12))
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
    B1.pack()
    popup.mainloop()


def popup_filepicker():
    """Creates a popup window for selecting the file"""
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    filename = filedialog.askopenfilename()
    root.destroy()
    return filename


def popup_folderpicker():
    """Creates a popup window for selecting the folder"""
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return folder_selected


def load_video(filename):
    """Load video frames from the selected file."""
    video = cv2.VideoCapture(filename)
    frame_count = int(video.get(7))
    frames = []

    for i in tqdm(range(frame_count), desc="Loading video"):
        ret, frame = video.read()
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_gray)
        else:
            break
    video.release()
    return frames


def make_video(filepath, data, fps):
    """ filepath - where the video will be saved (save as .mp4)
        data - list of images to make into video
        fps - frames per second"""
    if len(data[0].shape) == 3:
        size = data[0].shape[::-1][1:3]
        is_color = True
    else:
        size = data[0].shape[::-1]
        is_color = False
    video = cv2.VideoWriter(str(filepath),
                            cv2.VideoWriter_fourcc(*"mp4v"),  # May need to try "xvid" or "divx"
                            fps, size, is_color)
    for img in data:
        video.write(img)
    cv2.destroyAllWindows()
    video.release()


def popup_roi(image):
    """Create window for region of interest selection"""
    cv2.destroyAllWindows()
    cv2.namedWindow("Select the Area", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select the Area", 768, 512)
    r = cv2.selectROI("Select the Area", image)
    return r


def popup_previews(image, r):
    """Show result of ROI cropping to verify"""
    cropped_image_top = image[int(r[1]):int(r[1] + r[3]),
                        int(r[0]):int(r[0] + r[2])]
    cropped_image_side = image[int(r[1] + SHIFT_Y - PAD_Y):int(r[1] + r[3] + SHIFT_Y + PAD_Y),
                         int(r[0] + SHIFT_X):int(r[0] + r[2] + SHIFT_X)]

    cv2.imshow("Cropped Image Top", cropped_image_top)
    cv2.imshow("Cropped Image Side", cropped_image_side)


def roi_handler(image):
    """Manage the ROI selection windows and key presses"""
    done = False
    r = None
    while not done:
        r = popup_roi(image)
        popup_previews(image, r)

        k = cv2.waitKey(0)
        if k == 32 or k == 13:
            done = True
            cv2.destroyAllWindows()
        if k == 99:
            continue
    return r


def roi(frames):
    """Request Region of Interest from the user, and then crop all frames"""
    r = roi_handler(frames[0])
    r_f = roi_handler(frames[-1])

    xs = np.linspace(r[0], r_f[0], len(frames)).astype(np.uint)
    widths = np.linspace(r[2], r_f[2], len(frames)).astype(np.uint)
    ys = np.linspace(r[1], r_f[1], len(frames)).astype(np.uint)
    heights = np.linspace(r[3], r_f[3], len(frames)).astype(np.uint)

    frames_roi_top = []
    frames_roi_side = []

    for i in range(len(frames)):
        frame_roi_top = frames[i][int(ys[i]):int(ys[i] + heights[i]),
                        int(xs[i]):int(xs[i] + widths[i])]
        frames_roi_top.append(frame_roi_top)
        frame_roi_side = frames[i][int(ys[i] + SHIFT_Y - PAD_Y):int(ys[i] + heights[i] + SHIFT_Y + PAD_Y),
                         int(xs[i] + SHIFT_X):int(xs[i] + widths[i] + SHIFT_X)]
        frames_roi_side.append(frame_roi_side)

    return frames_roi_top, frames_roi_side


def thresh_clean(img, th):
    """Threshold and remove noise. Returns binary (0, 1) image"""
    _, thresh1 = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, KERNEL) / 255  # Clean up noise
    return binary


def bad_thresh(binary):
    """Checks if the binary image has any border pixels (ie background) or is all black or all white"""
    if 1 in binary[0, :] or 1 in binary[-1, :] or 1 in binary[:, 0] or 1 in binary[:, -1] or np.all(binary==binary[0,0]):
        return True
    else:
        return False


def threshold_top(top_roi):
    """Goal seeks the best threshold value via the most continuous length

    The length is measured as the maximum number of binary pixels in the x direction
    Length should change linear continuously, so the best threshold value is found by the value where
        the standard deviation of the derivative of length is minimized

    Width is found as the average of the real pixels masked by the binary image as this was found to be most stable.

    """
    errors = []  # Error for each threshold value = StDev(gradient(height))
    for th in tqdm(THRESH_RANGE, desc="Thresholding top"):  # Check each thresh value
        lengths = []
        bad = False
        for img in top_roi:  # Threshold each image in the video
            binary = thresh_clean(img, th)
            horz_sum = np.sum(binary, axis=1)  # Sum rows in x direction
            length = np.max(horz_sum)  # length is the max pixel row
            bad = bad_thresh(binary)
            lengths.append(length)

        error = np.std(np.gradient(lengths))
        if bad:
            error = 10000
        errors.append(error)

    th_best = THRESH_RANGE[np.argmin(errors)]  # best thresh value is the one with min error
    print(th_best, "is the best threshold value for the top")

    lengths = []
    widths = []
    for img in top_roi:
        binary = thresh_clean(img, th_best)
        horz_sum = np.sum(binary, axis=1)  # Sum rows in x direction
        length = np.max(horz_sum)  # length is the max pixel row
        lengths.append(length)

        mask = binary * (img / 255)
        vert_sum = np.sum(mask, axis=0)  # Sum columns in y direction, use real pixels bc it's more stable
        width = np.mean(vert_sum[vert_sum > 0])  # width is the mean non-zero pixels in the column
        widths.append(width)

    return th_best, lengths, widths


def threshold_side(side_roi):
    """Goal seeks the best threshold value via the most continuous thickness"""
    errors = []  # Error for each threshold value = RMSE / slope of height
    for th in tqdm(THRESH_RANGE, desc="Thresholding side"):  # Check each thresh value
        thicknesses = []
        bad = False
        for img in side_roi:  # Threshold each image in the video
            binary = thresh_clean(img, th)
            vert_sum = np.sum(binary * (img / 255), axis=0)  # Sum columns in x direction  #CHANGED from 1 to 0
            bad = bad_thresh(binary)
            if not bad:
                thickness = np.mean(vert_sum[vert_sum > 0])  # thickness is the total non-zero pixels in the column
            else:
                thickness = np.random.rand()*1000
            thicknesses.append(thickness)
        error = np.std(np.gradient(thicknesses))
        if bad:
            error = 10000
        errors.append(error)

    th_best = THRESH_RANGE[np.argmin(errors)]  # best thresh value is the one with min error
    print(th_best, "is the best threshold value for the side")

    thicknesses = []
    for img in side_roi:
        binary = thresh_clean(img, th_best)
        vert_sum = np.sum(binary * (img / 255), axis=0)  # Sum columns in x direction  #CHANGED from 1 to 0
        thickness = np.mean(vert_sum[vert_sum > 0])  # height is the total non-zero pixels in the column
        thicknesses.append(thickness)

    return th_best, thicknesses


def display(img, gray=True):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if gray:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    plt.show()
    plt.close()


def render_threshold_video(top_roi, side_roi, th_best, th_best_side):
    """Creates a stack of images showing the top and side and threshold views"""
    x_max = np.max([img.shape[0] for img in top_roi])
    y_max = np.max([img.shape[1] for img in top_roi])

    vid_frames = []
    for img, img2 in zip(top_roi, side_roi):
        diffx = x_max - img.shape[0]
        diffy = y_max - img.shape[1]
        top_sized = cv2.copyMakeBorder(img, 0, diffx, 0, diffy, cv2.BORDER_CONSTANT)
        binary_top = (thresh_clean(top_sized, th_best) * 255).astype(np.uint8)

        side_sized = cv2.copyMakeBorder(img2, 0, diffx, 0, diffy, cv2.BORDER_CONSTANT)
        binary_side = (thresh_clean(side_sized, th_best_side) * 255).astype(np.uint8)

        img_stack = np.vstack((top_sized, binary_top, side_sized, binary_side))
        vid_frames.append(img_stack)

    display(vid_frames[0])

    return vid_frames


vid_file = popup_filepicker()
frames = load_video(vid_file)

message = """Instructions:\n 
Select area by clicking the far upper left corner at the start of the arms
and dragging so the top ring is in the box.\n
Make sure the box goes from one arm to the other side arm \n
Space or Enter: Continues the process \n C: Cancels the selection """
popup_msg(message)

top_roi, side_roi = roi(frames)

th_best, lengths, widths = threshold_top(top_roi)
th_best_side, thicknesses = threshold_side(side_roi)

cs_area = np.multiply(thicknesses, widths)

vid_frames = render_threshold_video(top_roi, side_roi, th_best, th_best_side)

length_graph = plt.figure(dpi=FIG_DPI)
plt.title(f"Length (th={th_best})")
plt.scatter(np.arange(len(lengths)), np.array(lengths) / PPMM)
plt.xlabel("Time")
plt.ylabel("Top Length [mm]")
plt.tight_layout()
plt.show()

width_graph = plt.figure(dpi=FIG_DPI)
plt.title(f"Width (th={th_best})")
plt.scatter(np.arange(len(widths)), np.array(widths) / PPMM)
plt.xlabel("Time")
plt.ylabel("Top Width [mm]")
plt.tight_layout()
plt.show()

thickness_graph = plt.figure(dpi=FIG_DPI)
plt.title(f"Thickness (th={th_best})")
plt.scatter(np.arange(len(thicknesses)), np.array(thicknesses) / PPMM)
plt.xlabel("Time")
plt.ylabel("Side Thickness [mm]")
plt.tight_layout()
plt.show()

area_graph = plt.figure(dpi=FIG_DPI)
plt.title(r'Cross-Sectional Area (Width$\times$Thickness)')
plt.scatter(np.arange(len(thicknesses)), (np.array(thicknesses) / PPMM) * (np.array(widths) / PPMM))
plt.xlabel("Time")
plt.ylabel(r"CS Area [$\mathregular{mm^{2}}$]")
plt.tight_layout()
plt.show()


# %% ######################################### MECHANICAL CALCULATIONS ################################################

def load_force_data(csv_file):
    """Read Excel file and get column labeled 'Forces'"""
    df = pd.read_excel(csv_file)
    df = df.dropna()
    force = df['Forces'].to_list()
    return force


def dynamic_weighted_average(x, y, density, scaling_factor=10.0):
    """Windowed average calculation, window size scales with point density"""
    smoothed_x = []
    smoothed_y = []

    for i in range(len(x)):
        # Determine window size based on density
        window_size = int(density[i] * scaling_factor)  # Scale density for window size
        left = max(0, i - window_size)
        right = min(len(x), i + window_size)
        window_x = x[left:right]
        window_y = y[left:right]

        smoothed_x.append(np.mean(window_x))
        smoothed_y.append(np.mean(window_y))

    return np.array(smoothed_x), np.array(smoothed_y)


def line(x, y):
    """Create a line using two x coordinates and y coordinates"""
    p1 = [x[0], y[0]]
    p2 = [x[1], y[1]]
    a = (p1[1] - p2[1])
    b = (p2[0] - p1[0])
    c = (p1[0] * p2[1] - p2[0] * p1[1])
    return a, b, -c


def intersection(L1, L2):
    """https://stackoverflow.com/a/20679579"""
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False


def scale_and_smooth_data(x, y):
    """Normalizes data between 0 and 1 and then does weighted averaging"""
    SMOOTHING_FACTOR = 0.5  # Used for weighted averaging

    scaler_x = MinMaxScaler()
    x = x.reshape(-1, 1)
    scaler_x.fit(x)
    x_scaled = scaler_x.transform(x).flatten()

    scaler_y = MinMaxScaler()
    y = y.reshape(-1, 1)
    scaler_y.fit(y)
    y_scaled = scaler_y.transform(y).flatten()

    nbrs = NearestNeighbors(radius=0.1).fit(np.column_stack((x_scaled, y_scaled)))
    densities = nbrs.radius_neighbors(np.column_stack((x_scaled, y_scaled)), return_distance=False)
    weights = [len(density) for density in densities]

    x_sm, y_sm = dynamic_weighted_average(x_scaled, y_scaled, weights, scaling_factor=SMOOTHING_FACTOR)
    x_norm = MinMaxScaler().fit_transform(x_sm.reshape(-1, 1)).flatten()
    y_norm = MinMaxScaler().fit_transform(y_sm.reshape(-1, 1)).flatten()

    x_smooth = scaler_x.inverse_transform(x_norm.reshape(-1, 1)).flatten()
    y_smooth = scaler_y.inverse_transform(y_norm.reshape(-1, 1)).flatten()

    y_smooth = y_smooth - y_smooth[0]

    return x_norm, y_norm, x_smooth, y_smooth  # Smoothed and normalized X and Y data, non-normalized X and Y


def uniformly_distribute_points(x, y):
    """Fixes uneven distribution of points on a curve
    Replaces curve with N points which are equally spaced apart
    Returns the new XY data, and a key linking the data back to the original dataset indices"""
    N_POINTS = 100  # Evenly distribute N points over the entire curve
    dists = [0]
    for j in range(len(x) - 1):
        x1 = x[j]
        x2 = x[j + 1]
        y1 = y[j]
        y2 = y[j + 1]
        d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        dists.append(d)

    step = np.sum(dists) / N_POINTS
    x_uniform = [x[0]]
    y_uniform = [y[0]]

    i_codex = []

    traversed = 0
    for i in range(len(x)):
        traversed += dists[i]
        if traversed > step:
            x_uniform.append(x[i])
            y_uniform.append(y[i])
            traversed = 0
            i_codex.append(i)
    x_uniform.append(x[-1])
    y_uniform.append(y[-1])

    return x_uniform, y_uniform, i_codex


def find_linear_region(x_uniform, y_uniform, i_codex, region):
    """Locates the part of the graph with the highest R^2. Uses encoded data (ie uniformly distributed points)."""
    MIN_POINTS = 10  # Minimum number of points for line fitting
    r2s = []
    for i in np.arange(MIN_POINTS, len(x_uniform)):
        if region == "lower":
            x_fit = x_uniform[:i]
            y_fit = y_uniform[:i]
        elif region == "upper":
            x_fit = x_uniform[-i:]
            y_fit = y_uniform[-i:]
        else:
            raise ValueError("region must be either 'lower' or 'upper'")
        coefficients = np.polyfit(x_fit, y_fit, 1)  # calculate m and b for y = mx + b
        polynomial = np.poly1d(coefficients)
        y_pred = polynomial(x_fit)
        r2s.append(r2_score(y_fit, y_pred))
    i_coded = np.argmax(r2s)
    if region == "lower":
        i_decoded = i_codex[i_coded + MIN_POINTS - 1]
    elif region == "upper":
        i_decoded = i_codex[-i_coded - MIN_POINTS]
    else:
        raise ValueError("region must be either 'lower' or 'upper'")
    return i_decoded


xlsx_file = popup_filepicker()
force = load_force_data(xlsx_file)

# Calculate Stress and Stretch
area_mm2 = np.divide(cs_area, PPMM**2)
stress_kpa = np.divide(force, area_mm2)
stretch = np.divide(lengths, lengths[0])

# Smooth, normalize, and uniformly distribute data
x_norm, y_norm, x_smooth, y_smooth = scale_and_smooth_data(stretch, stress_kpa)
x_uniform, y_uniform, i_codex = uniformly_distribute_points(x_norm, y_norm)

# Find the indices for the low and high slope regions
i_low = find_linear_region(x_uniform, y_uniform, i_codex, region="lower")
i_high = find_linear_region(x_uniform, y_uniform, i_codex, region="upper")

# Find low slope line
coefficients_low = np.polyfit(x_smooth[:i_low+1], y_smooth[:i_low+1], 1)
polynomial = np.poly1d(coefficients_low)
x_pred_low = [np.min(x_smooth), np.max(x_smooth)]
y_pred_low = polynomial(x_pred_low)

# Find high slope line
coefficients_inv = np.polyfit(y_smooth[i_high:], x_smooth[i_high:], 1)
polynomial = np.poly1d(coefficients_inv)
y_pred_high = [np.min(y_smooth), np.max(y_smooth)]
x_pred_high = polynomial(y_pred_high)
coefficients_high = np.polyfit(x_smooth[i_high:], y_smooth[i_high:], 1)

# Find intersection of slope lines
low_line = line(x_pred_low, y_pred_low)
high_line = line(x_pred_high, y_pred_high)
sol = intersection(low_line, high_line)

# Create Stress-Stretch graph with slope lines
ss_graph = plt.figure(dpi=FIG_DPI)
#plt.scatter(stretch, stress_kpa)
plt.scatter(x_smooth, y_smooth)
plt.scatter(x_smooth[i_low], y_smooth[i_low], color="red")
plt.scatter(x_smooth[i_high], y_smooth[i_high], color="red")
plt.plot(x_pred_low, y_pred_low, color="red")
plt.plot(x_pred_high, y_pred_high, color="red")
plt.scatter(sol[0], sol[1], color="orange")
#plt.title("Stress-Stretch")
plt.xlabel(r"Stretch $\lambda$ [-]")
plt.ylabel(r"Stress $\sigma$ [kPa]")
plt.tight_layout()
plt.show()

# Find data on point on interest at stretch = 1.2
i_left = np.argmin(np.abs(x_smooth-1.2))
i_right = len(x_smooth) - np.argmin(np.abs(x_smooth[::-1]-1.2))
i_120 = np.round((i_left+i_right)/2).astype(np.uint)
window_size = int(np.round(len(x_smooth)*0.02))

# Local slope and windowed average
coefficients_120 = np.polyfit(x_smooth[int(i_120-window_size):int(i_120+window_size+1)],
                              y_smooth[int(i_120-window_size):int(i_120+window_size+1)], 1)
stress_120 = np.mean(y_smooth[int(i_120-window_size):int(i_120+window_size+1)])

# Save data
output_folder = popup_folderpicker()
data_name = os.path.splitext(os.path.basename(xlsx_file))[0]

length_graph.savefig(os.path.join(output_folder, "length_" + data_name + ".png"))
width_graph.savefig(os.path.join(output_folder, "width_" + data_name + ".png"))
thickness_graph.savefig(os.path.join(output_folder, "thickness_" + data_name + ".png"))
area_graph.savefig(os.path.join(output_folder, "area_" + data_name + ".png"))
ss_graph.savefig(os.path.join(output_folder, "ss_" + data_name + ".png"))

mech_data = pd.DataFrame({'stress' : y_smooth,
                          'stretch' : x_smooth,
                          'length' : np.divide(lengths, PPMM),
                          'width' : np.divide(widths, PPMM),
                          'thickness': np.divide(thicknesses, PPMM),
                          'force': force})

mech_data.to_excel(os.path.join(output_folder, "stress_stretch_" + data_name + ".xlsx"))

parameters = pd.DataFrame({'E_low': coefficients_low[0],
                           'yint_low':coefficients_low[1],
                           'stretch_low': x_smooth[i_low],
                           'stress_low': y_smooth[i_low],
                           'E_high': coefficients_high[0],
                           'yint_high': coefficients_high[1],
                           'stretch_high': x_smooth[i_high],
                           'stress_high': y_smooth[i_high],
                           'stretch_int' : sol[0],
                           'stress_int' : sol[1],
                           'E_1.2': coefficients_120[0],
                           'stress_1.2': stress_120,
                           'stress_max' : max(y_smooth)}, index=[0])

parameters.to_excel(os.path.join(output_folder, "best_parameters_" + data_name + ".xlsx"))

make_video(os.path.join(output_folder, "thresh_" + data_name + ".mp4"), vid_frames, 30)

# Clean up windows
plt.close("all")
cv2.destroyAllWindows()
