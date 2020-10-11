#!/usr/bin/python3

from __future__ import division

import pandas as pd
import numpy as np
import matplotlib.mlab as ml
import matplotlib.pyplot as pp

from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.interpolate import Rbf
from pylab import imread, imshow

# PIL pode substituir parcialmente o pylab imread (depois verificar se pode completamente)
from PIL import Image

import rospy
from std_msgs.msg import String

def surveyCallback(data):
    img_correction(data.data)
    grid_plots()
    max_plot()


def img_correction(input):
    input = input.split(':')
    img = Image.open(input[0]).rotate(90)
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []

    width, height = img.size
    x1 = 0
    y1 = 0
    x_min = width + 1
    x_max = -1
    y_min = height + 1
    y_max = -1
    for item in datas:
        if item[0] >= 250 and item[1] >= 250 and item[2] >= 250:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
        
        if item[0] not in range(200,210):
            if x_min > x1:
                x_min = x1

            if x_max < x1:
                x_max = x1
            
            if y_min > y1:
                y_min = y1
            
            if y_max < y1:
                y_max = y1
            
        x1 += 1
        if x1 == width:
            x1 = 0
            y1 += 1
        
    print(x_min,x_max,y_min,y_max)
    x_min -= 5
    y_min -= 5
    x_max += 5
    y_max += 5

    img.putdata(newData)
    img = img.crop((x_min, y_min, x_max, y_max))
    width, height = img.size
    width *= 2
    height *= 2
    img = img.resize((width, height), Image.NEAREST)
    img.save(input[0]+"hih.png", "PNG")

# inserção da imagem (o fundo a ser pintado precisa ser transparente)
    global layout 
    layout = imread(input[0]+"hih.png")
    df = pd.read_csv(input[1])
    df.drop_duplicates(subset=["Drawing X", "Drawing Y"], inplace = True)
    df['Drawing X'] = (df['Drawing X'] - x_min) * 2
    df['Drawing Y'] = (df['Drawing Y'] - y_min) * 2

    df2 = df.drop(columns=['Grid Point','Drawing X','Drawing Y'])
    global beacons
    beacons = df2.columns
    global csv_list
    csv_list = df2.values.tolist()

    df = df.transpose()
    global csv_transp
    csv_transp = df.values.tolist()

    grid_width = width
    grid_height = height

    global image_width
    image_width = width
    global image_height
    image_height = height

    global num_x
    num_x = int(image_width / 4)
    global num_y 
    num_y = int(num_x / (image_width / image_height))

    print("Resolution: %0.2f x %0.2f" % (num_x, num_y))

    global x
    x = np.linspace(0, grid_width, num_x)
    global y
    y = np.linspace(0, grid_height, num_y)

    global gx
    global gy 
    gx,gy = np.meshgrid(x, y)
    gx, gy = gx.flatten(), gy.flatten()


def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke

    if size is None:
        size = dict(size=pp.rcParams['legend.fontsize'])

    at = AnchoredText(title, loc=loc, prop=size,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)

    at.set_zorder(200)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at

def grid_plots():
    f = pp.figure()
    #pp.title("Roi")
    f.suptitle("Individual AP RSSI")
    
    # Adjust the margins and padding
    f.subplots_adjust(hspace=0.1, wspace=0.1, left=0.05, right=0.95, top=0.85,
            bottom=0.15)

    # Create a grid of subplots using the AxesGrid helper
    image_grid = AxesGrid(f, 111, nrows_ncols=(1, len(beacons)), axes_pad=0.1,
            label_mode="1", share_all=True, cbar_location="right",
            cbar_mode="single", cbar_size="3%")

    # for beacon, i in zip(s_beacons, range(len(s_beacons))):
    for i in range(len(csv_transp) - 3):
        # Hide the axis labels
        image_grid[i].xaxis.set_visible(False)
        image_grid[i].yaxis.set_visible(False)


        if interpolate:
            # Interpolate the data
            #rbf = Rbf(a['Drawing X'], a['Drawing Y'], a[beacon],
            #        function='linear')
            rbf = Rbf(csv_transp[1], csv_transp[2], csv_transp[i+3],function='linear')

            z = rbf(gx, gy)
            z = z.reshape((num_y, num_x))

            # Render the interpolated data to the plot
            image = image_grid[i].imshow(z, vmin=-85, vmax=-25, extent=(0,
                image_width, image_height, 0), cmap='RdYlBu_r', alpha=1)

            #c = image_grid[i].contourf(z, levels, alpha=0.5)
            #c = image_grid[i].contour(z, levels, linewidths=5, alpha=0.5)
        else:
            #z = ml.griddata(a['Drawing X'], a['Drawing Y'], a[beacon], x, y)
            z = ml.griddata(csv_transp[1], csv_transp[2], csv_transp[i+3], x, y)

            c = image_grid[i].contourf(x, y, z, levels, alpha=0.5)

        image_grid[i].imshow(layout, interpolation='bicubic', zorder=100)

    # Setup the data for the colorbar and its ticks
    image_grid.cbar_axes[0].colorbar(image)
    image_grid.cbar_axes[0].set_yticks(levels)

    # Add inset titles to each subplot
    for ax, im_title in zip(image_grid, beacons):
        t = add_inner_title(ax, "Beacon %s" % im_title, loc=3)

        t.patch.set_alpha(0.5)

    # pp.show()
    pp.savefig('/home/wagner/catkin_ws/src/wherebot-survey-node/simulation_files/fig-beacons.png', pad_inches=0.1, bbox_inches='tight')
    pp.clf()

def max_plot():
    # Get the maximum RSSI seen for each beacon
    # max_rssi = [max(i) for i in a[s_beacons]]
    max_rssi = [max(i) for i in csv_list]

    pp.title("Maximum RSSI seen for each beacon")

    if interpolate:
        # Interpolate the data
        #rbf = Rbf(a['Drawing X'], a['Drawing Y'], max_rssi, function='linear')
        rbf = Rbf(csv_transp[1], csv_transp[2], max_rssi, function='linear')

        z = rbf(gx, gy)
        z = z.reshape((num_y, num_x))

        # Render the interpolated data to the plot
        image = pp.imshow(z, vmin=-85, vmax=-25, extent=(0,
            image_width, image_height, 0), cmap='RdYlBu_r', alpha=1)

        #pp.contourf(z, levels, alpha=0.5)
        #pp.contour(z, levels, linewidths=5, alpha=0.5)
    else:
        #z = ml.griddata(a['Drawing X'], a['Drawing Y'], max_rssi, x, y)
        z = ml.griddata(csv_transp[1], csv_transp[2], max_rssi, x, y)

        pp.contourf(x, y, z, levels, alpha=0.5)

    pp.colorbar(image)

    pp.imshow(layout, interpolation='bicubic', zorder=100)
    # pp.show()
    pp.savefig('/home/wagner/catkin_ws/src/wherebot-survey-node/simulation_files/fig-final.png',pad_inches=0.1, bbox_inches='tight')
    pp.clf()
    state.data= "/home/wagner/catkin_ws/src/wherebot-survey-node/simulation_files/fig-beacons.png:/home/wagner/catkin_ws/src/wherebot-survey-node/simulation_files/fig-final.png"
    pub.publish(state)

rospy.init_node("heatmap")
state = String()

pub = rospy.Publisher('/survey/finish/data', String, queue_size=10)
rospy.Subscriber("/survey/heatmap/input", String, surveyCallback)

levels = [-85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25]

interpolate = True
rate = rospy.Rate(0.1)

while not rospy.is_shutdown():
    rate.sleep()