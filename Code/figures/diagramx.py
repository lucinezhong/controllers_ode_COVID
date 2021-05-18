import sys
sys.path.append('/Users/lucinezhong/Documents/pythonCode/PDE-COVID')
from os import path
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import math
import copy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection=Axes3D.name)

zline = [math.pow(5, i) * math.cos(i) +i for i in np.linspace(0, 5, 1100)]
zlinex = [math.pow(2, i) for i in np.linspace(0, 5, 1100)]
maxvalue_z = np.max(zline)
zlinex=[i/ maxvalue_z for i in zlinex]
zline = [i/ maxvalue_z for i in zline]
yline = [i*2 for i in np.linspace(0.1, 1, 1100)]
xline = [i+ 0.01 for i in np.linspace(0.1, 1, 1100)]

zline_copy= [math.pow(3.5, i) for i in np.linspace(0, 5, 1100)]
zline_copy=[i/maxvalue_z for i in zline_copy]
n=1010
zlinex[400]=zline[400]
for i in range(400,600):
    zlinex[i]=zline[400]+(i-400)/1000


zlinex=[i/ maxvalue_z for i in zlinex]
'''
for i in range(400,600):
    verts = [(xline[i],yline[i],zline[i])] + [(xline[i],yline[i],zline[i]),(xline[i],yline[i],zlinex[i])] ####lower upper
    ax.add_collection3d(Poly3DCollection([verts],color='#3182bd',alpha=0.9,linewidths=0.1)) # Add a polygon instead of fill_between

for i in range(800,1000):
    if i==800:
        zlinex[i] = zline[i]
    else:
        zlinex[i] = zlinex[i-1]-0.002
    print(i, zline[i], zlinex[i])
    #verts = [(xline[i], yline[i], zline[i]-1)] + [(xline[i], yline[i], zline[i]-1),(xline[i], yline[i], zline[i])]  ####lower upper

    verts = [(xline[i],yline[i],zlinex[i])] + [(xline[i],yline[i],zlinex[i]),(xline[i],yline[i],zline[i])] ###lower upper
    ax.add_collection3d(Poly3DCollection([verts],color='#3182bd',alpha=0.9,linewidths=0.1)) # Add a polygon instead of fill_between
'''


ax.plot3D(xline[0:n], yline[0:n], zline_copy[0:n], '#365c95',linestyle=':', linewidth=2)

ax.plot3D(xline[n-1:n], yline[n-1:n], zline_copy[n-1:n], '<', color='#365c95', markersize=3)

ax.plot3D(xline[0:n], yline[0:n], zline[0:n], '#365c95', linestyle='-', linewidth=2)
ax.plot3D(xline[n-1:n], yline[n-1:n], zline[n-1:n], '<', color='#365c95', markersize=3)

# Make a 3D quiver plot
x, y, z = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
u, v, w = np.array([[2.5, +0.08, +0.08], [0.9, 1.5, 0], [-0.2, -0.2, 0.5]])
#ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.05, color="grey", linewidth=0.5)
ax.grid(False)
fig.patch.set_visible(False)
ax.axis('off')

ax.set_xlim(-0.1, 2)
ax.set_ylim(-0.1, 1.5)
ax.set_zlim(-0.1, 1.1)

# Create a dummy axes to place annotations to
ax2 = fig.add_subplot(111, frame_on=False)
ax2.axis("off")
ax2.axis([0, 1, 0, 1])


def proj(X, ax1, ax2):
    """ From a 3D point in axes ax1,
        calculate position in 2D in ax2 """
    x, y, z = X
    x2, y2, _ = proj3d.proj_transform(x, y, z, ax1.get_proj())
    return ax2.transData.inverted().transform(ax1.transData.transform((x2, y2)))


def image(ax, arr, xy):
    """ Place an image (arr) as annotation at position xy """
    here = path.abspath(path.dirname(__file__))
    img = plt.imread(path.join(here, 'sphere.png'))

    alpha = 1
    origin = 'upper'
    lum_img = img[:, :, 0]
    thresh = 1.e-9
    my_cmap = copy.copy(plt.cm.get_cmap('PuBuGn'))  # get a copy of the color map

    my_cmap.set_bad(alpha=0)  # set how the colormap handles 'bad' values
    print(my_cmap(0))
    maximo = lum_img.max()
    lum_img[lum_img <= thresh] = np.nan  # insert 'bad' values (the white)
    lum_img = maximo - lum_img
    img = OffsetImage(lum_img, zoom=0.02, cmap=my_cmap, alpha=alpha, origin=origin)

    # im = offsetbox.OffsetImage(img, zoom=0.02, alpha=alpha, origin=origin)
    img.image.axes = ax
    ab = offsetbox.AnnotationBbox(img, xy, xybox=(-0, 0),
                                  xycoords='data', boxcoords="offset points",
                                  pad=0.3, frameon=False)

    ax.add_artist(ab)


for s in zip(xline[::200], yline[::200], zline[::200]):
    x, y = proj(s, ax, ax2)
    image(ax2, np.random.rand(10, 10), [x, y])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.savefig('/Users/lucinezhong/Documents/LuZHONGResearch/20200720COVID-Controllers/diagram/diagram.png',dpi=600)
plt.close()



