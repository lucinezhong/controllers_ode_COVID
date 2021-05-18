import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os, json, csv
from shapely.geometry import Point, Polygon
import pandas as pd
import geopandas as gpd
from joblib import Parallel, delayed
import random

path_to_cart = '/Users/luzhong/Downloads/cart-1.2.2'

# import mpl_toolkits
# mpl_toolkits.__path__.append('/usr/local/lib/python2.7/dist-packages/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap

# define standard US projection map once for all (Albers equal area)
themap = Basemap(llcrnrlon=-130, llcrnrlat=24, urcrnrlon=-65, urcrnrlat=50, epsg=2163)


def col_alpha_rescale(percentD, percentR, scale=0.4):
    """ Return RGBA color according to D, R vote percents, namely blue if D>R and red otherwise,
    with transparency A set by vote difference fully opaque colors at and above input parameter scale (default=0.4)"""
    red, blue = (0., 1.) if percentD > percentR else (1., 0.)
    alpha = min(max(0.2 + 0.8 * (np.abs(percentD - percentR) / (scale)), 0), 1)
    return (red, 0, blue, alpha)


def func_add_legend(fig, left=1.05, width=0.015, height=0.7, voffset=0):
    """Add red/blue bar legend to a figure, with offset position given as input"""
    cax = fig.add_axes([left, (1. - height) / 2, width, height])
    for cc in np.arange(0.8, 0.49, -0.1):
        cax.fill((0, 0, 1, 1), (cc, cc + 0.1, cc + 0.1, cc), facecolor=col_alpha_rescale((1 - cc) / 2, cc / 2))
        cax.plot((0, 1), (cc, cc), lw=1, c='w')
    for cc in np.arange(0.2, 0.51, 0.1):
        cax.fill((0, 0, 1, 1), (cc - 0.1, cc, cc, cc - 0.1), facecolor=col_alpha_rescale((1 - cc) / 2 + 0.01, cc / 2))
        cax.plot((0, 1), (cc, cc), lw=1, c='w')
    ax2 = cax.twiny()
    cax.yaxis.tick_right()
    cax.set_yticks(np.arange(0.2, 0.81, 0.1))
    # cax.set_yticklabels(map(str,range(30,1,-10)+range(0,31,10)))
    for ax in [cax, ax2]:
        ax.set_xticks([])
    cax.set_ylim([0.15, 0.85])
    cax.set_xlabel('D margin (%)')
    ax2.set_xlabel('R margin (%)')
    return fig


def millify(n, millnames=['n', 'u', 'm', '', 'k', 'M', 'B', 'T']):
    i_zero = millnames.index('')
    n = float(n)
    millidx = max(0, i_zero + min(len(millnames) - 1, int(np.floor(0 if n == 0 else np.log10(abs(n)) / 3))))
    strformat = '{:.1f}{}' if i_zero == 0 and n < 1 else '{:.0f}{}'
    return strformat.format(n / 10 ** (3 * (millidx - i_zero)), millnames[millidx])


# for n in (1.23456789 * 10**r for r in range(-9, 13, 1)):
#     print('%25.9f: %20s' % (n,millify(n)))


def cartogrid_remap(cartogrid, x1, y1):
    """ Given an nx*ny grid of (x,y) points used for the cartogram, remap the point (x1,y1) into index coordinates,
    for example grid[0][0]->[0,0], grid[-1][-1]->[nx,ny]
    """
    x0, y0 = cartogrid[0][0]
    x2, y2 = cartogrid[-1][-1]
    dx, dy = (x2 - x0) / len(cartogrid[0]), (y2 - y0) / len(cartogrid)

    return [(x1 - x0) / dx, (y1 - y0) / dy]


def interp_cartogram(cartogrid, gridx, gridy, point):
    """ Bilinear interpolation based on cartogram output """
    x1, y1 = cartogrid_remap(cartogrid, point[0], point[1])

    ix1, iy1 = int(np.floor(x1)), int(np.floor(y1))
    # nearest neighbor interpolation
    # xout, yout = gridx[ix1,iy1], gridy[ix1,iy1]
    # grids have ny+1, nx+1 shape
    (dim1, dim2) = gridx.shape
    (dim3, dim4) = gridy.shape
    #print((dim1, dim2),iy1,ix1)
    xout, yout = gridx[iy1, ix1], gridy[iy1, ix1]
    xout = np.array([ix1 + 1 - x1, x1 - ix1]).dot(
        np.array([[gridx[iy1, ix1], gridx[iy1 + 1, ix1]], [gridx[iy1, ix1 + 1], gridx[iy1 + 1, ix1 + 1]]])).dot(
        np.array([iy1 + 1 - y1, y1 - iy1]))
    yout = np.array([ix1 + 1 - x1, x1 - ix1]).dot(
        np.array([[gridy[iy1, ix1], gridy[iy1 + 1, ix1]], [gridy[iy1, ix1 + 1], gridy[iy1 + 1, ix1 + 1]]])).dot(
        np.array([iy1 + 1 - y1, y1 - iy1]))
    return [xout, yout]


def point_from_lat_lon(latlon):
    """Given a (lat,lon) pair, return a shapely.Point() object"""
    return Point(latlon[1], latlon[0])


def find_in_state(latlon):
    """Given a (lat,lon) pair, find the state that it is inside (using shapely.Polygon.contains() on all the state borders)"""
    point = point_from_lat_lon(latlon)
    state_flag = []
    for st in list(MSA_border.keys()):
        state_flag.append(Polygon(np.array(MSA_border[st]).T).contains(point))
    return np.array(list(MSA_border.keys()))[state_flag]


def run_cart(densitygrid, filename):
    """ Run Gastner-Newmann algorightm via the program 'cart', starting from an input 2D density grid
    (e.g. population density): outputs in folder where 'cart' executable is, executes it, imports
    output and returns shifted x,y grids which can be used to plot any other geographical feature.
    """
    nx, ny = len(densitygrid[0]), len(densitygrid)
    np.savetxt(os.path.join(path_to_cart, filename), densitygrid, fmt="%3.3g")
    # in jupyter, this would be:  !cd $path_to_cart; ./cart $nx $ny  filename.dat  output_filename.dat
    os.system('cd ' + path_to_cart + '; ./cart ' + str(nx) + ' ' + str(ny) + '  ' + filename + '  output_' + filename)
    cart_output = np.loadtxt(os.path.join(path_to_cart, 'output_' + filename))
    cx = cart_output.T[0].reshape(len(densitygrid) + 1, len(densitygrid[0]) + 1)
    cy = cart_output.T[1].reshape(len(densitygrid) + 1, len(densitygrid[0]) + 1)
    return cx, cy


###########population and vaccination-------------
'''
data = pd.read_csv('analysis-results/vaccination_intervention_10_28.csv')
dictionary = {}
for msa in data['MSA_cod']:
    dictionary[msa]={}
    dictionary[msa]['susceptible population']=
    dictionary[msa]['vaccination']=
    dictionary[msa]['intervention level']=

'''
###########geojson--------------------------------

us_df1 = gpd.read_file(os.path.join('input', '/Users/luzhong/Downloads/tl_2019_us_cbsa/tl_2019_us_cbsa.shp'))
us_df1 = us_df1.sort_values('NAME')
us_df1 = us_df1.reset_index(drop=True)

MSA_names = pd.unique(us_df1['NAME'])
cdf = us_df1
states_map = cdf.to_crs(epsg=2163)

MSA_border = {}
for msa, borders in us_df1[['NAME', 'geometry']].itertuples(index=False):
    state = msa.split(',')[1]
    poly = []
    if borders.geom_type == 'MultiPolygon':
        for pol in borders:
            poly.append(pol)
    else:
        poly.append(borders)
    main_pol = max(poly, key=lambda x: x.area)
    if 'AK' in state:
        MSA_border[msa] = list(np.array(
            [[0.32 * xarray[0] - 65, 0.44 * xarray[1]] for xarray in np.array(main_pol.exterior.coords.xy).T]).T)
    elif 'HI' in state:
        MSA_border[msa] = list(
            np.array([[xarray[0] + 53, xarray[1] + 7] for xarray in np.array(main_pol.exterior.coords.xy).T]).T)
    else:
        MSA_border[msa] = main_pol.exterior.coords.xy

# print(MSA_names)

dictionary = {}
for msa in MSA_names:
    dictionary[msa] = {}
    dictionary[msa]['popualtion'] = random.randrange(1000, 1000000000)

####################MSA-level======================

voter_sum = np.sum([np.log10(dictionary[msa]['popualtion'] + 1) for msa in MSA_names])
avg_density = 1609.34 ** 2 * float(voter_sum) / float(sum([Polygon(np.array(themap(*MSA_border[msa])).T).area
                                                           for msa in MSA_names]))

MSA_density = {}
for msa in MSA_names:
    MSA_density[msa] = 1609.34 ** 2 * np.log10(dictionary[msa]['popualtion'] + 1) / Polygon(
        np.array(themap(*MSA_border[msa])).T).area

cartogrid = []
for y in np.linspace(23, 51, 57):
    cartogrid.append([[x, y] for x in np.linspace(-130, -65, 131)])


print('start_grid')
densitygrid = []
count = 0
for line in cartogrid:
    print(count)
    outline = []
    for pp in line:
        st = find_in_state((pp[1], pp[0]))  # find_in_state() takes (lat,long) pair, the grid is in (long, lat)
        if len(st) == 0:
            density = avg_density
        else:
            density = MSA_density[st[0]]
        outline.append(density)
    densitygrid.append(outline)
    count += 1


print('start_cart')
cx, cy = run_cart(densitygrid, 'US_MSA.dat')

print(np.array(cartogrid).shape)
print(cx.shape,cy.shape)

morphed_border_st = {}
for st in MSA_names:
    morphed_border_st[st] = np.array(
        [interp_cartogram(cartogrid, cx, cy, [bb[0], bb[1]]) for bb in np.array(MSA_border[st]).T])

print('start_3')
figs0, axes = plt.subplots(1, 2, figsize=(12, 4))
for st in MSA_names:
    xst, yst = np.array(MSA_border[st])
    xm, ym = themap(xst, yst)
    axes[0].plot(xm, ym, lw=1, c='w')
    axes[0].fill(xm, ym, facecolor=plt.get_cmap('Greens')(np.log10(dictionary[st]['popualtion']) / 10))

    axes[1].plot(*morphed_border_st[st].T, lw=1, c='w')
    axes[1].fill(*morphed_border_st[st].T,
                 facecolor=plt.get_cmap('Greens')(np.log10(dictionary[st]['popualtion']) / 10))

for ax in axes: _ = ax.set_axis_off()
figs0.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0)

cax = figs0.add_axes([0.5, 0.1, 0.015, 0.7])
dd = 0.2
for cc in np.arange(0 - dd / 2, 1. + dd, dd):
    cax.fill((0, 0, 1, 1), (cc - dd / 2, cc + dd / 2, cc + dd / 2, cc - dd / 2), facecolor=plt.get_cmap('Greens')(cc))
cax.yaxis.tick_right()
cax.set_yticklabels(map(str, [40, 100, 250, 630, 1500, 4000, 10000]))
ax2 = cax.twiny()
cax.set_xticks([]), ax2.set_xticks([])
cax.set_ylim([-0.19, 1.19])
ax2.set_xlabel('population/mi$^2$\n');

plt.show()




