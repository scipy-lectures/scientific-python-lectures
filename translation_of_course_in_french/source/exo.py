from scipy import ndimage

dat = imread('MV_HFV_012.jpg')
dat = dat[60:]
filtdat = ndimage.median_filter(dat, size=(7,7))

hi_dat = np.histogram(dat, bins=np.arange(256))
hi_filtdat = np.histogram(filtdat, bins=np.arange(256))

void = filtdat <= 50
sand = np.logical_and(filtdat>50, filtdat<=114)
glass = filtdat > 114

phases = void.astype(np.int) + 2*glass.astype(np.int) +\
            3*sand.astype(np.int)

sand = ndimage.binary_opening(sand, iterations=2)
