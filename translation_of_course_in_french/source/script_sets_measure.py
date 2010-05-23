x, y = np.indices((100, 100))
sig = np.sin(2*np.pi*x/50.)*np.sin(2*np.pi*y/50.)*(1+x*y/50.**2)**2
mask = sig > 1
labels, nb = ndimage.label(mask)
nb
areas = ndimage.sum(mask, labels, np.arange(1, labels.max()+1))
areas
maxima = ndimage.maximum(sig, labels, np.arange(1, labels.max()+1))
maxima

