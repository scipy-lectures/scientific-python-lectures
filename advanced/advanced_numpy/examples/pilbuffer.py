"""
Exercise: using the buffer protocol
===================================

Skeleton of the code to do an exercise using the buffer protocole.
"""

import numpy as np
from PIL import Image

# Let's make a sample image, RGBA format

x = np.zeros((200, 200, 4), dtype=np.uint8)

# TODO: fill `data` with fully opaque red [255, 0, 0, 255]

# TODO: `x` is an array of bytes (8-bit integers)
#       What we need for PIL to understand this data is RGBA array,
#       where each element is a 32-bit integer, with bytes [RR,GG,BB,AA].
#       How do we convert `x` to such an array, called `data`?

data = ...

img = Image.frombuffer("RGBA", (200, 200), data)
img.save("test.png")

#
# Mini-exercise
#
# 1. Now, modify ``x`` and img.save() again. What happens?
#
