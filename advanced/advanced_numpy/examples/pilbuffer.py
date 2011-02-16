import numpy as np
import Image

# Let's make a sample image, RGBA format

x = np.zeros((200, 200, 4), dtype=np.int8)

TODO: fill `x` with fully opaque red [255, 0, 0, 255]

TODO: RGBA images consist of 32-bit integers whose bytes are [RR,GG,BB,AA]
      How to get that from ``x``?

data = ...

img = Image.frombuffer("RGBA", (200, 200), data)
img.save('test.png')

#
# Mini-exercise
#
# 1. Now, modify ``x`` and img.save() again. What happens?
#
