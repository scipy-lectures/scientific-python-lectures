"""
From buffer
============

Show how to exchange data between numpy and a library that only knows
the buffer interface.
"""

import numpy as np
from PIL import Image

# Let's make a sample image, RGBA format

x = np.zeros((200, 200, 4), dtype=np.uint8)

x[:, :, 0] = 255  # red
x[:, :, 3] = 255  # opaque

data = x.view(np.int32)  # Check that you understand why this is OK!

img = Image.frombuffer("RGBA", (200, 200), data)
img.save("test.png")

# Modify the original data, and save again.

x[:, :, 1] = 255
img.save("test2.png")
