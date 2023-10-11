from PIL import Image

import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

fig = Figure(figsize=(5, 4), dpi=100)
# A canvas must be manually attached to the figure (pyplot would automatically
# do it).  This is done by instantiating the canvas with the figure as
# argument.
canvas = FigureCanvasAgg(fig)

# Do some plotting.
ax = fig.add_subplot()

# Generate random points
x = np.random.rand(1000)
y = np.random.rand(1000)

for i in range(100):
    ax.clear()
    ax.plot(x[:i], y[:i])
    ax.set_title(f"Frame {i}")
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    im = Image.fromarray(rgba)
    im.save(f"test_{i:03d}.bmp")
    
ax.scatter(x, y)

# Option 1: Save the figure to a file; can also be a file-like object (BytesIO,
# etc.).
#fig.savefig("test.png")

# Option 2: Retrieve a memoryview on the renderer buffer, and convert it to a
# numpy array.
canvas.draw()
rgba = np.asarray(canvas.buffer_rgba())
# ... and pass it to PIL.
im = Image.fromarray(rgba)
# This image can then be saved to any format supported by Pillow, e.g.:
im.save("test.bmp")

# Uncomment this line to display the image using ImageMagick's `display` tool.
# im.show()
