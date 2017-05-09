import pipeline as pp

import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
pipe=pp.pipeline()
img = mpimg.imread("./test_images/test2.jpg")


img = pipe.lanes(img)
plt.imshow(img)
plt.title('Result')
plt.show()
