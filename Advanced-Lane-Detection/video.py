import pipeline as pp

import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
pipe=pp.pipeline()
def process_image(image):
    img = pipe.lanes(image)
    return img


output ='./output.mp4'
clip1 = VideoFileClip("./project_video.mp4")
white_clip = clip1.fl_image(process_image) 
white_clip.write_videofile(output, audio=False)


