# -*- coding: utf-8 -*-
"""entangled_cycles.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WfpyEem5SxOLq9HV_QG_nltV2y2ZX0tz
"""

from PIL import Image
import re
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np 
import random
import pandas as pd
# primes_list_path='primes.csv'
# primes_list=pd.read_csv(primes_list_path)
# df=primes_list

# def isPrime(N):
#     if df[df.eq(N).all(1)].values !=0:
#         answer =   True 
#     else:
#         answer =   False
            
#     return answer

!apt-get install qt

import matplotlib.pyplot as plt 
import matplotlib.image as mgimg
from matplotlib import animation

fig = plt.figure()

# initiate an empty  list of "plotted" images 
myimages = []

#loops through available png:s
for p in range(1, 4):

    ## Read in picture
    fname = "heatflow%03d.png" %p 
    img = mgimg.imread(fname)
    imgplot = plt.imshow(img)

    # append AxesImage object to the list
    myimages.append([imgplot])

## create an instance of animation
my_anim = animation.ArtistAnimation(fig, myimages, interval=1000, blit=True, repeat_delay=1000)

## NB: The 'save' method here belongs to the object you created above
#my_anim.save("animation.mp4")

## Showtime!
plt.show()