import numpy as np
import matplotlib.pyplot as plt

from ms_tools import *
import numpy as np


"""
This script reads in a particular datafile 
and maps QMS data collected during multiple cycles onto a single cycle.
It does this using the ideal length of the process (variable 'process_length') 
and the error on that value as determined by 'determine_cycleshift.py'
"""

# input parameters, you should only change these 
# if you do not use a hiden spectrometer, you may have to adapt the read_bar function.
# ========================================
# location of the file to be processed
filename = 'test_data/raw_file.csv' 

#  QMS cycles: start at 50, end at 950 cycles before the end (the spectrometer collected a little longer after the process ended)
begin = 50 
end = -950
# ==== Reordering parameters ====
# ideal length of the process (no error) in s
process_length = 450

start_time = 10 # this is not so important, just a way to shift the begin time of the remapped spectrum
# output of determine_cycleshift.py
# extrashift = 2.3826 # determined manually
extrashift = 2.4736 # determined by determine_cycleshift_faster
# below this line no changes should happen
# ========================================

p, m, t = read_bar(filename)

p = p[:,begin:end]
t = t[begin:end]

# Official time for process, and then some. this value is determined by determine_cycleshift.py
modulus = process_length + extrashift

# === construct time for each measuring point ===
# add endpoint to t, to make loop work.
t = np.append(t, 2*t[-1]-t[-2])

# 1d time for each pressure
t_detail = []
for i in np.arange(t.size-1):
    start = t[i]
    stop = t[i+1]
    t_detail.append(np.linspace(start, stop, m.size))

t_detail = np.array(t_detail).T

#make heatmap and apply smoothing window of 3 pixels
sorted_time, sorted_press, difference = get_singlepulse(start_time, modulus, t_detail, p, m, sm=True, wdw=3)

bar_heatmap(sorted_press, m, sorted_time, clipmin=1e-12)
plt.savefig('images/heatmap_smooth.png', dpi=300)
plt.show()
