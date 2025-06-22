import numpy as np
from ms_tools import * 

"""
This script reads in a particular datafile and determines exactly how long a typical cycle takes.
This is important as process times may be slightly off from their 'ideal value' (a couple milliseconds every step can be enough).
Determining how much is key to get a good QMS reorganization.

This file is somewhat 'dumb' in that it just runs over a number of conditions in a range specified by the user.
For a faster method: see determine_cycleshift_faster

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

# the range of shifts to be investigated 
# (here, from -1 to 5 s with a resolution of 0.01 
# ==> increase the resolution while decreasing the range for more precise determination
extrashift = np.arange(-1, 5, 0.01)

# below this line no changes should happen
# ========================================

# read raw data from hiden spectrometer:
# pressure, mass, time
p, m, t = read_bar(filename) 

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

# diff part
diff = np.zeros(extrashift.size)

for shiftindex in np.arange(extrashift.size):
    # not very important, determines where you start
    start_time = 0
    # extremely important
    modulus = process_length + extrashift[shiftindex]
    bettertime, sorted_press, d = get_singlepulse(start_time, modulus, t_detail, p, m)
    print(shiftindex, modulus, d)
    diff[shiftindex] = d

# determine the value for extrashift where the noise is minimal
shift = extrashift[np.argmin(diff)]
print(shift)

plt.figure()
plt.plot(extrashift, diff)
plt.xlabel(r'$\delta$ cycle time')
plt.ylabel('Noise')
plt.savefig('images/error.png', dpi=300)

# make image of optimum for quick inspection
sorted_time, sorted_press, difference = get_singlepulse(start_time, process_length + shift, t_detail, p, m, sm=False)
plt.figure()
bar_heatmap(sorted_press, m, sorted_time, clipmin=1e-12)
plt.savefig('images/heatmap_no_smooth.png', dpi=300)
plt.show()