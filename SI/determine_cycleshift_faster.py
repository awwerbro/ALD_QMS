import numpy as np
from ms_tools import * 
from lmfit import minimize, Parameters, Parameter

"""
This script reads in a particular datafile and determines exactly how long a typical cycle takes.
This is important as process times may be slightly off from their 'ideal value' (a couple milliseconds every step can be enough).
Determining how much is key to get a good QMS reorganization.

This file attempts to improve on the naive method used in 'determine_cycleshift' by using the lmfit package
and minimizing the noise instead of just doing a grid search. This should work much faster.
However, if you get weird results using this method, I recommend manually inspecting the grid through the slower method. 

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

# the range of shifts to be investigated (here, from -1 to 10 s)
extrashift_min = -1
extrashift_max = 10

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

def wrap(pars):
    # first parameter is start time, which is not important.
    # the only thing that needs to be optimized is the modulus
    _, _, d = get_singlepulse(0, process_length + pars['shift'].value, t_detail, p, m)
    return d

params = Parameters()
params['shift'] = Parameter(name='shift', value=0, min=extrashift_min, max=extrashift_max)

result = minimize(wrap, params, method='brute')
print(result.params.pretty_print())
shift = result.params['shift'].value

# make image of optimum for quick inspection
sorted_time, sorted_press, difference = get_singlepulse(start_time, process_length + shift, t_detail, p, m, sm=False)
plt.figure()
bar_heatmap(sorted_press, m, sorted_time, clipmin=1e-12)
plt.savefig('images/heatmap_no_smooth.png', dpi=300)
plt.show()