import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def read_bar(filename):

    # 1D arrays
    masses = []
    partpress = []
    times = []

    # counters
    cycle = 0
    l = 0

    with open(filename) as f:
        for line in f:
            # read basic info
            if l == 6:
                lowermass = float(line.split(',')[4])
                uppermass = float(line.split(',')[5])

                masses = np.arange(lowermass, uppermass + 1, 1)

            if line[0] == ' ':

                partpress.append(float(line.split(',')[1]))

            elif line[0] != '"':
                try:
                    times.append(float(line.split(',')[3]))
                    cycle += 1

                except ValueError:
                    None
                except IndexError:
                    None
            l+=1

            # if l%10000==0:
            #     print(l)

    times=np.array(times)
    masses=np.array(masses)
    partpress=np.array(partpress)
    partpress = np.reshape(partpress, newshape=(times.size, masses.size)).T
    times/=1000

    return partpress, masses, times

def read_mid(filename, mass):
    partpress = np.array([])
    times = np.array([])

    # counter
    l = 0

    masscheck = False
    massindex = 0

    with open(filename) as f:
        for line in f:
            linesplit = line.split(',')

            if masscheck:

                if ':' in linesplit[0]:
                    try:
                        partpress = np.append(partpress, float(linesplit[massindex]))
                        times = np.append(times, float(linesplit[1]))
                    except ValueError:
                        None
                    except IndexError:
                        None

            # read basic info

            # check whether mass is in spectrum
            else:


                try:
                    # "Scan 3","RGA","Faraday","mass",59,59,0.01,100,100,1,1,
                    # this line takes the 3 as massindex
                    float(linesplit[0].split(' ')[1][:-1])

                    if float(linesplit[4]) == mass:
                        masscheck = True
                        massindex+=2
                    else:
                        massindex+=1

                except ValueError:
                    None
                except IndexError:
                    None

            l+=1


    return times/1000, partpress


def bar_heatmap(pressure, masses, times, clipmin=1e-10):
    # data manipulation
    pressure = pressure.clip(min=clipmin)

    X, Y = np.meshgrid(times, masses)
    Z = pressure

    plt.pcolormesh(X, Y, Z, norm=LogNorm(vmin=Z.min(), vmax=Z.max()), cmap='viridis')
    cbar = plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('m/z')
    cbar.set_label('Signal intensity (arb.unit)', rotation=270, labelpad=10)
    plt.tight_layout()


def export_txt(filename, sorted_p, m, sorted_t):
    X, Y = np.meshgrid(sorted_t, m)
    Z = sorted_p
    np.savetxt(filename, np.array([X.flatten(), Y.flatten(), Z.flatten()]).T)

def read_txt(filename):
    t, m, p = np.loadtxt(filename, usecols=(0,1,2), unpack=True)
    high_mass = int(np.max(m))
    lower_mass = int(np.min(m))
    number_of_masses = high_mass-lower_mass+1

    # print(lower_mass, high_mass, number_of_masses)
    # m = np.arange(lower_mass, high_mass+1, 1)
    # print(t.size, m.size)

    p = np.reshape(p, (number_of_masses, int(t.size/number_of_masses)))
    t = t[:int(t.size/number_of_masses)]
    m = np.arange(lower_mass, high_mass+1, 1)


    return t, m, p

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_singlepulse(start_time, modulus, t_detail, p, m, sm=False, wdw=10):
    bettertime = (t_detail - start_time) % modulus
    sorted_press = np.zeros(p.shape)
    difference = 0

    for i in np.arange(m.size):
        inds = bettertime[i, :].argsort()
        # if smoothing per mass is wanted, adapt here
        if not sm:
            sorted_press[i, :] = p[i, inds]
        elif sm:
            sorted_press[i, :] = smooth(p[i, inds], wdw)

        sorted_time = bettertime[i, inds] #not optimal, updated every time now.

        # calc difference
        # sorted_press = sorted_press.clip(min=1e-10, max=sorted_press.max())
        sorted_press_shift = np.roll(sorted_press[i], 1)
        difference += np.sum(np.abs(sorted_press[i] - sorted_press_shift))


    return sorted_time, sorted_press, difference

def vertical_avg(p,m,t,t1,t2):
    index = np.argwhere(np.logical_and(t>t1, t<t2)).ravel()
    return m, np.sum(p[:,index], axis=1)/index.size

def horizontal(p,m,t,m1):
    index = np.argwhere(m == m1).ravel()[0]
    print(index)
    print(p[index].shape)
    return t, p[index]

def construct_t_detail(t, m):
    # === construct time for each measuring point ===
    # add endpoint to t, to make loop work.
    t = np.append(t, 2 * t[-1] - t[-2])

    # 1d time for each pressure
    t_detail = []
    for i in np.arange(t.size - 1):
        start = t[i]
        stop = t[i + 1]
        t_detail.append(np.linspace(start, stop, m.size))

    t_detail = np.array(t_detail).T
    return t_detail


def optimize_diff(p, m, t, base_time, extrashift, plot=True):

    t_detail = construct_t_detail(t, m)

    diff = np.zeros(extrashift.size)

    for shiftindex in np.arange(extrashift.size):
        # not very important, determines where you start
        start_time = 0
        # extremely important
        modulus = base_time + extrashift[shiftindex]
        bettertime, sorted_press, d = get_singlepulse(start_time, modulus, t_detail, p, m)
        #
        # bar_heatmap(sorted_press, m, bettertime)
        # plt.show()

        print(shiftindex, modulus, d)
        diff[shiftindex] = d

    shift = extrashift[np.argmin(diff)]
    print(shift)
    if plot:
        plt.figure()
        plt.plot(extrashift, diff)
        plt.show()
    return extrashift, diff, shift