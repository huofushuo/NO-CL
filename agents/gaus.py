import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import pylab
# plt.ion()
# plt.switch_backend('agg')
matplotlib.use('TKAgg')
def normal_distribution(x, mean, sigma):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


mean1, sigma1 = 0, 1
x1 = np.linspace(mean1 - 6 * sigma1, mean1 + 6 * sigma1, 100)

# mean2, sigma2 = 0, 2
# x2 = np.linspace(mean2 - 6 * sigma2, mean2 + 6 * sigma2, 100)
#
# mean3, sigma3 = 5, 1
# x3 = np.linspace(mean3 - 6 * sigma3, mean3 + 6 * sigma3, 100)

y1 = normal_distribution(x1, mean1, sigma1)
# y2 = normal_distribution(x2, mean2, sigma2)
# y3 = normal_distribution(x3, mean3, sigma3)

plt.plot(x1, y1, 'r', label='m=0,sig=1')
# plt.plot(x2, y2, 'g', label='m=0,sig=2')
# plt.plot(x3, y3, 'b', label='m=1,sig=1')
# plt.legend()
# plt.grid()
plt.show()
pylab.show()

plt.savefig('zz_1&2.png')