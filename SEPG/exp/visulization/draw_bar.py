import numpy as np
import matplotlib.pyplot as plt

size = 3
x = np.arange(size)
a = np.random.random(size)
b = np.random.random(size)
# seg
a = np.array([21.9, 43.5, 20.1])
b = np.array([24.3, 43.8, 24.3])

cat = ['AP','AP50', 'AP75']

# det
a = np.array([22.1, 47.3, 22.1])
b = np.array([27.3, 45.3, 28.5])


total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

bar1 = plt.bar(x, a,  width=width, label='P2BNet')
bar2 = plt.bar(x + width, b, width=width, label='SAM')
plt.xticks(x, cat)
# for bar in bar1+bar2:
#     height = bar.get_height()
#     print(height)
#     plt.text(bar.get_x(), +bar.get_width()/2, height, float(height), ha='center', va = 'bottom')
# plt.yticks([]), 
# plt.bar(x + 2 * width, c, width=width, label='c')
plt.legend()
plt.show()

