import matplotlib.pyplot as plt
import numpy as np
fov = 2


plt.show()

x = np.linspace(0, 10, 100)

# 创建一个大图，子图布局为 3 行 2 列
fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# 拉平 axs 方便迭代
axs = axs.ravel()

for i in range(2,8):
    corrections = np.load('corrections' + str(fov) +'.npy')
    plt.xlim(-1, 1)
    axs[i-2].hist(corrections,bins=50)
    axs[i-2].set_title('Fov pi/' + str(i))

# 删除多余的子图

plt.tight_layout()
plt.show()