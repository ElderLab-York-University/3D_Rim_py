import matplotlib.pyplot as plt
import numpy as np
fov = 2


plt.show()

x = np.linspace(0, 10, 100)

# 创建一个大图，子图布局为 3 行 2 列
fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# 拉平 axs 方便迭代
axs = axs.ravel()

for i in range(5):
    corrections = np.load('correctionsfov' + str(fov) + str(i*20+10)+'.npy')
    plt.xlim(-1, 1)
    axs[i].hist(corrections)
    axs[i].set_title('Plot ' + str(i+1))

# 删除多余的子图
fig.delaxes(axs[5])

plt.tight_layout()
plt.show()