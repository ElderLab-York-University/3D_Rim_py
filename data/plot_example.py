import matplotlib.pyplot as plt
import numpy as np

# 假设 n 是你的训练步长，将其替换为实际的值
n = 100

# 这里只是为了演示，你需要用实际的reward数据替换这些
reward1 = np.random.rand(n)
reward2 = np.random.rand(n)
reward3 = np.random.rand(n)
reward4 = np.random.rand(n)

# 创建一个新的图像
plt.figure()

# 绘制四条曲线
plt.plot(reward1, label='Learning Rate 1', color='red')
plt.plot(reward2, label='Learning Rate 2', color='blue')
plt.plot(reward3, label='Learning Rate 3', color='green')
plt.plot(reward4, label='Learning Rate 4', color='orange')

# 设置标题和坐标轴标签
plt.title('Reward Data for Different Learning Rates')
plt.xlabel('Training Step')
plt.ylabel('Reward')

# 在左下角放置图例
plt.legend(loc='lower left')

# 显示图像
plt.show()