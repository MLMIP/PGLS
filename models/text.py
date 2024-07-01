import matplotlib.pyplot as plt

# 定义数据
total_loss = [77.921, 75.519, 72.976, 69.017, 67.406, 64.921, 63.81, 62.722, 61.175, 60.817, 59.697]
loss = [75.837, 72.746, 69.72, 65.95, 64.448, 62.187, 61.057, 59.73, 58.229, 58.264, 57.321]
loss_cov = [77.473, 74.852, 72.01, 68.27, 66.591, 64.209, 63.392, 62.015, 60.197, 60.085, 59.197]
loss_reg = [76.768, 73.953, 71.391, 67.566, 66.116, 63.551, 62.378, 60.983, 59.746, 59.399, 58.239]

# 创建画布和子图对象
fig, ax = plt.subplots()

# 绘制折线图
ax.plot(range(0, 11), total_loss, label='total loss')
ax.plot(range(0, 11), loss, label='loss')
ax.plot(range(0, 11), loss_cov, label='loss + cov_loss')
ax.plot(range(0, 11), loss_reg, label='loss + 0.1*super_loss')

# 添加标题和坐标轴标签
ax.set_title('ablation experiment')
ax.set_xlabel('Session')
ax.set_ylabel('ACC')

# 添加图例
ax.legend()

# 显示图形
plt.show()