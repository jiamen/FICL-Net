import matplotlib.pyplot as plt
import numpy as np

# 这两行代码解决 plt 中文显示的问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 输入统计数据
datasets = ('Oxford', 'U.S.', 'R.A.', 'B.D.')
FN_T_Net = [91.85, 95.55, 89.52, 87.65]
FN_Non_T_Net = [93.85, 96.65, 92.48, 91.16]

bar_width = 0.25  # 条形宽度
t_net = np.arange(len(datasets))  # t_net条形图的横坐标
non_t_net = t_net + bar_width     # non_t_net条形图的横坐标

# 使用两次 bar 函数画出两组条形图
plt.bar(t_net, height=FN_T_Net, width=bar_width, color='royalblue', label='FN(T-Net)-FL-Net')
plt.bar(non_t_net, height=FN_Non_T_Net, width=bar_width, color='darkorange', label='FN(non-T-Net)-FL-Net')

plt.legend(loc='lower right')         # 显示图例
plt.xticks(t_net + bar_width/2, datasets)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Average recall @1%')    # 纵坐标轴标题
# plt.title('result')  # 图形标题


for a,b in zip(t_net, FN_T_Net):
    plt.text(a, b+0.05, '%.2f' % b, ha='center', va='bottom', fontsize=8)

for a,b in zip(non_t_net, FN_Non_T_Net):
    plt.text(a, b+0.05, '%.2f' % b, ha='center', va='bottom', fontsize=8)

plt.show()