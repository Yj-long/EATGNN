import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体，防止图表中的中文标题或标签出现乱码（这里以 Windows 的黑体为例）
# 如果是 macOS，可以将 'SimHei' 改为 'Arial Unicode MS'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def generate_line_charts_from_csv(file_path):
    """
    读取 CSV 文件并将每一列生成以行数为横坐标的折线图。
    """
    try:
        # 1. 读取 CSV 文件
        # 如果文件包含中文路径或特殊编码，可以尝试加上 encoding='utf-8' 或 encoding='gbk'
        df = pd.read_csv(file_path)

        # 2. 获取行数作为横坐标 (这里自动使用 DataFrame 的 index，即 0, 1, 2, ...)
        x_axis = df.index

        # 3. 遍历表格的每一列
        for column_name in df.columns:
            # 创建一个新的图形，设置大小
            plt.figure(figsize=(10, 5))

            # 绘制折线图：横坐标为 x_axis，纵坐标为当前列的数据
            # marker='.' 表示在数据点上画一个小点，方便查看具体位置
            plt.plot(x_axis, df[column_name], marker='.', linestyle='-', linewidth=2)

            # 设置图表的标题和坐标轴标签
            plt.title(f'折线图 - {column_name}', fontsize=16)
            plt.xlabel('行数', fontsize=12)
            plt.ylabel(column_name, fontsize=12)

            # 开启网格线，让图表更易读
            plt.grid(True, linestyle='--', alpha=0.7)

            # 调整布局以防止标签被截断
            plt.tight_layout()

            # 显示图表
            # 注意：在弹出的图表窗口中，你需要关闭当前窗口才会自动生成并显示下一列的图表
            plt.show()

            # 如果你想直接将图表保存为本地图片而不是一张张查看，
            # 可以注释掉上面的 plt.show()，并取消注释下面这行代码：
            plt.savefig(f'{column_name}_折线图.png', dpi=300)

        print("所有列的折线图已处理完毕！")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'，请检查文件路径是否正确。")
    except Exception as e:
        print(f"发生错误：{e}")


# ================= 运行示例 =================
# 将 'your_file.csv' 替换为你实际的 CSV 文件路径
if __name__ == "__main__":
    csv_file = r'/checkpoints_eatgnn_01\training_log.csv'
    generate_line_charts_from_csv(csv_file)