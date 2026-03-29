import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_michelson():
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 设置坐标轴范围和比例
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_aspect('equal')
    ax.axis('off') # 隐藏坐标轴
    
    # --- 定义组件位置 ---
    center = (5, 5)       # 分光镜中心
    source_x = -1         # 光源位置
    m1_y = 10             # M1 (上方反射镜)
    m2_x = 10             # M2 (右侧反射镜)
    detector_y = -1       # 探测器位置
    
    # --- 绘制组件 ---
    
    # 1. 分束立方 (Beam Splitter Cube)
    # 绘制正方形轮廓
    bs_cube = patches.Rectangle((4, 4), 2, 2, facecolor='aliceblue', edgecolor='blue', alpha=0.4, linewidth=1.5, zorder=2)
    ax.add_patch(bs_cube)
    
    # 绘制半透半反面对角线 (左下到右上，将左侧入射光反射到上方)
    ax.plot([4, 6], [4, 6], color='blue', linewidth=2, alpha=0.6, zorder=2)
    
    ax.text(5.5, 3.5, 'BS Cube\n(分束立方)', fontsize=10, fontweight='bold', color='blue', ha='center')

    # 2. 补偿板 (已移除)

    # 3. 反射镜 M1 (上方，固定)
    m1 = patches.Rectangle((4, 9.8), 2, 0.4, facecolor='silver', edgecolor='black', zorder=3)
    ax.add_patch(m1)
    ax.text(5, 10.5, 'M1 (固定反射镜)', ha='center', fontsize=12, fontweight='bold')
    
    # 4. 反射镜 M2 (右侧，可动)
    m2 = patches.Rectangle((9.8, 4), 0.4, 2, facecolor='silver', edgecolor='black', zorder=3)
    ax.add_patch(m2)
    ax.text(10.5, 5, 'M2 (固定在100nm精密位移台上的可动反射镜)', va='center', rotation=270, fontsize=12, fontweight='bold')
    
    # 5. 光源 (Laser)
    source = patches.Rectangle((-1.5, 4.5), 1.5, 1, facecolor='#333333', edgecolor='black', zorder=3)
    ax.add_patch(source)
    ax.text(-0.75, 5.8, '光源 (Laser)', ha='center', fontsize=12, fontweight='bold')
    
    # 6. 毛玻璃 (Ground Glass)
    frosted = patches.Rectangle((4.1, 1.8), 1.8, 0.5, facecolor='whitesmoke', edgecolor='gray', alpha=0.8, hatch='///', zorder=3)
    ax.add_patch(frosted)
    ax.text(5, 2.6, '毛玻璃 (接收干涉条纹)', ha='center', fontsize=11, color='dimgray')

    # 7. 探测器 (CCD)
    detector = patches.Rectangle((4, -1.5), 2, 1, facecolor='#333333', edgecolor='black', zorder=3)
    ax.add_patch(detector)
    ax.text(5, -2, 'CCD(连接电脑)', ha='center', fontsize=12, fontweight='bold')
    
    # --- 绘制光路 (使用箭头) ---
    
    # 样式设置
    ray_props = dict(head_width=0.15, head_length=0.2, fc='red', ec='red', width=0.03, length_includes_head=True, zorder=1)
    
    # 光路 1: 光源 -> 分光镜
    ax.arrow(0, 5, 4.5, 0, **ray_props)
    
    # 光路 2: 分光镜 -> M1 (反射光)
    ax.arrow(5, 5, 0, 4.8, **ray_props)
    # M1 -> 分光镜 (返回)
    ax.arrow(5, 9.8, 0, -4.8, **ray_props)
    
    # 光路 3: 分光镜 -> M2 (透射光)
    ax.arrow(5, 5, 4.8, 0, **ray_props)
    # M2 -> 分光镜 (返回)
    ax.arrow(9.8, 5, -4.8, 0, **ray_props)
    
    # 光路 4: 分光镜 -> 探测器 (汇聚)
    ax.arrow(5, 5, 0, -5.5, **ray_props)
    
    # --- 标注光路 ---
    ax.text(2, 5.2, '入射光', color='red', fontsize=10)
    ax.text(5.2, 7.5, '光臂 1', color='red', fontsize=10)
    ax.text(7.5, 5.2, '光臂 2', color='red', fontsize=10)
    ax.text(5.2, 2, '干涉光', color='red', fontsize=10)

    # 标题
    plt.title('迈克尔逊干涉仪光路图', fontsize=16, pad=20)
    
    # 保存
    output_path = 'michelson_diagram.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"原理图已生成并保存至: {output_path}")

if __name__ == "__main__":
    # 检查是否安装了 matplotlib，如果没有则提示
    try:
        import matplotlib
        # 配置字体以支持中文 (尝试常见的中文字体)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'SimSun']
        plt.rcParams['axes.unicode_minus'] = False
        draw_michelson()
    except ImportError:
        print("错误: 未安装 matplotlib 库。请运行 'pip install matplotlib' 安装。")
    except Exception as e:
        print(f"绘图出错: {e}")
