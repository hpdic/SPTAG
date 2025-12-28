import numpy as np
import os
import subprocess
import sys

# ================= 配置区域 =================

# 获取当前脚本所在目录，用于构建相对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 关键路径配置 (使用相对路径指向 Release 目录)
# 假设结构是:
#   ~/SPTAG/hpdic/script/ (本脚本)
#   ~/SPTAG/Release/      (二进制和索引)
RELEASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../Release"))

BINARY_PATH = os.path.join(RELEASE_DIR, "indexsearcher")
INDEX_FOLDER = os.path.join(RELEASE_DIR, "sift1m_index")

# 2. 数据路径 (绝对路径)
QUERY_FILE = "/home/cc/AdaDisk/experiments/data/sift/sift_query.fvecs"
TRUTH_FILE_IVECS = "/home/cc/AdaDisk/experiments/data/sift/sift_groundtruth.ivecs"

# 3. 临时文件存放位置 (建议放在脚本同级目录，或者也放回 Release)
# 这里我们把它放在脚本当前目录下，方便管理
TRUTH_FILE_TXT = os.path.join(CURRENT_DIR, "sift_groundtruth_converted.truth")

# 4. 性能参数 (调整这里来控制 Recall 和 QPS)
# 目标: 85-95% Recall
MAX_CHECK = 1024  
THREAD_NUM = 32
K_NEIGHBORS = 100 

# ==========================================================

def convert_ivecs_to_txt(ivecs_path, txt_path):
    if os.path.exists(txt_path):
        # 简单检查一下文件大小不为0
        if os.path.getsize(txt_path) > 0:
            print(f"[Info] Truth 文件已存在: {txt_path}，跳过转换。")
            return

    print(f"[Converting] 正在生成 TXT 格式 Truth 文件: {txt_path} ...")
    try:
        raw_data = np.fromfile(ivecs_path, dtype='int32')
        dim = raw_data[0]
        data = raw_data.reshape(-1, dim + 1)[:, 1:]
        np.savetxt(txt_path, data, fmt='%d')
        print(f"[Success] 转换完成。")
    except Exception as e:
        print(f"[Error] 转换失败: {e}")
        sys.exit(1)

def run_benchmark():
    # 检查文件是否存在
    if not os.path.exists(BINARY_PATH):
        print(f"[Error] 找不到可执行文件: {BINARY_PATH}")
        print("请检查路径是否正确，或是否已编译 Release 版本。")
        return
    if not os.path.exists(INDEX_FOLDER):
        print(f"[Error] 找不到索引目录: {INDEX_FOLDER}")
        return

    # 组装命令
    cmd = [
        BINARY_PATH,
        "-x", INDEX_FOLDER,
        "-i", QUERY_FILE,
        "-r", TRUTH_FILE_TXT,
        "-d", "128",
        "-v", "Float",
        "-f", "XVEC",
        "-t", str(THREAD_NUM),
        "-m", str(MAX_CHECK),
        "-k", str(K_NEIGHBORS)
    ]

    print("\n" + "="*60)
    print(f"Working Directory: {CURRENT_DIR}")
    print(f"Target Binary:     {BINARY_PATH}")
    print(f"MaxCheck:          {MAX_CHECK}")
    print("="*60 + "\n")

    try:
        # 注意：这里不需要改变 cwd，因为我们要引用绝对路径或相对路径
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] 运行出错: {e.returncode}")

if __name__ == "__main__":
    # 确保 numpy 存在
    try:
        import numpy
    except ImportError:
        print("Error: 需要安装 numpy (pip install numpy)")
        sys.exit(1)

    # 1. 确保目标目录存在
    if not os.path.exists(os.path.dirname(TRUTH_FILE_TXT)):
        os.makedirs(os.path.dirname(TRUTH_FILE_TXT))

    # 2. 转换数据
    convert_ivecs_to_txt(TRUTH_FILE_IVECS, TRUTH_FILE_TXT)

    # 3. 运行
    run_benchmark()