import os
import sys
import platform
import subprocess

def check_matlab_pandas_compatibility():
    print("--- MATLAB Runtime & Pandas 兼容性检测 ---")
    
    # 1. 检测 Pandas
    try:
        import pandas as pd
        pandas_version = pd.__version__
        has_pandas = True
        print(f"✅ [Python] 已安装 Pandas: {pandas_version}")
    except ImportError:
        has_pandas = False
        print("❌ [Python] 未找到 Pandas，无法使用 DataFrame 传递功能。")

    # 2. 检测 MATLAB Runtime 版本
    # 在 Windows 上，最可靠的方法是检查环境变量或注册表
    mcr_versions = []
    if platform.system() == "Windows":
        path_env = os.environ.get('PATH', '')
        # 寻找类似于 v913 这种格式的路径
        import re
        mcr_matches = re.findall(r'MATLAB Runtime\\(v\d+)', path_env)
        mcr_versions = list(set(mcr_matches)) # 去重
    else:
        # Linux/macOS 通常检查 LD_LIBRARY_PATH
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        import re
        mcr_versions = re.findall(r'v(\d+)', ld_path)

    # 3. 判定逻辑
    # R2022b 对应的内部版本号是 v913
    SUPPORT_VERSION_INT = 913 
    
    if not mcr_versions:
        print("⚠️  [Runtime] 未在系统 PATH 中检测到 MATLAB Runtime。")
        print("   (提示：请确保已安装 MCR 并且其 \runtime\win64 目录已添加到环境变量)")
    else:
        for v in mcr_versions:
            v_num = int(v.replace('v', ''))
            is_supported = v_num >= SUPPORT_VERSION_INT
            status = "✅ 支持" if is_supported else "❌ 不支持"
            print(f"📍 [Runtime] 检测到版本: {v} -> {status} (需要 v913/R2022b 或更高)")

    # 4. 最终结论
    print("\n--- 结论 ---")
    if has_pandas and any(int(v.replace('v', '')) >= SUPPORT_VERSION_INT for v in mcr_versions):
        print("🚀 准备就绪！你可以直接在 Python 和 MATLAB 编译包之间传递 DataFrame。")
    else:
        print("🛠️  环境不匹配。如果需要传递 DataFrame，请升级至 MATLAB R2022b Runtime。")

if __name__ == "__main__":
    check_matlab_pandas_compatibility()