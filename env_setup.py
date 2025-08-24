import os
import re
import subprocess

def set_env():
    # 获取 ~/.bashrc 的绝对路径
    bashrc_path = os.path.expanduser("~/.bashrc")
    # 读取文件内容
    with open(bashrc_path, "r") as f:
        content = f.read()
    pattern = r'^(export\s+)?CUDA_PROJ_HOME=[^\n]+'
    # 判断bashrc中是否定义了CUDA_PROJ_HOME
    is_defined = re.search(pattern, content, flags=re.MULTILINE)
    if is_defined:
        # 尝试获取CUDA_PROJ_HOME环境变量
        CUDA_PROJ_HOME = os.getenv("CUDA_PROJ_HOME")
        print(f'CUDA_PROJ_HOME 环境变量已定义, CUDA_PROJ_HOME={CUDA_PROJ_HOME}')
        # 启动一个新的 Bash Shell，并执行 source 后保留交互式环境
        subprocess.run(["bash", "-c", "unset PYTHONPATH && source ~/.bashrc && exec bash"], check=True)
        return
    else:
        print("CUDA_PROJ_HOME 环境变量不存在, 正在设置默认值...")
        current_file_path = os.path.abspath(__file__)
        CUDA_PROJ_HOME = os.path.dirname(current_file_path)
        
        # 把CUDA_PROJ_HOME追加在末尾
        content = content.rstrip() + "\n\n" + f'export CUDA_PROJ_HOME={CUDA_PROJ_HOME}\n'
        
        # 把PYTHONPATH追加在末尾
        pythonpath = os.getenv("PYTHONPATH")
        append_pythonpath = "$CUDA_PROJ_HOME"
        content = content.rstrip() + "\n" + f'export PYTHONPATH={append_pythonpath}:$PYTHONPATH\n'
        
        # 写回文件
        with open(bashrc_path, "w") as f:
            f.write(content)
        
        # 启动一个新的 Bash Shell，并执行 source 后保留交互式环境
        subprocess.run(["bash", "-c", "source ~/.bashrc && exec bash"], check=True)
        
        print(f'CUDA_PROJ_HOME 环境变量设置成功, CUDA_PROJ_HOME={CUDA_PROJ_HOME}')

if __name__ == "__main__":
    set_env()