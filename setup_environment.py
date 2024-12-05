import os
import sys
import subprocess
import platform
from pathlib import Path

class EnvironmentSetup:
    def __init__(self):
        # 定义项目所需的基本配置
        self.project_name = "qwen_mobile"
        self.python_version = "3.9"
        self.requirements = {
            'base': [
                'torch',
                'transformers',
                'onnx',
                'onnxruntime',
                'numpy',
                'pandas',
                'tqdm',
                'colorama',
                'pytest'  # 用于测试
            ],
            'development': [
                'black',  # 代码格式化
                'flake8',  # 代码检查
                'isort',  # import排序
                'pre-commit'  # Git钩子
            ]
        }
        
        # 确定操作系统类型
        self.is_windows = platform.system() == "Windows"
        self.is_cuda_available = self._check_cuda_availability()
    
    def _check_cuda_availability(self):
        """检查系统是否支持CUDA"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            # 如果torch未安装，检查nvidia-smi
            try:
                subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True
            except FileNotFoundError:
                return False
    
    def _run_command(self, command, description):
        """执行shell命令并打印状态"""
        print(f"\n{description}...")
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"✓ {description} 完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ {description} 失败")
            print(f"错误信息: {str(e)}")
            return False
    
    def setup_conda_environment(self):
        """设置Conda环境"""
        # 检查conda是否已安装
        if not self._run_command("conda --version", "检查Conda安装"):
            print("请先安装Miniconda或Anaconda")
            print("下载地址: https://docs.conda.io/miniconda/miniconda-install/")
            return False
        
        # 创建新的conda环境
        env_name = self.project_name
        if not self._run_command(
            f"conda create -y -n {env_name} python={self.python_version}",
            f"创建Conda环境 {env_name}"
        ):
            return False
        
        # 设置国内镜像源
        if self.is_windows:
            conda_rc = Path(os.environ['USERPROFILE']) / '.condarc'
        else:
            conda_rc = Path.home() / '.condarc'
        
        with open(conda_rc, 'w', encoding='utf-8') as f:
            f.write("""channels:
  - defaults
  - pytorch
  - conda-forge
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
""")
        
        return True
    
    def install_dependencies(self):
        """安装项目依赖"""
        # 安装PyTorch
        if self.is_cuda_available:
            torch_command = "conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
        else:
            torch_command = "conda install -y pytorch torchvision torchaudio cpuonly -c pytorch"
        
        if not self._run_command(torch_command, "安装PyTorch"):
            return False
        
        # 设置pip国内镜像
        pip_config = """[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
"""
        if self.is_windows:
            pip_conf_path = Path(os.environ['APPDATA']) / 'pip' / 'pip.ini'
        else:
            pip_conf_path = Path.home() / '.pip' / 'pip.conf'
        
        pip_conf_path.parent.mkdir(parents=True, exist_ok=True)
        pip_conf_path.write_text(pip_config)
        
        # 安装其他依赖
        for req in self.requirements['base']:
            if not self._run_command(f"pip install {req}", f"安装 {req}"):
                return False
        
        # 安装开发依赖
        for req in self.requirements['development']:
            if not self._run_command(f"pip install {req}", f"安装开发工具 {req}"):
                return False
        
        return True
    
    def verify_installation(self):
        """验证安装"""
        verification_code = """
import torch
import transformers
import onnx
import onnxruntime

def print_package_info():
    print("环境检查结果:")
    print("-" * 50)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    
    print(f"Transformers版本: {transformers.__version__}")
    print(f"ONNX版本: {onnx.__version__}")
    print(f"ONNX Runtime版本: {onnxruntime.__version__}")
    print("-" * 50)

print_package_info()
"""
        
        verify_path = "verify_install.py"
        with open(verify_path, 'w', encoding='utf-8') as f:
            f.write(verification_code)
        
        print("\n运行安装验证...")
        subprocess.run([sys.executable, verify_path])
        os.remove(verify_path)
    
    def setup(self):
        """运行完整的环境设置流程"""
        print("开始设置开发环境...")
        
        if not self.setup_conda_environment():
            return False
        
        if not self.install_dependencies():
            return False
        
        self.verify_installation()
        
        print("\n环境设置完成！")
        print(f"请使用 'conda activate {self.project_name}' 激活环境")
        return True

if __name__ == "__main__":
    setup = EnvironmentSetup()
    setup.setup()