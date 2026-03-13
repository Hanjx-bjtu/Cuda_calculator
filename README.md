# CUDA 计算器

> 开发 CUDA 计算器项目，实现高效的 GPU 加速矩阵运算。

> **目前版本：V0.1.0 开发中**

## 项目描述

这是一个基于 NVIDIA CUDA 的计算器项目，专注于实现各种矩阵运算的 GPU 加速版本，包括矩阵加法、乘法、卷积等操作。通过利用 GPU 的并行计算能力，提供比 CPU 实现更快的计算性能。

## 功能特性

- **矩阵加法**: CUDA 实现的矩阵加法运算
- **矩阵乘法**: 高效的矩阵乘法 GPU 加速
- **矩阵卷积**: CUDA 卷积运算实现
- **Python 接口**: 提供 Python 绑定，便于集成和使用
- **测试套件**: 包含完整的测试用例验证正确性

## 项目结构

```
Cuda_calculator/        # 项目文档目录
├── LICENSE             # 许可证文件
├── README.md           # 项目说明
├── requirements.txt    # 依赖
└── Demostrate_Code/    # 示例代码目录
    ├── matrix_add.cu   # 加法示例
    ├── matrix_add.py   # 加法 Python 示例
    ├── matrix_conv.cu  # 卷积示例
    ├── matrix_conv.py  # 卷积 Python 示例
    ├── matrix_mul.cu   # 乘法示例
│   └── matrix_mul.py   # 乘法 Python 示例
```

## 环境要求

- **CUDA Toolkit**: 版本 10.0 或更高
- **NVIDIA GPU**: 支持 CUDA 的显卡
- **Python**: 3.6+
- **编译器**: GCC (Linux) 或 MSVC (Windows) 支持 CUDA

## 安装和编译

1. 安装 CUDA Toolkit：
   ```bash
   # 下载并安装 CUDA Toolkit
   # https://developer.nvidia.com/cuda-toolkit
   ```

2. 安装 Python 依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 编译 CUDA 程序：
   ```bash
   # 使用提供的任务编译
   nvcc -g -G -o matrix_add matrix_add.cu
   nvcc -g -G -o matrix_mul matrix_mul.cu
   nvcc -g -G -o matrix_conv matrix_conv.cu
   ```

## 使用方法

### CUDA 程序直接运行

```bash
# 矩阵加法
./Cuda_calculator/matrix_add

# 矩阵乘法
./Cuda_calculator/matrix_mul

# 矩阵卷积
./Cuda_calculator/matrix_conv
```

### Python 接口

```python
# 导入相应的模块
import matrix_add_cuda
import matrix_mul_cuda
import matrix_conv_cuda

# 使用示例
# 具体使用方法请参考各个模块的文档
```

### 运行测试

```bash
python matrix_test.py
python matrix_add_cuda.py
python matrix_mul_test.py
python matrix_conv_test.py
python matrix_fft_test.py
```

## 开发和贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。

### 开发环境设置

1. 克隆项目
2. 安装依赖
3. 编译 CUDA 代码
4. 运行测试验证

## 许可证

请查看 LICENSE 文件了解许可证信息。

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

