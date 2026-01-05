# 检查cude 是否可用
import torch
def test_cuda():
    print(f'''Testing CUDA availability...
PyTorch version: {torch.__version__}
CUDA version: {torch.version.cuda}
          ''')

    if torch.cuda.is_available():
        print("CUDA is available. GPU can be used for computations.")
    else:
        print("CUDA is not available. Using CPU for computations.")

if __name__ == "__main__":
    test_cuda()