Project/
├── main.py                  # [入口文件] 程序的启动入口
├── requirements.txt         # [依赖文件] 项目所需的第三方库
├── pyproject.toml           # [依赖文件] 项目所需的第三方库
├── README.md                # [说明文档] 项目介绍
├── .gitignore               # [Git配置] 忽略不需要提交的文件
├── config.py                # [全局配置] 常量、路径配置 (可选，也可放在 src/config)
├── setup.py                 # [安装脚本] 打包和分发脚本 (可选)
│
├── assets/                  # [资源文件夹] 存放图片、图标、样式表、字体
│   ├── icons/
│   ├── images/
│   └── styles.qss           # (如果是PyQt) 样式表文件
│
├── src/                     # [核心源码] 所有的源代码都放在这里
│   ├── __init__.py
│   │
│   ├── ui/                  # [界面层] 只负责“长什么样”，不写复杂逻辑
│   │   ├── __init__.py
│   │   ├── main_window.py   # 主窗口布局
│   │   └── widgets/         # 自定义的小组件
│   │       ├── __init__.py
│   │       └── custom_button.py
│   │
│   ├── core/                # [业务逻辑层] 负责“怎么做”，不包含界面代码
│   │   ├── __init__.py
│   │   ├── data_manager.py  # 数据处理、计算逻辑
│   │
│   ├── database/            # [数据层] 数据库模型和操作 (可选)
│   │   ├── __init__.py
│   │   └── models.py
│   │
│   └── utils/               # [工具层] 通用函数，与业务无关
│       ├── __init__.py
│       ├── logger.py        # 日志记录配置
│       └── helpers.py       # 路径处理、格式转换等辅助函数
│
└── tests/                   # [测试目录]
    ├── __init__.py
    ├── test_core.py
    └── test_ui.py


安装环境
```bash
uv pip install torch torchvision -f https://mirrors.aliyun.com/pytorch-wheels/cu128
```