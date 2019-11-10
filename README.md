### WGAN的使用
#### 0.算法主要原理

#### 1.安装所需要的包
命令：pip install -r requirements.txt
#### 2.使用命令进行训练
命令：python main.py funcation --参数＝值
#### 3.可选参数设置选择
    data_path = "data/"　#　加载数据的位置
    virs = "runs" # tensorboard数据保存位置
    clip_value = 0.01  # 权重裁剪边界
    num_workers = 4  # 多线程
    img_size = 96  # 剪切图片的大小
    batch_size = 512
    max_epoch = 100000
    lr1 = 0.01  # 生成器
    lr2 = 0.01  # 判别器
    gpu = True
    nz = 100  # 噪声维度
    ngf = 64  # 生成器的feature map 数
    ndf = 64  # 判别器的feature map 数

    save_path = 'Imgs/'  # 生成图片的保存路径
    d_every = 1  # 每一个batch 训练一次判别器
    g_every = 5  # 每5个batch训练一次生成模型
    save_every = 200  # 每200次保存一次模型
    netd_path = None
    netg_path = None

    gen_img = "result.png"
    gen_num = 64  #生成图片的数量
#### 4.打印帮助信息
    usage:python file.py <function> [--args = value]
    <function> := train | generate_images | help_info
    example:
    python main.py train 
    python main.py generate_images
    python main.py help_info
#### 5.CSDN博客
[WGAN]()