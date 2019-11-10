class Config(object):
    """
    定义一个配置类
    """
    data_path = "data/"  # 加载数据的位置
    virs = "runs"  # Tensorboard数据保存位置
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
    netg_path = "checkpoints/netg_199.pth"

    gen_img = "result.png"
    gen_num = 64  # 生成图片的数量


opt = Config()
