from tqdm import tqdm
from module import NetD, NetG
from tensorboardX import SummaryWriter
import torch as t
from torch.utils.data import DataLoader
import torchvision as tv
from config import opt
from torchvision.datasets import ImageFolder
from torchnet.meter import AverageValueMeter
import fire


def train(**kwargs):
    """训练网络"""

    # 0.配置参数
    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = t.device("cuda") if opt.gpu else t.device("cpu")

    # 1.加载数据并预处理
    transforms = tv.transforms.Compose([
        tv.transforms.CenterCrop(opt.img_size),
        tv.transforms.Resize(opt.img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(opt.data_path, transform=transforms)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                            num_workers=opt.num_workers, drop_last=True)

    # 2.初始化网络
    netg, netd = NetG(opt), NetD(opt)
    netg.to(device)
    netd.to(device)

    # 3.选择优化方法
    optimizer_g = t.optim.SGD(netg.parameters(), lr=opt.lr1)
    optimizer_d = t.optim.SGD(netd.parameters(), lr=opt.lr2)

    # 随机噪声
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    # 4.记录数据
    loss_d = AverageValueMeter()
    loss_g = AverageValueMeter()
    write = SummaryWriter(log_dir=opt.virs, comment="WGAN")

    # 5.训练
    for epoch in range(opt.max_epoch):
        for ii, (img, _) in tqdm(enumerate(dataloader)):
            real_img = img.to(device)

            # 5.1 训练判别器
            if ii % opt.d_every == 0:
                optimizer_d.zero_grad()
                fake_img = netg(noises).detach()
                D_loss = -1 * t.mean(netd(real_img)) + t.mean(netd(fake_img))
                D_loss.backward()
                optimizer_d.step()

                # 权重裁剪
                for p in netd.parameters():
                    p.data.clamp_(-opt.clip_value, opt.clip_value)
                loss_d.add(D_loss.item())

            # 5.2 训练Generator
            if ii % opt.g_every == 0:
                optimizer_g.zero_grad()
                gen_img = netg(noises)
                G_loss = -1 * (t.mean(netd(gen_img)))
                G_loss.backward()
                optimizer_g.step()
                loss_g.add(G_loss.item())

        # 6 数据存储
        # 6.1 权重
        if epoch % 1000 == 0:
            for name, param in netd.named_parameters():
                write.add_histogram(name, param.cpu().data.numpy(), epoch)

        # 6.2 loss
        if epoch % 200 == 0:
            write.add_scalar("Discriminator_loss", loss_d.value()[0], epoch)
            write.add_scalar("Generator_loss", loss_g.value()[0], epoch)

        if (epoch + 1) % opt.save_every == 0:
            g_img = netg(noises)

            tv.utils.save_image(g_img.data[:64], "%s%s.png" % (opt.save_path, epoch), normalize=True)
            t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
            t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
            loss_d.reset()
            loss_g.reset()

    write.close()


@t.no_grad()
def generate_images(**kwargs):
    """生成图片"""

    # 0．配置属性
    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = t.device("cuda") if opt.gpu else t.device("cpu")

    # １．制造随机噪声
    noise = t.randn(opt.gen_num, opt.nz, 1, 1)
    noise.to(device)

    # 2.加载网络与数据
    netg = NetG(opt).eval()
    netg.to(device)
    map_location = lambda storage, loc: storage
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))

    # 3.测试数据
    image = netg(noise)

    tv.utils.save_image(image, "result/gen_img.png", normalize=True)


def help_info():
    """输出帮助文档"""
    print("帮助信息:")
    print("""
    usage:python file.py <function> [--args = value]
    <function> := train | generate_images | help_info
    example:
    python {0} train 
    python {0} generate_images
    python {0} help_info
    """.format(__file__))
    print("参数的设定：\n")

    # 获得opt类参数的源码
    from inspect import getsource
    print(getsource(opt.__class__))


if __name__ == "__main__":
    fire.Fire()
