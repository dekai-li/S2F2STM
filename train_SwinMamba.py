import sys
import os
# sys.path.append('/content/drive/MyDrive/hyperspectral classification/MambaHSI')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES']='0' #选择服务器
import time
import torch
# print(torch.cuda.device_count())  # 检查GPU数量
# print(torch.cuda.get_device_name(0))  # 检查设备名称
import random
import argparse
import numpy as np
from torchvision import models, transforms
# import matplotlib.pyplot as plt
# from visual.visualize_map import DrawResult
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss, resize
from utils.evaluation import Evaluator
from utils.HSICommonUtils import normlize3D, ImageStretching
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict
from PIL import Image
# from model.MambaHSI import ImprovedMambaHSI as MambaHSI
from model.SwinMamba import SwinMambaHSI as SwinMamba_head_1
from model.SwinMamba import SwinMambaHSI_NoSwinTransformer as SwinMambaHSI_NoSwinTransformer
from model.SwinMamba import SwinMambaHSI_NoMamba as SwinMambaHSI_NoMamba
from model.SwinMamba import SwinMambaHSI_NoMultiScaleConv as SwinMambaHSI_NoMultiScaleConv
from model.SwinMamba import SwinMambaHSI_NoMultiScale as SwinMambaHSI_NoMultiScale
from calflops import calculate_flops
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

# max_split_size_mb 控制内存分配时的块大小，防止内存碎片化。
# garbage_collection_threshold 控制内存垃圾回收的触发条件，有助于更有效地回收和管理 GPU 内存。
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6'
torch.autograd.set_detect_anomaly(True)

# Initialize mixed precision scaler
# GradScaler 是 PyTorch 中一个用于 混合精度训练 的工具，主要用于处理梯度的缩放（scaling）操作。
scaler = GradScaler()

# Define gradient accumulation steps 梯度累积
accumulation_steps = 4  # Adjust this based on your GPU memory

# 返回当前本地时间 '24-11-28-15.23'
time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())


def vis_a_image(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=False):
    visualize_predict(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=only_vis_label)
    visualize_predict(gt_vis, pred_vis, save_single_predict_path.replace('.png', '_mask.png'), save_single_gt_path, only_vis_label=True)

# random seed setting 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)  # 设置 CPU 上的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 哈希种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    random.seed(seed)  # 设置 Python 内建 random 模块的随机种子
    torch.backends.cudnn.deterministic = True  # 确保每次计算的结果是确定性的
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 自动优化，确保每次运行的一致性

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=int, default=0)
    parser.add_argument('--data_set_path', type=str, default='./data')
    parser.add_argument('--work_dir', type=str, default='./')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--train_samples', type=int, default=30)
    parser.add_argument('--val_samples', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='RUNS')
    parser.add_argument('--record_computecost', type=bool, default=False)

    args = parser.parse_args()
    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = get_parser()
record_computecost = args.record_computecost
exp_name = args.exp_name
seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_list = [args.train_samples, args.val_samples] # 用于存储训练样本数和验证样本数

dataset_index = args.dataset_index # 选择的数据集索引（如 0、1、2 等），通过索引选择不同的数据集。
max_epoch = args.max_epoch
learning_rate = args.lr
net_name = 'SwinMamba_head_1'
# 'net_name'：网络名称。
# 'dataset_index'：数据集的索引。
# 'num_list'：包含训练样本数和验证样本数的列表。
# 'lr'：学习率。
# 'seed_list'：随机种子列表。
paras_dict = {'net_name': net_name, 'dataset_index': dataset_index, 'num_list': num_list, 'lr': learning_rate, 'seed_list': seed_list}
data_set_name_list = ['UP', 'HanChuan', 'HongHu', 'Houston','LongKou','Salinas','indian','Botswana','XuZhou','Pavia']
data_set_name = data_set_name_list[dataset_index]
split_image = data_set_name in ['HanChuan', 'Houston','Pavia']

transform = transforms.Compose([
    transforms.ToTensor(),
]) # 图像数据从 PIL 图片或 numpy 数组转换为 PyTorch 张量（Tensor）


if __name__ == '__main__':
    data_set_path = args.data_set_path   # ./data
    work_dir = args.work_dir          #./
    # tr30val10_lr0.0003
    setting_name = 'tr{}val{}'.format(str(args.train_samples), str(args.val_samples)) + '_lr{}'.format(str(learning_rate))
    dataset_name = data_set_name # UP 。。。
    exp_name = args.exp_name

    save_folder = os.path.join(work_dir, exp_name, net_name, dataset_name) # './RUNS/SwinMamba/UP'
    # 路径不存在创建路径
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("makedirs {}".format(save_folder))

    save_log_path = os.path.join(save_folder, 'train_tr{}_val{}.log'.format(num_list[0], num_list[1]))
    # './RUNS/SwinMamba/UP/train_tr30_val10.log'

    logger = setup_logger(name='{}'.format(dataset_name), logfile=save_log_path)
    torch.cuda.empty_cache() # 清理 PyTorch 的 GPU 显存缓存
    logger.info(save_folder)

    # Load data and apply preprocessing
    data, gt = data_load_operate.load_data(data_set_name, data_set_path)

    # Apply Gaussian filtering
    data_filtered = gaussian_filter(data, sigma=1)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=30)
    # 将数据重塑为一个二维数组，其中 -1 表示根据数据的总大小自动计算行数，data_filtered.shape[2] 是光谱波段数，表示每一行代表一个像素的所有光谱特征。
    # 例如，假设 data_filtered 的形状是 (100, 100, 200)，那么 data_reshaped 将变成一个形状为 (10000, 200) 的二维矩阵，其中 10000 是图像的所有像素点（100 × 100），200 是每个像素的光谱波段数。
    data_reshaped = data_filtered.reshape(-1, data_filtered.shape[2])
    # data_pca 是降维后的数据，形状为 (10000, 30)，每行表示一个像素，只有 30 个主成分特征。
    data_pca = pca.fit_transform(data_reshaped)
    # data_pca 在此时的形状是 (10000, 30)，而 data_filtered.shape 是 (100, 100, 200)。通过重新调整形状，data_pca 的形状变为 (100, 100, 30)，表示每个像素点现在只有 30 个主成分特征。
    data_pca = data_pca.reshape(data_filtered.shape[0], data_filtered.shape[1], -1)

    # Update data shape and other parameters based on PCA-preprocessed data
    height, width, channels = data_pca.shape
    # 假设 data_pca 的形状是 (100, 100, 30)，那么：height = 100 width = 100 channels = 30
    gt_reshape = gt.reshape(-1)
    # 通过 -1 自动计算出合适的维度，将 gt 展平为一个长向量，便于后续处理。
    # 例如，假设 gt 是一个形状为 (100, 100) 的二维数组，表示每个像素的类别标签。通过 reshape(-1)，它会变成一个形状为 (10000,) 的一维数组。
    img = ImageStretching(data_pca)
    class_count = max(np.unique(gt)) # 计算类别数量

    flag_list = [1, 0]  # ratio or num
    ratio_list = [0.1, 0.01]  # [train_ratio, val_ratio]
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)

    OA_ALL = []  # 用于存储所有实验的总体精度（Overall Accuracy，OA）
    AA_ALL = []  # 用于存储所有实验的平均精度（Average Accuracy，AA）
    KPP_ALL = []  # 用于存储所有实验的Kappa系数（Kappa Coefficient，KPP）
    EACH_ACC_ALL = []  # 用于存储每个类别的精度（Class-wise Accuracy）
    Train_Time_ALL = []  # 用于存储每个实验的训练时间
    Test_Time_ALL = []  # 用于存储每个实验的测试时间
    CLASS_ACC = np.zeros([len(seed_list), class_count])  # 用于存储每个实验和每个类别的准确度
    evaluator = Evaluator(num_class=class_count)

    # 循环执行多个实验
    for exp_idx, curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)

        # 创建实验结果保存目录
        single_experiment_name = 'run{}_seed{}'.format(str(exp_idx), str(curr_seed)) # run0_seed0
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name) # # './RUNS/SwinMamba/UP/run0_seed0'
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis') # './RUNS/SwinMamba/UP/run0_seed0/vis'
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)
            print("makedirs {}".format(save_vis_folder))

        save_weight_path = os.path.join(save_single_experiment_folder, "best_tr{}_val{}.pth".format(num_list[0], num_list[1]))
        results_save_path = os.path.join(save_single_experiment_folder, 'result_tr{}_val{}.txt'.format(num_list[0], num_list[1]))
        predict_save_path = os.path.join(save_single_experiment_folder, 'pred_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        gt_save_path = os.path.join(save_single_experiment_folder, 'gt_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))

        # 输入参数：
        # ratio_list：一个包含训练集和验证集样本比例的列表，例如 [train_ratio, val_ratio]。
        # num_list：一个包含每个类别固定样本数量的列表，例如 [train_num, val_num]。
        # gt_reshape：重新调整形状后的标签数组，包含了每个像素点的类别标签。
        # class_count：类别总数，通常是数据集中标签的类别数。
        # Flag：控制分割方式的标志位，Flag=0 表示按比例分割，Flag=1 表示按固定样本数分割。
        train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(ratio_list, num_list, gt_reshape, class_count, flag_list[0])
        index = (train_data_index, val_data_index, test_data_index)
        # 输入参数：
        # data_padded：输入的影像数据，通常是经过预处理或填充的图像数据（在此函数中未直接使用，可能用于后续的任务中）。
        # hsi_h：图像的高度（即行数），用于生成二维标签图。
        # hsi_w：图像的宽度（即列数），也用于生成二维标签图。
        # label_reshape：重新排列的标签数组，通常包含所有类别的标签，用于将数据从一维标签映射到二维标签图中。
        # index：包含训练集、验证集、测试集样本索引的元组。具体来说：
        # index[0]：训练集的样本索引。
        # index[1]：验证集的样本索引。
        # index[2]：测试集的样本索引。
        # 输出：
        # y_tensor_train：训练集标签的张量，数据类型为 FloatTensor。
        # y_tensor_val：验证集标签的张量，数据类型为 FloatTensor。
        # y_tensor_test：测试集标签的张量，数据类型为 FloatTensor。
        train_label, val_label, test_label = data_load_operate.generate_image_iter(data_pca, height, width, gt_reshape, index)

        # build Model  单GPU
        net = SwinMamba_head_1(in_channels=channels, num_classes=class_count, hidden_dim=128)

        # # ####################修改模型定义部分#######################     多GPU
        # net = MambaHSI(in_channels=channels, num_classes=class_count, hidden_dim=128)
        # net = torch.nn.DataParallel(net)  # 将模型包装为 DataParallel
        # net.to(device)  # 移动到 GPU
        # #############################################################

        logger.info(paras_dict)
        logger.info(net)
        # np.array(img) 将图像 img 转换为 NumPy 数组。
        # transform(np.array(img)) 对图像进行预处理操作，通常包括归一化、数据增强、尺寸调整等。具体的转换操作会在 transform 函数中定义。
        # .unsqueeze(0) 在图像数据前添加一个维度，转换为 [1, channels, height, width] 的格式，以适配模型的输入要求（批量大小为 1）。
        x = transform(np.array(img))
        x = x.unsqueeze(0).float().to(device)
        print(f"x shape: {x.shape}")

        train_label = train_label.to(device)
        test_label = test_label.to(device)
        val_label = val_label.to(device)

        # ############################################
        # val_label = test_label
        # ############################################

        net.to(device)

        train_loss_list = [100]
        train_acc_list = [0]
        val_loss_list = [100]
        val_acc_list = [0]

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

        logger.info(optimizer)
        best_loss = 99999
        if record_computecost:
            net.eval()
            torch.cuda.empty_cache()

            # Reduce the input size for calculating FLOPs to avoid out-of-memory errors
            # flops：每秒浮点运算数，是衡量模型计算复杂度的指标。
            # macs1：乘加操作数，表示模型计算过程中所执行的乘法加法操作的总数。
            # para：模型的参数总数，表示模型的大小。
            flops, macs1, para = calculate_flops(model=net, input_shape=(1, x.shape[1], x.shape[2], x.shape[3]))

            logger.info("para:{}\n,flops:{}".format(para, flops))

        tic1 = time.perf_counter()
        best_val_acc = 0
        for epoch in range(max_epoch):
            y_train = train_label.unsqueeze(0) # 将 train_label 数据的维度增加一个轴，通常是因为模型期望输入的是一个 4D 张量 (batch_size, channels, height, width)，而 train_label 可能是 3D 的。
            train_acc_sum, trained_samples_counter = 0.0, 0
            batch_counter, train_loss_sum = 0, 0
            time_epoch = time.time()
            loss_dict = {}

            net.train()

            if split_image:
                x_part1 = x[:, :, :x.shape[2] // 2 + 5, :]
                y_part1 = y_train[:, :x.shape[2] // 2 + 5, :]
                x_part2 = x[:, :, x.shape[2] // 2 - 5:, :]
                y_part2 = y_train[:, x.shape[2] // 2 - 5:, :]

                # 第一部分前向传播
                y_pred_part1 = net(x_part1)
                ls1 = head_loss(loss_func, y_pred_part1, y_part1.long())
                optimizer.zero_grad()
                ls1.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                # 第二部分前向传播
                y_pred_part2 = net(x_part2)
                ls2 = head_loss(loss_func, y_pred_part2, y_part2.long())
                optimizer.zero_grad()
                ls2.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                logger.info('Iter:{}|loss:{}'.format(epoch, (ls1 + ls2).detach().cpu().numpy()))


            else:
                try:
                    # autocast()：这个上下文管理器启用混合精度训练，它可以减少计算时间并减少显存占用。scaler 用于处理梯度缩放，使得反向传播时能更稳定。
                    with autocast():
                         y_pred = net(x)
                         ls = head_loss(loss_func, y_pred, y_train.long())
                    optimizer.zero_grad()
                    scaler.scale(ls).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.empty_cache()

                # except 处理：如果内存不足或计算出现问题（比如 OOM 错误），会切换为 split_image = True，重新将图像切分为两部分并训练。这是一个容错机制，防止内存不足导致训练崩溃。
                except:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    split_image = True
                    x_part1 = x[:, :, :x.shape[2] // 2 + 5, :]
                    y_part1 = y_train[:, :x.shape[2] // 2 + 5, :]
                    x_part2 = x[:, :, x.shape[2] // 2 - 5:, :]
                    y_part2 = y_train[:, x.shape[2] // 2 - 5:, :]

                    y_pred_part1 = net(x_part1)
                    ls1 = head_loss(loss_func, y_pred_part1, y_part1.long())
                    optimizer.zero_grad()
                    ls1.backward()
                    optimizer.step()

                    y_pred_part2 = net(x_part2)
                    ls2 = head_loss(loss_func, y_pred_part2, y_part2.long())
                    optimizer.zero_grad()
                    ls2.backward()
                    optimizer.step()

                    logger.info('Iter:{}|loss:{}'.format(epoch, (ls1 + ls2).detach().cpu().numpy()))

            torch.cuda.empty_cache()

            # Evaluation stage
            net.eval()
            with torch.no_grad():
                evaluator.reset()
                output_val = net(x) # 使用验证数据进行前向传播。
                y_val = val_label.unsqueeze(0)
                seg_logits = resize(input=output_val, size=y_val.shape[1:], mode='bilinear', align_corners=True)
                predict = torch.argmax(seg_logits, dim=1).cpu().numpy() # 将模型输出的 logits 转换为类别标签
                Y_val_np = val_label.cpu().numpy()
                Y_val_255 = np.where(Y_val_np == -1, 255, Y_val_np)
                evaluator.add_batch(np.expand_dims(Y_val_255, axis=0), predict)
                OA = evaluator.Pixel_Accuracy()
                mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                Kappa = evaluator.Kappa()
                logger.info('Evaluate {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA, mAcc, Kappa, mIOU, IOU, Acc))

                # Save the best model based on validation accuracy
                if OA >= best_val_acc:
                    best_epoch = epoch + 1
                    best_val_acc = OA
                    torch.save(net.state_dict(), save_weight_path)

                if (epoch + 1) % 50 == 0:
                    save_single_predict_path = os.path.join(save_vis_folder, 'predict_{}.png'.format(str(epoch + 1)))
                    save_single_gt_path = os.path.join(save_vis_folder, 'gt.png')
                    vis_a_image(gt, predict, save_single_predict_path, save_single_gt_path)

            torch.cuda.empty_cache()
        toc1 = time.perf_counter()  # 记录结束时间
        train_time = toc1 - tic1  # 计算时间间隔
        logger.info(f"train_time: {train_time} seconds")

        # Final testing phase with the best model
        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        tic2 = time.perf_counter()
        pred_test = []

        load_weight_path = save_weight_path
        net.update_params = None
        best_net = SwinMamba_head_1(in_channels=channels, num_classes=class_count, hidden_dim=128)
        best_net.to(device)
        best_net.load_state_dict(torch.load(load_weight_path))
        best_net.eval()

        test_evaluator = Evaluator(num_class=class_count)
        # torch.no_grad() 确保在推理阶段不会计算梯度，减少内存消耗。
        # test_evaluator.reset() 重置评估器的混淆矩阵，以便对新的预测结果进行统计。
        # output_test = best_net(x) 将输入 x（测试集图像）传入模型 best_net，获取预测输出 output_test。
        # seg_logits_test 使用 resize 将模型输出的尺寸调整为与实际标签 y_test 相同的大小。resize 使用双线性插值来调整输出尺寸。
        # predict_test = torch.argmax(seg_logits_test, dim=1) 对模型输出进行分类，选择每个像素点的最大概率类作为预测类别。
        # Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np) 这里将标签中为 -1 的部分替换为 255，通常用于标记忽略的区域。
        # test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), predict_test) 将预测结果和真实标签传递给评估器进行评估
        with torch.no_grad():
            test_evaluator.reset()
            output_test = best_net(x)

            y_test = test_label.unsqueeze(0)
            seg_logits_test = resize(input=output_test, size=y_test.shape[1:], mode='bilinear', align_corners=True)
            predict_test = torch.argmax(seg_logits_test, dim=1).cpu().numpy()
            Y_test_np = test_label.cpu().numpy()
            Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)
            test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), predict_test)
            OA_test = test_evaluator.Pixel_Accuracy()
            mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
            mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
            Kappa_test = evaluator.Kappa()
            logger.info('Test {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA_test, mAcc_test, Kappa_test, mIOU_test, IOU_test, Acc_test))
            vis_a_image(gt, predict_test, predict_save_path, gt_save_path)
        toc2 = time.perf_counter()  # 记录结束时间
        test_time = toc2 - tic2

        # Save results to file
        f = open(results_save_path, 'a+')
        str_results = '\n======================' \
                      + " exp_idx=" + str(exp_idx) \
                      + " seed=" + str(curr_seed) \
                      + " learning rate=" + str(learning_rate) \
                      + " epochs=" + str(max_epoch) \
                      + " train ratio=" + str(ratio_list[0]) \
                      + " val ratio=" + str(ratio_list[1]) \
                      + " ======================" \
                      + "\nOA=" + str(OA_test) \
                      + "\nAA=" + str(mAcc_test) \
                      + '\nkpp=' + str(Kappa_test) \
                      + '\nmIOU_test:' + str(mIOU_test) \
                      + "\nIOU_test:" + str(IOU_test) \
                      + "\nAcc_test:" + str(Acc_test) + "\n"
        logger.info(str_results)
        f.write(str_results)
        f.close()

        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)
        Train_Time_ALL.append(train_time)
        Test_Time_ALL.append(test_time)

        torch.cuda.empty_cache()
            # Summarize the results across all runs
    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    logger.info("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    # logger.info('List of OA:', list(OA_ALL))
    # logger.info('List of AA:', list(AA_ALL))
    # logger.info('List of KPP:', list(KPP_ALL))
    # logger.info('OA=', round(np.mean(OA_ALL) * 100, 2), '+-', round(np.std(OA_ALL) * 100, 2))
    # logger.info('AA=', round(np.mean(AA_ALL) * 100, 2), '+-', round(np.std(AA_ALL) * 100, 2))
    # logger.info('Kpp=', round(np.mean(KPP_ALL) * 100, 2), '+-', round(np.std(KPP_ALL) * 100, 2))
    # logger.info('Acc per class=', np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2), '+-',
    #             np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2))
    #
    # logger.info("Average training time=", round(np.mean(Train_Time_ALL), 2), '+-', round(np.std(Train_Time_ALL), 3))
    # logger.info("Average testing time=", round(np.mean(Test_Time_ALL) * 1000, 2), '+-', round(np.std(Test_Time_ALL) * 1000, 3))

    ############改################
    logger.info('List of OA: {}'.format(list(OA_ALL)))
    logger.info('List of AA: {}'.format(list(AA_ALL)))
    logger.info('List of KPP: {}'.format(list(KPP_ALL)))
    logger.info('OA: {:.2f} ± {:.2f}'.format(np.mean(OA_ALL) * 100, np.std(OA_ALL) * 100))
    logger.info('AA: {:.2f} ± {:.2f}'.format(np.mean(AA_ALL) * 100, np.std(AA_ALL) * 100))
    logger.info('Kpp: {:.2f} ± {:.2f}'.format(np.mean(KPP_ALL) * 100, np.std(KPP_ALL) * 100))
    logger.info('Acc per class: {} ± {}'.format(
        np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2).tolist(),
        np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2).tolist()
    ))
    # 检查训练时间和测试时间数组是否为空
    if len(Train_Time_ALL) > 0:
        avg_train_time = np.mean(Train_Time_ALL)
        std_train_time = np.std(Train_Time_ALL)
    else:
        avg_train_time, std_train_time = 0, 0  # 默认值
        logger.warning("Train_Time_ALL 为空，训练时间无法计算，使用默认值 0。")

    if len(Test_Time_ALL) > 0:
        avg_test_time = np.mean(Test_Time_ALL) * 1000
        std_test_time = np.std(Test_Time_ALL) * 1000
    else:
        avg_test_time, std_test_time = 0, 0  # 默认值
        logger.warning("Test_Time_ALL 为空，测试时间无法计算，使用默认值 0。")

    # 日志记录
    logger.info('Average training time: {:.2f} ± {:.3f}'.format(avg_train_time, std_train_time))
    logger.info('Average testing time: {:.2f} ± {:.3f}'.format(avg_test_time, std_test_time))
    ##############################

    # Save final summary results
    mean_result_path = os.path.join(save_folder, 'mean_result.txt')
    with open(mean_result_path, 'w') as f:
        str_results = '\n\n***************Mean result of ' + str(len(seed_list)) + ' times runs ********************' \
                      + '\nList of OA:' + str(list(OA_ALL)) \
                      + '\nList of AA:' + str(list(AA_ALL)) \
                      + '\nList of KPP:' + str(list(KPP_ALL)) \
                      + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                      + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                      + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(round(np.std(KPP_ALL) * 100, 2)) \
                      + '\nAcc per class=\n' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)) + '+-' + str(
            np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)) \
                      + "\nAverage training time=" + str(np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' + str(
            np.round(np.std(Train_Time_ALL), decimals=3)) \
                      + "\nAverage testing time=" + str(np.round(np.mean(Test_Time_ALL) * 1000, decimals=2)) + '+-' + str(
            np.round(np.std(Test_Time_ALL) * 100, decimals=3))
        f.write(str_results)

    del net

# Optional cleanup
torch.cuda.empty_cache()


