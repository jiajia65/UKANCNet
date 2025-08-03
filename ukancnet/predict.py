import os
import time
# import transforms as T
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
# from src.ukan.archs import UKAN
# from src.ukan.kanCCFM import kanCCFMabc
# from src.baseline import UMamba
# from src.ukan.kanCCFM import kanCCFMabc
# from src.method3.TCNnet import TCNNet #method3
from src.method4.trans4passplus import Trans4PASS #method4
# from src.method5.vit_seg_modeling import VisionTransformer   #method5
# from src.method5.vit_seg_configs import get_b32_config  # #method5
# from src.method3.TCNnet import TCNNet
# from src.deeplabv3.deeplabv3_model import deeplabv3_resnet50
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # classes = 4  # exclude background
    classes = 2
    # weights_path = "./save_weights/best_model.pth"
    weights_path = "./save_weights/heat_method4.pth"
    # imgs_path = r"E:\dataset\WTB\new\small_test\images"
    # imgs_path = r"E:\dataset\WTB\new\small_test\images"
    imgs_path = r"E:\dataset\seg\Heat_Sink_Surface_Defect_Dataset\predict\images"
    # roi_mask_path = r"E:\dataset\WTB\new\training\masks"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(imgs_path), f"image {imgs_path} not found."
    # assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # model = UKAN(in_channels=3, num_classes=classes+1, base_c=32)
    # model = kanCCFMabc( num_classes=classes + 1)
    # model = TCNNet()  # method3
    # config = get_b32_config()  # 返回的是一个对象，不是字符串#method5
    # model = VisionTransformer(config=config, img_size=512, num_classes=3)  # method5
    model = Trans4PASS(num_classes=3, emb_chans=128, encoder='trans4pass_v2')  # method4
    # load weights
    state_dict = torch.load(weights_path, map_location='cpu')
    # 过滤掉不需要的键
    # state_dict = {k: v for k, v in state_dict.items() if not any(layer in k for layer in ['channel_attention.fc1', 'channel_attention.fc2', 'channel_attention.fc3', 'channel_attention.fc4'])}
    # model.load_state_dict(state_dict, strict=False)

    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    for filename in os.listdir(imgs_path):
        if filename.endswith('.png'):
            a = filename  # 获取文件名
            img_path = os.path.join(imgs_path, a)  # 合并路径
            # mask_path = os.path.join(roi_mask_path, a)  # 将合并后的路径添加到c列表中

            # roi_img = Image.open(mask_path)
            # roi_img = np.array(roi_img)

            # load image
            original_img = Image.open(img_path).convert('RGB')

            # from pil image to tensor and normalize
            # data_transform = transforms.Compose([transforms.ToTensor(),
            #                                      transforms.Normalize(mean=mean, std=std)])
            trans = [transforms.Resize(512)]
            trans.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            data_transform = transforms.Compose(trans)
            img = data_transform(original_img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            model.eval()  # 进入验证模式
            with torch.no_grad():
                # init model
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)

                # t_start = time_synchronized()
                output = model(img.to(device))
                # t_end = time_synchronized()
                # print("inference time: {}".format(t_end - t_start))

                prediction = output['out'].argmax(1).squeeze(0)
                prediction = prediction.to("cpu").numpy().astype(np.uint8)
                h, w = prediction.shape
                # 将前景对应的像素值改成255(白色)
                new_predict = np.zeros(shape=(h, w, 3), dtype=np.uint8)
                prediction = prediction[np.newaxis, :, :]
                new_predict[np.all(prediction == 1, axis=0)] = [128, 0, 0]
                new_predict[np.all(prediction == 2, axis=0)] = [0, 128, 0]
                # new_predict[np.all(prediction == 3, axis=0)] = [128, 128, 0]
                # new_predict[np.all(prediction == 4, axis=0)] = [0, 0, 128]
                # new_predict[np.all(prediction == 1, axis=0)] = [255, 255, 255]
                # 将不敢兴趣的区域像素设置成0(黑色)
                new_predict[np.all(prediction == 0, axis=0)] = [0, 0, 0]
                mask = Image.fromarray(new_predict)
                # folder_a = 'E:/dataset/WTB/new/small_test/predict_89.6'
                # folder_a = 'E:/dataset/mvtec/predict/method1'
                folder_a = 'E:\dataset\seg\Heat_Sink_Surface_Defect_Dataset\predict\method4'
                save_path = os.path.join(folder_a, a)
                mask.save(save_path)



if __name__ == '__main__':
    main()
