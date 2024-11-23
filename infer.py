import torch
import torchvision.transforms.v2 as transforms
import segmentation_models_pytorch as segm
import numpy as np
import cv2
import os
from argparse import ArgumentParser

color_dict= {2: (0, 0, 0),0: (255, 0, 0),1: (0, 255, 0)}

def mask_to_rgb(mask):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))

    for k in color_dict.keys():
        output[mask==k] = color_dict[k]

    return np.uint8(output) 

def main(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transforms = transforms.Compose([transforms.Resize((256,256)),
        transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #ImageNet mean and std
    ])
                        
    net = segm.UnetPlusPlus(encoder_name = "resnet50",encoder_depth = 5, encoder_weights = 'imagenet', in_channels=3, classes=3, decoder_channels=(256,128,64,32,16),decoder_use_batchnorm=True)

    checkpoint = torch.load("weight.pth")
    net.load_state_dict(checkpoint)
    net.eval()
    net.to(device)

    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_w = ori_img.shape[0]
    ori_h = ori_img.shape[1]
    img = cv2.resize(ori_img, (256, 256))
    img = test_transforms(img)
    input_img = img.unsqueeze(0).to(device)
    output_mask = net(input_img).squeeze(0).cpu().detach().numpy().transpose(1,2,0)
    mask = cv2.resize(output_mask, (ori_h, ori_w))
    mask = np.argmax(mask, axis=2)
    mask_rgb = mask_to_rgb(mask)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    hehe = os.path.dirname(img_path)
    if hehe == "":
        hehe = "."
    cv2.imwrite(f"{hehe}/output.jpeg", mask_rgb)
    print(f"Image successfully created at {hehe}/output.jpeg!")
 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()
    main(args.image_path)

