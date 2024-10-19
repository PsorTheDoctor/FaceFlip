from argparse import Namespace
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from encoder4editing.models.psp import pSp

import clip
from manipulate import Manipulator
from StyleCLIP import GetDt, GetBoundary

dataset_name = 'ffhq'
experiment_type = 'ffhq_encode'

EXPERIMENT_ARGS = {
    "model_path": "e4e_ffhq_encode.pt"
}
EXPERIMENT_ARGS['transform'] = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def run_alignment(image_path):
    import dlib
    from encoder4editing.utils.alignment import align_face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return images, latents


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    network_pkl = './model/' + dataset_name + '.pkl'
    device = torch.device('cuda')
    M = Manipulator()
    M.device = device
    G = M.LoadModel(network_pkl, device)
    M.G = G
    M.SetGParameters()
    num_img = 100_000
    M.GenerateS(num_img=num_img)
    M.GetCodeMS()
    np.set_printoptions(suppress=True)

    """
    encoder4editing setup
    """
    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')
    return model, M, net


def run_model(image_path, model, manipulator, net, target, neutral='a face',
              alpha=4.1, beta=0.15, resize_dims=(256, 256)):
    """
    :param model: CLIP ViT model
    :param manipulator: custom image manipulator
    :param net: neural network
    :param image_path: a path to the given image
    :param target: a prompt describing desired modifications
    :param neutral: a prompt describing the input image
    :param alpha: manipulation strength, ranges (-10, 10)
    :param beta: disentanglement threshold, ranges (0.08 - 0.3)
    :param resize_dims: image size
    :return: a modified PIL image
    """
    M = manipulator

    original_image = Image.open(image_path)
    original_image = original_image.convert("RGB")

    if experiment_type == "ffhq_encode":
        input_image = run_alignment(image_path)
    else:
        input_image = original_image

    input_image.resize(resize_dims)

    """
    Invert the image
    """
    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)

    with torch.no_grad():
        images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
        result_image, latent = images[0], latents[0]

    torch.save(latents, 'latents.pt')

    img_index = 0
    latents = torch.load('/content/encoder4editing/latents.pt')
    dlatents_loaded = M.G.synthesis.W2S(latents)

    img_indexs = [img_index]
    dlatents_loaded = M.S2List(dlatents_loaded)

    dlatent_tmp = [tmp[img_indexs] for tmp in dlatents_loaded]

    M.num_images = len(img_indexs)

    M.alpha = [0]
    M.manipulate_layers = [0]
    codes, out = M.EditOneC(0, dlatent_tmp)
    original = Image.fromarray(out[0, 0]).resize((512, 512))
    M.manipulate_layers = None

    classnames = [target, neutral]
    dt = GetDt(classnames, model)

    file_path = './npy/' + dataset_name + '/'
    fs3 = np.load(file_path + 'fs3.npy')

    M.alpha = [alpha]
    boundary_tmp2, c = GetBoundary(fs3, dt, M, threshold=beta)
    codes = M.MSCode(dlatent_tmp, boundary_tmp2)
    out = M.GenerateImg(codes)
    generated = Image.fromarray(out[0, 0])
    return generated


# Example usage
# if __name__ == '__main__':
    # model, manipulator, net = load_model()
    # run_model('adam.jpg', model, manipulator, target='a grumpy face')
