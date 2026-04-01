import argparse
import torch
import cv2
from model.ad_model import MetricMIS
def test_image(checkpoint_path, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricMIS().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    img = cv2.imread(image_path)
    H,W, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth, mask, fxn, fyn = model.infer_image(img)

    K = [[fxn, 0, 0.5],
         [0, fyn, 0.5],
         [0, 0, 1]]
    K[0,:]*=W
    K[1,:]*=H

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the MetricMIS model on a single image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    test_image(args.checkpoint, args.image)
