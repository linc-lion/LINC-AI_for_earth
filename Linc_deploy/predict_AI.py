import time
import io
import torch
from PIL import Image
import torchvision
from models import detection
import json
from tempfile import SpooledTemporaryFile # To compare path for Jsonization
draw_confidence_threshold = 0.5

to_tensor = torchvision.transforms.ToTensor()
convert_to_pil = torchvision.transforms.ToPILImage()


class LINC_detector():
    def __init__(self, model_path, cpu = False):

        print('Loading checkpoint from hardrive... ', end='', flush=True)
        print(f"CPU={cpu}")
        self.model = self.load_model(model_path, cpu)
        print('Init done.')

    def load_model(self, model_path, cpu):
        self.device = 'cuda' if torch.has_cuda and not cpu else 'cpu'
        print(f"Running inference on {self.device} device")
        print('Building model and loading checkpoint into it... ', end='', flush=True)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.label_names = checkpoint['label_names']
        model = detection.fasterrcnn_resnet50_fpn(
            num_classes=len(self.label_names) + 1, pretrained_backbone=False
        )
        model.to(self.device)
        model.load_state_dict(checkpoint['model'])
        the_model = model.eval()
        print('Load done.')
        return the_model
    
    def detect(self, image_paths, image_names, conf_threshold):
        with torch.no_grad():
            
            for image_path, image_name in zip(image_paths, image_names):

                print('Loading image... ', end='', flush=True)
                image = to_tensor(Image.open(image_path)).to(self.device)

                print('Running image through model... ', end='', flush=True)
                tic = time.time()
                outputs = self.model([image])
                toc = time.time()
                print(f'Done in {toc - tic:.2f} seconds!')
                
                print(outputs)

                print(f'Saving results to file... ', end='', flush=True)
                image_dict = {'boxes': []}
                for i, score in enumerate(outputs[0]['scores']):
                    if score > conf_threshold:
                        box = outputs[0]['boxes'][i]
                        label = outputs[0]['labels'][i]
                        image_dict['boxes'].append({
                            'conf': float(score), 
                            'class': int(label), 
                            'ROI': box.tolist()
                        })
                # Check for real path
                if type(image_path) != SpooledTemporaryFile:
                    image_dict['path'] = image_path
                image_dict['name'] = image_name
                print('Done.')
        print(json.dumps(image_dict))
        return image_dict


if __name__ == '__main__':
    # Test
    image_paths = ['Images/fourLions.jpeg']
    image_names = ['image1.jpg']
    model_path = "DeployModels/body_parts.pth"
    conf_threshold = 0.5
    # load model
    LINC = LINC_detector(model_path, True)
    # detect images
    results = LINC.detect(image_paths, image_names, conf_threshold)
    # draw images
    # image = open(image_paths[0], 'rb')    
    # draw_boxes(image, results)



