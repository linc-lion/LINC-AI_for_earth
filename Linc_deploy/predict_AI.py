import time
import torch
from PIL import Image
import torchvision
from utils import draw_boxes
import json
draw_confidence_threshold = 0.5

to_tensor = torchvision.transforms.ToTensor()
convert_to_pil = torchvision.transforms.ToPILImage()


class LINC_detector():
    def __init__(self, model_path, cpu = False):

        print('Loading checkpoint from hardrive... ', end='', flush=True)
        print(cpu)
        self.model = self.load_model(model_path, cpu)
        # print(self.model)
        print('Init done.')

    def load_model(self, model_path, cpu):
        print(cpu)
        self.device = 'cuda' if torch.has_cuda and not cpu else 'cpu'
        print(f"Running inference on {self.device} device")
        print('Building model and loading checkpoint into it... ', end='', flush=True)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.label_names = checkpoint['label_names']
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            num_classes=len(self.label_names) + 1, pretrained_backbone=False
        )
        model.to(self.device)
        model.load_state_dict(checkpoint['model'])
        model = model.eval()
        print('Load done.')
        return model
    
    def detect(self, image_paths, image_names, conf_threshold):
        with torch.no_grad():
            
            image_dicts = []
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
                image_dict['path'] = image_path
                image_dict['name'] = image_name
                image_dicts.append(image_dict)
                print('Done.')
        # print(json.dumps(image_dicts))
        return image_dicts

if __name__ == '__main__':
    # Test
    image_paths = ['Images/good1.jpg', 'Images/good2.jpg', 'Images/bad1.jpg']
    image_names = ['good1.jpg', 'good2.jpg', 'bad1.jpg']
    model_path = "Models/body_parts_0_0_1.pth" 
    conf_threshold = 0.5
    # load model
    LINC = LINC_detector(model_path, True)
    # detect images
    LINC.detect(image_paths, image_names, conf_threshold)



