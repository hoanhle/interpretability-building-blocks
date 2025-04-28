from pyviewer.docking_viewer import DockingViewer, dockable
from imgui_bundle import imgui
import numpy as np
import os
from pyviewer.params import *  # type: ignore
from examples.deepdream import deep_dream_static_image, save_and_maybe_display_image, SupportedModels, SupportedPretrainedWeights, OUT_IMAGES_PATH, INPUT_DATA_PATH
from enum import Enum
from copy import deepcopy
import cv2 as cv
from PIL import Image
    

class VGG16Layers(Enum):
    conv1_1 = 'conv1_1'
    relu1_1 = 'relu1_1'
    conv1_2 = 'conv1_2'
    relu1_2 = 'relu1_2'
    conv2_1 = 'conv2_1'
    relu2_1 = 'relu2_1'
    conv2_2 = 'conv2_2'
    relu2_2 = 'relu2_2'
    conv3_1 = 'conv3_1'
    relu3_1 = 'relu3_1'
    conv3_2 = 'conv3_2'
    relu3_2 = 'relu3_2'
    relu3_3 = 'relu3_3'
    relu4_1 = 'relu4_1'
    relu4_2 = 'relu4_2'
    relu4_3 = 'relu4_3'
    relu5_1 = 'relu5_1'
    relu5_2 = 'relu5_2'
    relu5_3 = 'relu5_3'
    mp5 = 'mp5'

class InputImage(Enum):
    DOG_CAT = 'cat-dog.jpeg'
    FISH = 'n01443537_goldfish.JPEG'


@strict_dataclass
class State(ParamContainer):
    input_image: Param = EnumParam('Input Image', InputImage.DOG_CAT, InputImage)
    img_width: Param = IntParam('Image Width', 224, 1, 1000)
    layers_to_use: Param = EnumParam('Layers to Use', VGG16Layers.relu1_1, VGG16Layers)
    channel_to_use: Param = IntParam('Channel to Use', -1, -1, 32, buttons=True)
    model_name: Param = EnumParam('Model Name', SupportedModels.VGG16_EXPERIMENTAL.name, [m.name for m in SupportedModels])
    pretrained_weights: Param = EnumParam('Pretrained Weights', SupportedPretrainedWeights.IMAGENET.name, [pw.name for pw in SupportedPretrainedWeights])
    pyramid_size: Param = IntParam('Pyramid Size', 1, 1, 10)
    pyramid_ratio: Param = FloatParam('Pyramid Ratio', 1.8, 1.0, 2.0)
    num_gradient_ascent_iterations: Param = IntParam('Gradient Ascent Iterations', 10, 1, 100)
    lr: Param = FloatParam('Learning Rate', 0.09, 0.01, 1.0)
    spatial_shift_size: Param = IntParam('Spatial Shift Size', 0, 0, 100)
    smoothing_coefficient: Param = FloatParam('Smoothing Coefficient', 0.5, 0.0, 2.0)
    use_noise: Param = BoolParam('Use Noise', False)
    seed: Param = IntParam('Seed', 0, 0, 100)


if __name__ == '__main__':
    class Test(DockingViewer):
        def setup_state(self):
            self.state = State()
            self.state.seed = 0
            self.state_last = None
            self.cache = {}
            self.export_img = False
            self.clear_cache = False

        def compute(self):
            if self.state_last != self.state:
                self.state_last = deepcopy(self.state)
            key = str(self.state)
            if key not in self.cache:
                self.cache[key] = self.process()
            
            if self.export_img:
                self.export_image(self.cache[key])
                self.export_img = False

            return self.cache[key]

        def state_to_config(self):
            config = dict()
            config['input'] = self.state.input_image.value
            config['img_width'] = self.state.img_width
            config['layers_to_use'] = [self.state.layers_to_use.value]
            if self.state.channel_to_use != -1:
                config['channel_to_use'] = self.state.channel_to_use
            config['model_name'] = self.state.model_name
            config['pretrained_weights'] = self.state.pretrained_weights
            config['pyramid_size'] = self.state.pyramid_size
            config['pyramid_ratio'] = self.state.pyramid_ratio
            config['num_gradient_ascent_iterations'] = self.state.num_gradient_ascent_iterations
            config['lr'] = self.state.lr
            config['should_display'] = False
            config['spatial_shift_size'] = self.state.spatial_shift_size
            config['smoothing_coefficient'] = self.state.smoothing_coefficient
            config['use_noise'] = self.state.use_noise
            config['seed'] = self.state.seed
            config['dump_dir'] = os.path.join(OUT_IMAGES_PATH, f'{config["model_name"]}_{config["pretrained_weights"]}')
            return config
    
        def preview_callback(self, input_img, output_img):
            self.update_image(np.concatenate((input_img, output_img), axis=1))

        def process(self):
            config = self.state_to_config()
            print("Before", self.state.pyramid_ratio)
            input_img, output_img = deep_dream_static_image(config, callback=self.preview_callback)
            print("After", self.state.pyramid_ratio)
            img = np.concatenate((input_img, output_img), axis=1)

            return img

        @dockable
        def toolbar(self):
            if self.export_img:
                imgui.text('Exporting...')
            elif imgui.button('Export image'):
                self.export_img = True
            
            imgui.same_line()
            if self.clear_cache:
                imgui.text('Clearing cache...')
                self.cache = {}
                self.clear_cache = False
            elif imgui.button('Clear cache'):
                self.clear_cache = True


            # draw_container(self.state)

            for _, p in self.state:
                if isinstance(p, Param) and p.active:
                    imgui.set_next_item_width(self.ui_scale * 150)
                    p.draw()
    

        def export_image(self, img: np.ndarray):
            img = Image.fromarray(np.uint8(img * 255))
            print(self.state.pyramid_ratio)
            img.save(os.path.join(OUT_IMAGES_PATH, f'{self.state.input_image.value}_{self.state.model_name}_{self.state.pretrained_weights}_{self.state.layers_to_use.value}_{self.state.channel_to_use}_{self.state.pyramid_size}_{self.state.pyramid_ratio}_{self.state.num_gradient_ascent_iterations}_{self.state.lr}_{self.state.spatial_shift_size}_{self.state.smoothing_coefficient}_{self.state.use_noise}_{self.state.seed}.png'))



    _ = Test('Deep Dream Viewer')
    print('Done')