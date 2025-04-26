from pyviewer.docking_viewer import DockingViewer, dockable
from imgui_bundle import imgui
import numpy as np
import argparse
import os
from pyviewer.params import *  # type: ignore
from examples.deepdream import deep_dream_static_image, save_and_maybe_display_image, SupportedModels, SupportedPretrainedWeights, OUT_IMAGES_PATH, INPUT_DATA_PATH
from enum import Enum
from copy import deepcopy
import cv2 as cv

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

@strict_dataclass
class State(ParamContainer):
    img_width: Param = IntParam('Image Width', 224, 1, 1000)
    layers_to_use: Param = EnumParam('Layers to Use', VGG16Layers.relu4_3, VGG16Layers)
    channel_to_use: Param = IntParam('Channel to Use', -1, -1, 32)
    model_name: Param = EnumParam('Model Name', SupportedModels.VGG16_EXPERIMENTAL.name, [m.name for m in SupportedModels])
    pretrained_weights: Param = EnumParam('Pretrained Weights', SupportedPretrainedWeights.IMAGENET.name, [pw.name for pw in SupportedPretrainedWeights])
    pyramid_size: Param = IntParam('Pyramid Size', 1, 1, 10)
    pyramid_ratio: Param = FloatParam('Pyramid Ratio', 1.8, 1.0, 2.0)
    num_gradient_ascent_iterations: Param = IntParam('Gradient Ascent Iterations', 10, 1, 100)
    lr: Param = FloatParam('Learning Rate', 0.09, 0.01, 1.0)
    spatial_shift_size: Param = IntParam('Spatial Shift Size', 32, 0, 100)
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

        def compute(self):
            if self.state_last != self.state:
                self.state_last = deepcopy(self.state)
            key = str(self.state)
            if key not in self.cache:
                self.cache[key] = self.process(self.state)
            return self.cache[key]

        def state_to_config(self, state: State):
            config = dict()
            config['input'] = 'n01443537_goldfish.JPEG'
            config['img_width'] = state.img_width
            config['layers_to_use'] = [state.layers_to_use.value]
            if state.channel_to_use != -1:
                config['channel_to_use'] = state.channel_to_use
            config['model_name'] = state.model_name
            config['pretrained_weights'] = state.pretrained_weights
            config['pyramid_size'] = state.pyramid_size
            config['pyramid_ratio'] = state.pyramid_ratio
            config['num_gradient_ascent_iterations'] = state.num_gradient_ascent_iterations
            config['lr'] = state.lr
            config['should_display'] = False
            config['spatial_shift_size'] = state.spatial_shift_size
            config['smoothing_coefficient'] = state.smoothing_coefficient
            config['use_noise'] = state.use_noise
            config['seed'] = state.seed
            config['dump_dir'] = os.path.join(OUT_IMAGES_PATH, f'{config["model_name"]}_{config["pretrained_weights"]}')
            return config

        def process(self, state: State):
            config = self.state_to_config(state)
            input_img, output_img = deep_dream_static_image(config)

            img = np.concatenate((input_img, output_img), axis=1)

            return img

        @dockable
        def toolbar(self):
            draw_container(self.state)


    _ = Test('Deep Dream Viewer')
    print('Done')