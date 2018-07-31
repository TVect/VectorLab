'''
Generator
- input: noises (batch_size, noise_dim)
- output: images (batch_size, height, width, channels)
- arthitecture: CNN, DNN
- use deconvolution (transpose convolution) layer in CNN
'''

import tensorflow as tf

class Generator:
    
    def __init__(self):
        self._build()
    
    
    def _build(self):
        self.output_height = 64
        self.output_width = 64
        self.noise_dim = 100
        self.noise_placeholder = tf.placeholder(dtype=tf.float32, 
                                                shape=[None, self.noise_dim], 
                                                name="noise_placeholder")
        
        
        s_h, s_w = self.output_height, self.output_width
        # 2 is stride
        s_h2, s_w2 = int(math.ceil(s_h/2)), int(math.ceil(s_w/2))
        s_h4, s_w4 = int(math.ceil(s_h2/2)), int(math.ceil(s_w2/2))
        s_h8, s_w8 = int(math.ceil(s_h4/2)), int(math.ceil(s_w4/2))
        s_h16, s_w16 = int(math.ceil(s_h8/2)), int(math.ceil(s_w8/2))


