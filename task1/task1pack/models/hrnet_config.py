cfg = {
    'MODEL': {
        'NAME': 'pose_hrnet',
        'INIT_WEIGHTS': True,
        'PRETRAINED': '',
        'NUM_CORNERS': 17,
        'IMAGE_SIZE': [400, 400],
        'HEATMAP_SIZE': [100, 100],
        'EXTRA': {
            'PRETRAINED_LAYERS': ['*'],
            'STEM_INPLANES': 64,
            'FINAL_CONV_KERNEL': 1,

            'STAGE2': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 2,
                'NUM_BLOCKS': [4, 4],
                'NUM_CHANNELS': [32, 64],
                'BLOCK': 'BASIC',
                'FUSE_METHOD': 'SUM',
            },

            'STAGE3': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 3,
                'NUM_BLOCKS': [4, 4, 4],
                'NUM_CHANNELS': [32, 64, 128],
                'BLOCK': 'BASIC',
                'FUSE_METHOD': 'SUM',
            },

            'STAGE4': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 4,
                'NUM_BLOCKS': [4, 4, 4, 4],
                'NUM_CHANNELS': [32, 64, 128, 256],
                'BLOCK': 'BASIC',
                'FUSE_METHOD': 'SUM',
            }
        }
    }
}