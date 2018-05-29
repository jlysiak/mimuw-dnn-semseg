import json

_DEFAULT_CONF_PATH = "net_conf.json"

# ======== DEFAULT CONFIGURATON
# It's possible to annotate layers 
_DEFAULT_CONF = {
    "DATASET_DIR" : "training",
    "CKPT_DIR" : "checkpoint",
    "CKPT_NAME" : "semseg.ckpt",
    "LOG_PATH" : "/tmp/semseg-train.log",
    "TB_DIR" : "/tmp/semseg-tb-log",
    "IMAGES_DIR" : "images",
    "LABELS_DIR" : "labels_plain",
    "VALID_DS_FRAC" : 0.05,
    "VALID_DS_FILE_LIMIT": 500,
    "DS_FRAC": 1.0,
    "LEARNING_RATE" : 0.0005,
    "DEVICE": "CPU",
    "DATA_FORMAT": "NHWC",
    "BATCH_SZ" : 16,
    "INPUT_SZ" : 256,
    "TIME_LIMIT": "11:50:00", 
    "EPOCHS" : 1,
    "VALIDATION_IVAL" : 5000,
   
    # Dataset augumentation
    "RANDOM_CROPS" : 3,
    "CENTRAL_CROPS" : 3,

    "ARCHITECTURE": [
        # input -> 256 x 256 x 3
        ['conv', 32],
        ['bnorm'],
        ['relu'],

        ['conv', 64],
        ['bnorm'],
        ['relu'], 

        ['conv', 64],
        ['bnorm'],
        ['relu:0'], 

        ['pool'],   # 128 x 128 x 64

        ['conv', 128],
        ['bnorm'],
        ['relu'],  

        ['conv'],
        ['bnorm'],
        ['relu:1'], 

        ['pool'],   # 64 x 64 x 128
        
        ['conv'],
        ['bnorm'],
        ['relu:3'], 
        
        ['conv', 256],
        ['bnorm'],
        ['relu:2'], 

        ['pool'],   # 32 x 32 x 256
        
        ['conv', 512],
        ['bnorm'],
        ['relu'],

        ['conv'],
        ['bnorm'],
        ['relu'],

        ['pool'],   # 16 x 16 x 512

        ['conv'],
        ['bnorm'],
        ['relu'], 

        ['conv'],
        ['bnorm'],
        ['relu:4'], # 16 x 16 x  512

        ['pool'],
        
        ['conv', 1024],
        ['bnorm'],
        ['relu'],   # 8 x 8 x 1024

        ['resize', 16], # 16 x 16 x 1024
        ['conv', 512, 1], # 16 x 16 x 512
        ['concat', 4], # 16 x 16 x 1024

        ['conv', 512, 1], # 16 x 16 x 512
        ['bnorm'],
        ['relu'],
        
        ['conv', 256, 1], # 16 x 16 x 256
        ['bnorm'],
        ['relu'],

        ['resize', 64], # 64 x 64 x 256
        ['concat', 3], # 64 x 64 x (256+128)
        ['conv', 128, 1],
        ['bnorm'],
        ['relu'],
        
        ['conv', 128, 1],
        ['bnorm'],
        ['relu'],
        
        ['conv', 64, 1],
        ['bnorm'],
        ['relu'],

        ['resize', 256], # 256 x 256 x 64
        ['conv', 64, 1],
        ['bnorm'],
        ['relu'],
        ['concat', 0], # 256 x 256 x 128
        ['conv', 128, 1], 
        ['bnorm'],
        ['relu'],

        ['conv', 66, 1] # 256 x 256 x 66 <-= logits
    ],
    
    # Validation phase configuration
    "VALIDATION": {
        # np.linspace(0.5, 1, CENTRAL_CROPS)
        "CENTRAL_CROPS" : 3,            
        # Auxilary crops used in validation pipe data augumentation 
        # List of normalized coordinates [x1, y1, x2, y2]
        "AUX_CROPS" : [
            [0, 0, 0.5, 0.5],
            [0.5, 0, 1, 0.5],
            [0, 0.5, 0.5, 1],
            [0.5, 0.5, 1, 1],
            [0.2, 0.1, 0.9, 0.7],
            [0.1, 0.3, 0.6, 0.8]
        ],
        "FLIP_LR" : True
    }
}


class DEFAULT_VALUES:
    DEFAULT_CONF_PATH = _DEFAULT_CONF_PATH
    DEFAULT_CONF = _DEFAULT_CONF


def create_default(path):
    """
    Create default network configuration.
    """
    s = json.dumps(_DEFAULT_CONF, sort_keys=True, indent=4)
    with open(path, 'w') as f:
        f.write(s)


def load_conf(path):
    """
    Load network configuration from JSON file
    """
    with open(path, 'r') as f:
        r = f.read()
    return json.loads(r)

