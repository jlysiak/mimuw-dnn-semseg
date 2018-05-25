import json

DEFAULT_CONF_PATH = "net_conf.json"

# ======== DEFAULT CONFIGURATON
# It's possible to annotate layers 
DEFAULT_CONF = {
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
    "LEARNING_RATE" : 0.02,
    "DEVICE": "CPU",
    "DATA_FORMAT": "NHWC",
    "BATCH_SZ" : 16,
    "INPUT_SZ" : 256,
    "OUTPUT_SZ" : 256,
   
    # Dataset augumentation
    "RANDOM_CROPS" : 3,
    "CENTRAL_CROPS" : 3,

    "ARCHITECTURE": [
        ['conv', 32],
        ['relu'],

        ['conv', 64],
        ['bnorm'],
        ['relu:0'], # 256 x 256 x 64
        ['pool'],

        ['conv'],
        ['bnorm'],
        ['relu:1'], # 128 x 128 x 64 
        ['pool'],
        
        ['conv', 128],
        ['bnorm'],
        ['relu:2'], # 64 x 64 x 128
        ['pool'],
        
        ['conv'],
        ['bnorm'],
        ['relu:3'], # 32 x 32 x 128
        ['pool'],

        ['conv'],
        ['bnorm'],
        ['relu'], # 16 x 16 x 128

        ['upconv'], # 32 x 32 x 128 
        ['concat', 3], # 32 x 32 x 256
        ['conv', 128], # 32 x 32 x 128
        ['bnorm'],
        ['relu'],

        ['upconv'], # 64 x 64 x 128
        ['concat', 2], # 64 x 64 x 256
        ['conv', 128],
        ['bnorm'],
        ['relu'],
        
        ['upconv', 128], # 128 x 128 x 128
        ['concat', 1], # 128 x 128 x 192
        ['conv', 128], # 128 x 128 x 128
        ['bnorm'],
        ['relu'],

        ['upconv'], # 256 x 256 x 128
        ['concat', 0], # 256 x 256 x 192
        ['conv', 128], # 256 x 256 x 128
        ['bnorm'],
        ['relu'],
        ['conv', 66] # 256 x 256 x 66 <-= logits
    ]
}

# ============= CONFIG FILE OPS

def create_default(path):
    """
    Create default network configuration.
    """
    s = json.dumps(DEFAULT_CONF, indent=4)
    with open(path, 'w') as f:
        f.write(s)


def load_conf(path):
    """
    Load network configuration from JSON file
    """
    with open(path, 'r') as f:
        r = f.read()
    return json.loads(r)


if __name__ == "__main__":
    print(s)
    for k,v in s.items():
        print(k, v)




