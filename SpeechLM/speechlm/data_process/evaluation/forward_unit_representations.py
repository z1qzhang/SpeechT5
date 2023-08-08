import os
from tqdm import tqdm
import soundfile as sf
import numpy as np

import torch
import torch.nn.functional as F
from SpeechLM import SpeechLMConfig, SpeechLM

checkpoint = torch.load('/home/ziqzhang/data/speechlm_tri6bsemi_new/exp/pretrain/base_speechlmp_32gpu_1accum/checkpoint_298_400000.pt')
cfg = SpeechLMConfig(checkpoint['cfg']['model'])
model = SpeechLM(cfg)
try:
    model.load_state_dict(checkpoint['model'])
except Exception as err:
    print(err)
    model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
model.cuda()

MAX_SAMPLE = 40000
data_tsv = '/home/ziqzhang/dataset-diskd/LibriLM/tri6bsemi_decode_new/dev_clean.phn'
with open(data_tsv, 'r') as f:
    unit_samples = [4 + np.array(list(map(int, l.strip().split()))[::2]) for l in f]

normalize = checkpoint['cfg']['task']['normalize']  # False for base model, True for large model
representation = None
for unit_sample in tqdm(unit_samples, mininterval=1.0):
    unit_input_50hz = torch.from_numpy(unit_sample).unsqueeze(0).long().cuda()

    # extract the representation of each layer
    with torch.no_grad():
        encoder_out = model.unit_encoder(unit_input_50hz, return_all_hiddens=True)

    layer_results = torch.cat(encoder_out['encoder_states'], dim=1) # [N, 7, 768]
    layer_results = layer_results.cpu().numpy()
    if representation is not None:
        representation = np.concatenate([representation, layer_results], axis=0)
    else:
        representation = layer_results
    
    if representation.shape[0] > MAX_SAMPLE:
        break

data_saved = '/home/ziqzhang/data/speechlm_tri6bsemi_new/exp/pretrain/base_speechlmp_32gpu_1accum/representations/dev_clean.unit_encoder0-6.unit.npy'
if not os.path.exists(os.path.dirname(data_saved)):
    os.system(f"mkdir -p {os.path.dirname(data_saved)}")
np.save(data_saved, representation)
