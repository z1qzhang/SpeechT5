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
data_tsv = '/home/ziqzhang/dataset-diskd/LibriLM/tri6bsemi_decode_new/dev_clean.tsv'
with open(data_tsv, 'r') as f:
    wav_root = f.readline().strip()
    wav_files = [os.path.join(wav_root, l.split()[0]) for l in f]

normalize = checkpoint['cfg']['task']['normalize']  # False for base model, True for large model
representation = None
for wav_file in tqdm(wav_files, mininterval=1.0):
    wav, sr = sf.read(wav_file)
    assert sr == 16000
    wav_input_16khz = torch.from_numpy(wav).unsqueeze(0).float().cuda()

    if normalize:
        wav_input_16khz = F.layer_norm(wav_input_16khz[0], wav_input_16khz[0].shape).unsqueeze(0)

    # extract the representation of each layer
    output_layer = model.cfg.encoder_layers + model.cfg.text_transformer.encoder.layers
    rep, layer_results = model.extract_features(wav_input_16khz, output_layer=output_layer, ret_layer_results=True)[0]

    layer_results = torch.cat(layer_results[6:], dim=1) # [N, 7, 768]
    layer_results = layer_results.cpu().numpy()
    if representation is not None:
        representation = np.concatenate([representation, layer_results], axis=0)
    else:
        representation = layer_results
    
    if representation.shape[0] > MAX_SAMPLE:
        break

data_saved = '/home/ziqzhang/data/speechlm_tri6bsemi_new/exp/pretrain/base_speechlmp_32gpu_1accum/representations/dev_clean.unit_encoder0-6.speech.npy'
if not os.path.exists(os.path.dirname(data_saved)):
    os.system(f"mkdir -p {os.path.dirname(data_saved)}")
np.save(data_saved, representation)
