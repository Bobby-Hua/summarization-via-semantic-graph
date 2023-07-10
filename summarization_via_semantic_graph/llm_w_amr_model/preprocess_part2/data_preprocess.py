import itertools
import json
import jsonlines
import itertools
from tqdm import tqdm
dataset_name='train'
# processed_fp='/mnt/swordfish-datastore/bobby/summscreen/data/ICSI/icsi_%s_processed.json'%dataset_name
LM_format_fp='/mnt/swordfish-datastore/bobby/summscreen/data/FD/%s.json'%dataset_name
# ep2scene_fp='/mnt/swordfish-datastore/bobby/summscreen/data/FD/AMR/fd_%s_ep_to_scene.json'%dataset_name
import nltk

#%%
from transformers import (
    AutoTokenizer,)
tokenizer = AutoTokenizer.from_pretrained('MingZhong/DialogLED-base-16384')
#%%
# with open(processed_fp,'r') as f:
#     all_scenes=json.load(f)

LM_format_data=[]
with open(LM_format_fp,'r') as f:
    for line in f:
        line=json.loads(line)
        LM_format_data.append(line)

# with open(ep2scene_fp,'r') as f:
#     ep2scene=json.load(f)

#%%
final_data=[]
scene_start=0
for i, ep in enumerate(LM_format_data):
    # scene_end=scene_start+ep2scene[str(i)]
    # scenes=[x['dialogue'] for x in all_scenes[scene_start:scene_end]]
    # scene_start=scene_end
    # epi_src=' '.join(scenes) #use <s> token, the sep_token for beginning of segment. use <u> for beginning of utterance.
    #<s> and <s> appear after the entire segment/sent.
    # epi_src=epi_src.replace('\r\n',' ')
    # epi_src=''+epi_src  #<s> not adding a seg break in the beginning because the tokenizer will add it by default.
    # ep['src']=epi_src.lower()
    ep['src']=ep['src'].replace(':',' <u>')
    final_data.append(ep)
# assert(scene_start==sum(ep2scene.values()))
# assert(len(ep2scene)==len(LM_format_data))
#%%
with jsonlines.open(LM_format_fp[:-5]+'.ori_src.json','w') as writer:
    writer.write_all(final_data)

#%%

#%%