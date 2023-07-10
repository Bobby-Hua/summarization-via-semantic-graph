from __future__ import absolute_import, division, print_function
from torch.utils.data import Dataset


import logging
import os
import json
import random
import glob
import torch
from tqdm import tqdm
import torch.utils.data
import numpy as np
import pickle

logger = logging.getLogger(__name__)
PAD, UNK = '<PAD>', '<UNK>'
CLS = '<CLS>'
STR, END = '<STR>', '<END>'
SEL, rCLS, TL = '<SELF>', '<rCLS>', '<TL>'

class DoubleDataset(Dataset):

    def __init__(self, dataset1, dataset2):
        self.data_1 = dataset1
        self.data_2 = dataset2
        self.len1 = dataset1.__len__()
        self.len2 = dataset2.__len__()
        assert (self.len1 == self.len2 or self.len2==10) #if length=10, it's a toy dataset.

    def __getitem__(self, index):
        _item1 = self.data_1.__getitem__(index)
        _item2 = self.data_2.__getitem__(index)

        return _item1, _item2

    def __len__(self):
        return min(self.len1, self.len2)

class AMRDataset(torch.utils.data.Dataset):
    def __init__(
            self, amr_filename, vocabs, lex_map, ep_to_scene, train_or_eval, cached_features_dir, target_concept=None, max_seg_count=80, node_selection_loss=False):
        '''
        :param amr_filename: added
        :param vocabs: added
        :param lex_map: added
        :param ep_to_scene: added
        '''
        #AMR
        self.ep_to_scene = json.load(open(ep_to_scene, encoding='utf8'))
        self.max_seg_count=max_seg_count
        print(f'limit max segment count to: {self.max_seg_count}')
        self.node_selection_loss=node_selection_loss

        if self.node_selection_loss and train_or_eval=='train':
            with open(target_concept, "rb") as input_file:
                summary_concepts_list_ep = pickle.load(input_file)

        self.vocabs = vocabs
        cached_train_features_file = os.path.join(cached_features_dir, "cached_features_for_" + train_or_eval + ".pk")
        if os.path.exists(cached_train_features_file):
            print('loading cached AMR features from: ',cached_train_features_file)
            with open(cached_train_features_file, "rb") as f:
                self.amr_batchify, self.gold_nodes_tgt, self.pos_weight = pickle.load(f)
                self.num_ep = len(self.ep_to_scene)
        else:
            os.makedirs(cached_features_dir, exist_ok=True)
            self.data = json.load(open(amr_filename, encoding='utf8'))

            self.max_concept_size = 200
            print("limiting max concept to %d" % self.max_concept_size)
            for scene_gsp in self.data:
                scene_gsp['concept'] = scene_gsp['concept'][:self.max_concept_size]
                scene_gsp['depth'] = scene_gsp['depth'][:self.max_concept_size]
                delete = [key for key in scene_gsp['relation'] if int(key) > self.max_concept_size]
                for key in delete:
                    del scene_gsp['relation'][key]
                for k, v in scene_gsp['relation'].items():
                    delete = [key for key in v if int(key) > self.max_concept_size]
                    for key in delete:
                        del v[key]

            for d in self.data:
                cp_seq, token2idx, idx2token = lex_map.get(d['concept'], vocabs['predictable_token'])
                d['cp_seq'] = cp_seq
                d['token2idx'] = token2idx
                d['idx2token'] = idx2token
            print("Get %d AMR-English pairs from %s" % (len(self.data), amr_filename))

            self.num_ep = len(self.ep_to_scene)
            self.amr_batchify = []
            amr_batch_arr = []
            scene_start = 0
            for scene_idx in tqdm(range(self.num_ep)):
                amr_batch = []
                scene_end = scene_start + self.ep_to_scene[str(scene_idx)]
                max_scene_len = 0
                scene_count = 0
                for amr in self.data[scene_start:scene_end]:

                    max_scene_len = max(len(amr['concept']), max_scene_len)
                    max_scene_len = max(self.max_concept_size, max_scene_len)
                    amr_batch.append(amr)
                    scene_count += 1
                    if scene_count>=self.max_seg_count:
                        break

                self.amr_batchify.append(batchify(amr_batch, self.vocabs))
                amr_batch_arr.append(amr_batch)
                scene_start = scene_end
            if not self.node_selection_loss or train_or_eval !='train':
                self.gold_nodes_tgt=None
                self.pos_weight=None
                print('----------- node selection off ----------------')
            else:
                gold_labels = []
                for index, val in enumerate(amr_batch_arr):
                    gold_label = []
                    for index2, val2 in enumerate(val):
                        gold_label_tmp = []
                        for i in val2['concept'][0:]:
                            if i in summary_concepts_list_ep[index]:
                                gold_label_tmp.append(1)
                            else:
                                gold_label_tmp.append(0)
                        gold_label.append(gold_label_tmp)
                    gold_labels.append(gold_label)
                self.gold_nodes_tgt  = []
                for i in gold_labels:
                    gold_labels_flat_tmp = []
                    for j in i:
                        for k in j:
                            gold_labels_flat_tmp.append(k)
                    self.gold_nodes_tgt.append(np.array(gold_labels_flat_tmp, dtype='int'))

                total_nodes=0
                pos_nodes=0
                for l in self.gold_nodes_tgt:
                    total_nodes+=len(l)
                    pos_nodes+=sum(l)
                self.pos_weight=(total_nodes-pos_nodes)/pos_nodes
            with open(cached_train_features_file, 'wb') as f:
                pickle.dump([self.amr_batchify, self.gold_nodes_tgt, self.pos_weight], f)


    def __len__(self):
        return int(self.num_ep)

    def __getitem__(self, idx):

        concept, concept_char, concept_depth, relation, relation_bank, relation_length, local_idx2token, local_token2idx, cp_seq = self.amr_batchify[idx]
        batch_max_len=concept.size()[1]
        if batch_max_len>self.max_seg_count:

            raise Exception('cached file length longer than the limit')
        else:
            if self.gold_nodes_tgt is not None :
                return concept, concept_char, concept_depth, relation, relation_bank, relation_length, local_idx2token, local_token2idx, cp_seq, self.gold_nodes_tgt[idx]
            else:
                return concept, concept_char, concept_depth, relation, relation_bank, relation_length, local_idx2token, local_token2idx, cp_seq, None


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    snt_batch = [batch[0][0:5]]
    amr_batch = batch[0][5:14]
    for x in zip(*snt_batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    batch_tensors += amr_batch
    return batch_tensors


def get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, 'ckpt-*',"pytorch_model.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, 'ckpt-*',"optim.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    # os.path.basename(output_dir)
    both_set = set([int(fn.split('/')[-2].split('-')[-1]) for fn in fn_model_list]
                   ) & set([int(fn.split('/')[-2].split('-')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def load_and_cache_examples(
        example_file, tokenizer, local_rank, cached_features_file, shuffle=True):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if cached_features_file is not None and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", example_file)

        examples = []
        with open(example_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                examples.append(json.loads(line))
        features = []

        for example in tqdm.tqdm(examples):
            if isinstance(example["src"], list):
                source_tokens = example["src"]
                target_tokens = example["tgt"]
            else:
                source_tokens = tokenizer.tokenize(example["src"])
                target_tokens = tokenizer.tokenize(example["tgt"])
            features.append({
                    "source_ids": tokenizer.convert_tokens_to_ids(source_tokens),
                    "target_ids": tokenizer.convert_tokens_to_ids(target_tokens),
                })

        if shuffle:
            random.shuffle(features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features

def batchify(data, vocabs, unk_rate=0., train=True):
    _conc = ListsToTensor([ [CLS]+x['concept'] for x in data], vocabs['concept'], unk_rate=unk_rate)
    _conc_char = ListsofStringToTensor([ [CLS]+x['concept'] for x in data], vocabs['concept_char'])
    _depth = ListsToTensor([ [0]+x['depth'] for x in data])

    all_relations = dict()
    cls_idx = vocabs['relation'].token2idx(CLS)
    rcls_idx = vocabs['relation'].token2idx(rCLS)
    self_idx = vocabs['relation'].token2idx(SEL)
    pad_idx = vocabs['relation'].token2idx(PAD)
    all_relations[tuple([pad_idx])] = 0
    all_relations[tuple([cls_idx])] = 1
    all_relations[tuple([rcls_idx])] = 2
    all_relations[tuple([self_idx])] = 3
    _relation_type = []
    loop_data(data, vocabs, all_relations, _relation_type)
    _relation_type = ArraysToTensor(_relation_type).transpose_(0, 2)

    B = len(all_relations)
    _relation_bank = dict()
    _relation_length = dict()
    for k, v in all_relations.items():
        _relation_bank[v] = np.array(k, dtype=np.int)
        _relation_length[v] = len(k)
    _relation_bank = [_relation_bank[i] for i in range(len(all_relations))]
    _relation_length = [_relation_length[i] for i in range(len(all_relations))]
    _relation_bank = ArraysToTensor(_relation_bank).t_()
    _relation_length = torch.LongTensor(_relation_length)

    local_token2idx = [x['token2idx'] for x in data]
    local_idx2token = [x['idx2token'] for x in data]



    _cp_seq = ListsToTensor([ x['cp_seq'] for x in data], vocabs['predictable_token'], local_token2idx)


    ret = {
        'concept': _conc,
        'concept_char': _conc_char,
        'concept_depth': _depth,
        'relation': _relation_type,
        'relation_bank': _relation_bank,
        'relation_length': _relation_length,
        'local_idx2token': local_idx2token,
        'local_token2idx': local_token2idx,
        'cp_seq': _cp_seq,
    }
    return _conc,_conc_char,_depth,_relation_type,_relation_bank,_relation_length,local_idx2token,local_token2idx,_cp_seq

def ListsofStringToTensor(xs, vocab, max_string_len=20):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [PAD]*(max_len -len(x))
        zs = []
        for z in y:
            z = list(z[:max_string_len])
            zs.append(vocab.token2idx([STR]+z+[END]) + [vocab.padding_idx]*(max_string_len - len(z)))
        ys.append(zs)

    data = torch.LongTensor(ys).transpose(0, 1).contiguous()
    return data

def ArraysToTensor(xs):
    "list of numpy array, each has the same demonsionality"
    x = np.array([ list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis = 0))
    data = np.zeros(shape, dtype=np.int)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)
        slices = tuple([slice(i, i+1)]+[slice(0, x) for x in slicing_shape])
        data[slices] = x
        tensor = torch.from_numpy(data).long()
    return tensor

def ListsToTensor(xs, vocab=None, local_vocabs=None, unk_rate=0.):
    pad = vocab.padding_idx if vocab else 0

    def toIdx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if random.random() < unk_rate:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.token2idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = toIdx(x, i) + [pad]*(max_len-len(x))
        ys.append(y)
    data = torch.LongTensor(ys).t_().contiguous()
    return data

def loop_data(data, vocabs, all_relations, _relation_type, is_train=True):
    for bidx, x in enumerate(data):
        n = len(x['concept'])
        num_concepts, num_paths = 0, 0
        num_concepts = max(n+1, num_concepts)
        brs = [ [3]+[1]*(n) ] if is_train else [ [[3]]+[[1]]*(n) ]
        for i in range(n):
            i=i+1
            rs = [2] if is_train else [[2]]
            adj_dict = x['relation'][str(i)]
            adj_set = set([int(k) for k in adj_dict.keys()])
            for j in range(n):
                if i == j: # self loop
                    path = [SEL]
                elif j in adj_set:
                    path = adj_dict[str(j)][0]['edge']
                else:
                    path = [PAD]
                path = tuple(vocabs['relation'].token2idx(path))
                rtype = all_relations.get(path, len(all_relations))
                if rtype == len(all_relations):
                    all_relations[path] = len(all_relations)
                if not is_train:
                    num_paths = max(len(rs), num_paths)
                rs.append(rtype if is_train else [rtype])
            if is_train:
                rs = np.array(rs, dtype=np.int)
            brs.append(rs)
        if is_train:
            brs = np.stack(brs)
        _relation_type.append(brs)
        if not is_train:
            return num_concepts, num_paths