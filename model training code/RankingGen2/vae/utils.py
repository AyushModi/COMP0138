import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from squad_utils import (convert_examples_to_features_answer_id,
                         convert_examples_to_harv_features,
                         read_squad_examples)


def get_squad_data_loader(tokenizer, file, shuffle, args,query_list):
    examples = read_squad_examples(file, is_training=True, debug=args.debug)
    features = convert_examples_to_features_answer_id(examples,
                                                    tokenizer=tokenizer,
                                                    max_seq_length=args.max_c_len,
                                                    max_query_length=args.max_q_len,
                                                    max_ans_length=args.max_q_len,
                                                    doc_stride=128,
                                                    is_training=True)
    all_index_ids = torch.tensor([f.unique_id for f in features], dtype=torch.long)
    all_q_ids = torch.tensor([f.q_ids for f in features], dtype=torch.long)
    # unique_qa_ids = np.unique([f.unique_id for f in features])
    #chosen_qa_ids = np.random.choice(unique_qas, size=15, replace=False)
    chosen_q_ids = []
    for query in query_list:
        index = query[0]
        index_in_all = (all_index_ids == int(index)).nonzero(as_tuple=True)[0] # Reference: https://stackoverflow.com/questions/47863001/how-pytorch-tensor-get-the-index-of-specific-value
        chosen_q_ids.append((all_q_ids[index_in_all,:],index))
    all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
    all_tag_ids = torch.tensor([f.tag_ids for f in features], dtype=torch.long)
    all_a_ids = (all_tag_ids != 0).long()
    all_start_positions = torch.tensor([f.noq_start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.noq_end_position for f in features], dtype=torch.long)
    data = TensorDataset(all_index_ids,all_c_ids,all_a_ids, all_start_positions, all_end_positions)
    data_loader = DataLoader(data, shuffle=False, batch_size=args.batch_size)

    return data_loader,examples, chosen_q_ids

def get_harv_data_loader(tokenizer, file, shuffle, ratio, args):
    examples = read_squad_examples(file, is_training=True, debug=args.debug)
    random.shuffle(examples)
    num_ex = int(len(examples) * ratio)
    examples = examples[:num_ex]
    features = convert_examples_to_harv_features(examples,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=args.max_c_len,
                                                 max_query_length=args.max_q_len,
                                                 doc_stride=128,
                                                 is_training=True)
    all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_c_ids)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=args.batch_size)

    return features, dataloader

def batch_to_device(batch, device):
    batch = (b.to(device) for b in batch)
    c_ids, q_ids, a_ids, start_positions, end_positions = batch

    c_len = torch.sum(torch.sign(c_ids), 1)
    max_c_len = torch.max(c_len)
    c_ids = c_ids[:, :max_c_len]
    a_ids = a_ids[:, :max_c_len]

    q_len = torch.sum(torch.sign(q_ids), 1)
    max_q_len = torch.max(q_len)
    q_ids = q_ids[:, :max_q_len]

    return c_ids, q_ids, a_ids, start_positions, end_positions
