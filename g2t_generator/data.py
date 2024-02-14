import copy
import json
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


def safe_setting(matrix, x_start, x_end, y_start, y_end):
    if x_start >= matrix.shape[0] or y_start >= matrix.shape[0]:
        return

    matrix[x_start:min(matrix.shape[0], x_end), y_start:min(matrix.shape[1], y_end)] = 1
    return


class WebNLGDataLoader(DataLoader):

    def __init__(self, args, dataset, mode):
        if mode == "train":
            sampler = RandomSampler(dataset)
            batch_size = args["train_batch_size"]
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args["predict_batch_size"]
        super(WebNLGDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size,
                                               num_workers=args["num_workers"])


# Downstream dataset (webnlg17, webquestions, pathquestions)
class WebNLGDataset(Dataset):
    def __init__(self, args, data, tokenizer, mode):
        self.tokenizer = tokenizer
        self.data = data

        if args["debug"]:
            self.data = self.data[:1000]
        assert type(self.data) == list
        assert all(["id" in d for d in self.data]), self.data[0].keys()
        if type(self.data[0]["id"]) == int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])

        self.args = args
        self.data_type = mode

        self.metric = "BLEU"

        self.head_ids, self.rel_ids, self.tail_ids = self.tokenizer.encode(' [head]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [relation]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [tail]', add_special_tokens=False)
        self.graph_ids, self.text_ids = self.tokenizer.encode(' [graph]', add_special_tokens=False), \
                                        self.tokenizer.encode(' [text]', add_special_tokens=False)

        if "bart" in self.args["model_name"]:
            self.mask_token = self.tokenizer.mask_token
            self.mask_token_id = self.tokenizer.mask_token_id
        else:
            self.mask_token = self.tokenizer.additional_special_tokens[0]
            self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0])

        if "bart" in self.args["model_name"]:
            if self.args["append_another_bos"]:
                self.add_bos_id = [self.tokenizer.bos_token_id] * 2
            else:
                self.add_bos_id = [self.tokenizer.bos_token_id]
        else:
            self.add_bos_id = []
        self.prevent_predicates = ["class name", "entity name", "domains", "literal name", "sentence category",
                                   "Sentence Data Source", ]
        self.property_predicates = ["class name", "entity name", "literal name"]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def linearize_v2(entity, entity_change, head_ids, rel_ids, tail_ids,
                     relation_change, cnt_edge, adj_matrix):
        # string_label: encoder ids
        # string_label_tokens: encoder tokens

        if len(entity[0]) == 0:
            return [], '', [], [], cnt_edge, adj_matrix
        nodes, edges = [], []
        string_label = copy.deepcopy(head_ids)
        string_label_tokens = ' [head]'
        nodes.extend([-1] * len(string_label))
        edges.extend([-1] * len(string_label))

        string_label += entity_change[entity[0]][0]
        string_label_tokens += ' {}'.format(entity[0])
        nodes.extend([entity_change[entity[0]][1]] * len(entity_change[entity[0]][0]))
        edges.extend([-1] * len(entity_change[entity[0]][0]))

        for rel in entity[2]:
            if len(rel[0]) != 0 and len(rel[1]) != 0:
                rel_label = relation_change[rel[0]]
                rel_label_token = copy.deepcopy(rel[0])
                words_label = rel_ids + rel_label + tail_ids + entity_change[rel[1]][0]
                words_label_tokens = ' [relation] {} [tail] {}'.format(rel_label_token, rel[1])
                nodes.extend(
                    [-1] * (len(rel_ids) + len(rel_label) + len(tail_ids)) + [entity_change[rel[1]][1]] * len(
                        entity_change[rel[1]][0]))
                edges.extend([-1] * len(rel_ids) + [cnt_edge] * len(rel_label) + [-1] * (
                        len(tail_ids) + len(entity_change[rel[1]][0])))
                if entity_change[entity[0]][1] < len(adj_matrix) and entity_change[rel[1]][1] < len(adj_matrix):
                    adj_matrix[entity_change[entity[0]][1]][entity_change[rel[1]][1]] = cnt_edge

                cnt_edge += 1
                string_label += words_label
                string_label_tokens += words_label_tokens
        assert len(string_label) == len(nodes) == len(edges)
        return string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix

    @staticmethod
    def get_all_entities_per_sample(mark_entity_number, mark_entity, entry):
        text_entity = set()
        text_relation = set()
        for entity_id in mark_entity_number:
            entity = entry['kbs'][entity_id]
            if len(entity[0]) == 0:
                continue
            for rel in entity[2]:
                if len(rel[0]) != 0 and len(rel[1]) != 0:
                    text_relation.add(rel[0])
                    text_entity.add(rel[1])

        text_entity_list = list(text_entity)
        text_relation_list = list(text_relation)
        for entity_ele in mark_entity:
            if entity_ele in text_entity_list:
                text_entity_list.remove(entity_ele)

        return text_entity_list, text_relation_list

    def get_change_per_sample(self, mark_entity, text_entity, text_relation):
        # during fine-tuning, we don't mask entities or relations
        ent_change = {}
        total_entity = mark_entity + text_entity

        for ent_id in range(len(total_entity)):
            entity_toks = self.tokenizer.encode(" {}".format(total_entity[ent_id]), add_special_tokens=False)
            ent_change[total_entity[ent_id]] = [entity_toks, ent_id]

        # relation change only includes the relation tokens and ids
        rel_change = {}
        for rel_id in range(len(text_relation)):
            rel_change[text_relation[rel_id]] = self.tokenizer.encode(' {}'.format(text_relation[rel_id]),
                                                                      add_special_tokens=False)

        return ent_change, rel_change

    def truncate_pair_ar(self, a, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add_bos_id + graph_ids + a + text_ids + b + eos_token_id
        length_a_b = self.args["max_input_length"] - len(add_bos_id) - len(graph_ids) - len(text_ids) - 1
        if len(a) > length_a_b:
            a = a[:length_a_b]
            node_ids = node_ids[:length_a_b]
            edge_ids = edge_ids[:length_a_b]
        input_ids = add_bos_id + graph_ids + a + text_ids + [self.tokenizer.eos_token_id]
        input_node_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + node_ids + [-1] * (len(text_ids) + 1)
        input_edge_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + edge_ids + [-1] * (len(text_ids) + 1)
        attn_mask = [1] * len(input_ids) + [0] * (self.args["max_input_length"] - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (self.args["max_input_length"] - len(input_ids))
        input_node_ids += [-1] * (self.args["max_input_length"] - len(input_node_ids))
        input_edge_ids += [-1] * (self.args["max_input_length"] - len(input_edge_ids))
        assert len(input_ids) == len(attn_mask) == self.args["max_input_length"] == len(input_node_ids) == len(
            input_edge_ids)
        return input_ids, attn_mask, input_node_ids, input_edge_ids

    def ar_prep_data(self, answers, questions, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add bos and eos
        decoder_label_ids = copy.deepcopy(answers)
        if len(decoder_label_ids) > self.args["max_output_length"] - len(add_bos_id) - 1:
            decoder_label_ids = decoder_label_ids[:(self.args["max_output_length"] - len(add_bos_id) - 1)]
        decoder_label_ids = add_bos_id + decoder_label_ids + [self.tokenizer.eos_token_id]
        decoder_attn_mask = [1] * len(decoder_label_ids) + [0] * (self.args["max_output_length"] - len(decoder_label_ids))
        decoder_label_ids += [self.tokenizer.pad_token_id] * (self.args["max_output_length"] - len(decoder_label_ids))
        # decoder_label_ids += [self.tokenizer.eos_token_id] * (self.args["max_output_length - len(decoder_label_ids))
        assert len(decoder_label_ids) == self.args["max_output_length"] == len(decoder_attn_mask)

        input_ids, input_attn_mask, input_node_ids, input_edge_ids = self.truncate_pair_ar(questions, add_bos_id,
                                                                                           graph_ids, text_ids,
                                                                                           node_ids, edge_ids)

        return input_ids, input_attn_mask, decoder_label_ids, decoder_attn_mask, input_node_ids, input_edge_ids

    def __getitem__(self, idx):

        entry = self.data[idx]

        entities = []
        for _ in entry['kbs']:
            entities.append(_)

        #  dataloader for entity names
        entity_names = set()
        entity_constrain_len = 0
        for each_triple in entry['kbs']:
            for each_po in entry['kbs'][each_triple][2]:
                if " : " in each_po[1] and 'entity name' == each_po[0]:
                    p_and_o = each_po[1].split(" : ")
                    entity_names.add(p_and_o[0].lower())
        entity_constrain_list = []
        if self.data_type != "train":
            for each in entity_names:
                entity_constrain_list.append(self.tokenizer.encode("" + each, add_special_tokens=False))
                con_len = len(entity_constrain_list[-1])
                if con_len > entity_constrain_len:
                    entity_constrain_len = con_len
        np_constrain_entity = np.zeros([len(entity_constrain_list), entity_constrain_len], dtype=int)
        for idx in range(0, len(entity_constrain_list)):
            np_constrain_entity[idx, 0:len(entity_constrain_list[idx])] = entity_constrain_list[idx]

        all_constraint_prompt = set()
        if "grail" in self.args["dataset"]:
            triple_list = []
            entity_dict = {}
            for each_triple in entry['kbs']:
                tmp_subject = entry["kbs"][each_triple][0]
                for each_po in entry['kbs'][each_triple][2]:
                    if each_po[0] not in self.prevent_predicates:
                        if "function : " in each_po[0] and "." not in each_po[0] and "_" not in each_po[0]:
                            function_and_name = each_po[0].lower().split(":")
                            tmp_predicate = function_and_name[1] + " : " + function_and_name[0]
                            if ">=" in each_po[0]:
                                tmp_predicate = tmp_predicate.replace(">=", "greater or equal")
                            elif ">" in each_po[0]:
                                tmp_predicate = tmp_predicate.replace(">", "greater")
                            elif "<=" in each_po[0]:
                                tmp_predicate = tmp_predicate.replace("<=", "less or equal")
                            elif "<" in each_po[0]:
                                tmp_predicate = tmp_predicate.replace("<", "less")
                            elif "arg" in each_po[0]:
                                tmp_predicate = tmp_predicate.replace("argmax", "maximal")
                                tmp_predicate = tmp_predicate.replace("argmin", "minimal")
                        else:
                            tmp_predicate = each_po[0].lower()
                        tmp_predicate = tmp_predicate.replace(" : ", " < name > ")
                        tmp_object = each_po[1]
                        triple_list.append([tmp_subject, tmp_predicate, tmp_object])
                    if each_po[0] in self.property_predicates:
                        tmp_property = each_po[1].lower().replace(" : ", " < class > ")
                        if tmp_subject not in entity_dict:
                            entity_dict[tmp_subject] = tmp_property
                        elif tmp_property not in entity_dict[tmp_subject]:
                            entity_dict[tmp_subject] += tmp_property
            for each_triple in triple_list:
                if each_triple[0] in entity_dict:
                    each_triple[0] = entity_dict[each_triple[0]]
                else:
                    each_triple[0] = each_triple[0].lower().replace(" : ", " < class > ")
                    each_triple[0] = each_triple[0].lower().replace("^^", " < class > ")
                if each_triple[2] in entity_dict:
                    each_triple[2] = entity_dict[each_triple[2]]
                else:
                    each_triple[2] = each_triple[2].lower().replace(" : ", " < class > ")
                    each_triple[2] = each_triple[2].lower().replace("^^", " < class > ")
                all_constraint_prompt.add("predicate: " + each_triple[1] + ", subject: " + each_triple[0] + ", object: "
                                          + each_triple[2] + ", sentence: ")
                # all_constraint_prompt.add(tmp_predicate)
        elif "webnlg17" in self.args["dataset"] or "dart" in self.args["dataset"]:
            for each_relation in entry["kbs"]:
                subject_name = entry["kbs"][each_relation][0].lower()
                for each_po in entry["kbs"][each_relation][2]:
                    predicate_name = each_po[0]
                    if predicate_name not in self.prevent_predicates:
                        object_name = each_po[1].lower()
                        all_constraint_prompt.add("predicate: " + predicate_name + ", subject: " + subject_name +
                                                  ", object: " + object_name + ", sentence: ")
        elif "spq" in self.args["dataset"]:
            # spq only has one triple so take first one
            possible_entity_list = set()
            relation_id = 0
            subject_id = ""
            spq_predicate = ""
            for each_relation in entry['kbs']:
                if relation_id == 0:
                    # because each simple_question only has one predicate useful
                    subject_id = entry["kbs"][each_relation][0]
                    spq_predicate = entry["kbs"][each_relation][2][0][0]
                elif entry["kbs"][each_relation][0] == subject_id:
                    for each_po in entry["kbs"][each_relation][2]:
                        if each_po[0] == "type . object . name" or each_po[0] == "common . topic . alias":
                            possible_entity_list.add(each_po[1])
                relation_id += 1
            possible_entity_list = ", ".join(possible_entity_list)
            all_constraint_prompt.add("predicate: " + spq_predicate +
                                      ", subject: " + possible_entity_list + ", sentence: ")
        constrain_prompt_list = []
        constrain_prompt_len = 0
        if self.data_type != "train":
            for each in all_constraint_prompt:
                constrain_prompt_list.append(each)
                con_len = len(constrain_prompt_list[-1])
                if con_len > constrain_prompt_len:
                    constrain_prompt_len = con_len
        # np_constrain_predicate = np.zeros([len(constrain_prompt_list), constrain_prompt_len])
        # for idx in range(0, len(constrain_prompt_list)):
        #     np_constrain_predicate[idx, 0:len(constrain_prompt_list[idx])] = constrain_prompt_list[idx]
        # np_constrain_predicate = np.array(constrain_prompt_list)
        strings_label = []
        node_ids = []
        edge_ids = []
        strings_label_tokens = ''
        # mark_entity: entities with KB numbers which are important for this task
        # text_entity: entities without KB numbers but only with text, which are less important
        mark_entity = [entry['kbs'][ele_entity][0] for ele_entity in entities]
        mark_entity_number = entities
        text_entity, text_relation = self.get_all_entities_per_sample(mark_entity_number, mark_entity, entry)
        entity_change, relation_change = self.get_change_per_sample(mark_entity, text_entity, text_relation)
        total_entity = mark_entity + text_entity
        adj_matrix = [[-1] * (self.args["max_node_length"] + 1) for _ in range(self.args["max_node_length"] + 1)]
        cnt_edge = 0
        if 'title' in entry:
            entity = self.knowledge[entry['title_kb_id']]
            string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = self.linearize_v2(
                entity,
                entity_change,
                self.head_ids,
                self.rel_ids, self.tail_ids,
                relation_change, cnt_edge, adj_matrix)
            strings_label += string_label
            strings_label_tokens += string_label_tokens

        for i, entity_id in enumerate(entities):
            entity = entry['kbs'][entity_id]

            string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = self.linearize_v2(
                entity,
                entity_change,
                self.head_ids,
                self.rel_ids, self.tail_ids,
                relation_change, cnt_edge, adj_matrix)

            strings_label += string_label
            strings_label_tokens += string_label_tokens
            node_ids += nodes
            edge_ids += edges
        words_label_ids, words_label_tokens, words_input_ids, words_input_tokens = [], '', [], ''
        current_text = random.choice(entry['text'])

        for word in current_text.split():
            word_label_ids = self.tokenizer.encode(" {}".format(word), add_special_tokens=False)
            word_label_tokens = copy.deepcopy(word)

            words_label_ids += word_label_ids
            words_label_tokens += ' ' + word_label_tokens

        input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, input_node_ids_ar, input_edge_ids_ar = \
            self.ar_prep_data(words_label_ids, strings_label, self.add_bos_id, self.graph_ids,
                              self.text_ids, node_ids, edge_ids)

        node_length_ar = max(input_node_ids_ar) + 1
        edge_length_ar = max(input_edge_ids_ar) + 1

        def masked_fill(src, masked_value, fill_value):
            return [src[src_id] if src[src_id] != masked_value and src[src_id] < fill_value else fill_value for src_id
                    in range(len(src))]

        input_node_ids_ar, input_edge_ids_ar = masked_fill(input_node_ids_ar, -1, self.args["max_node_length"]), \
                                               masked_fill(input_edge_ids_ar, -1, self.args["max_edge_length"])

        def masked_fill_matrix(adj_matrix_input, masked_value, fill_value):
            adj_matrix_tmp = copy.deepcopy(adj_matrix_input)
            for a_id in range(len(adj_matrix_tmp)):
                for b_id in range(len(adj_matrix_tmp)):
                    if adj_matrix_tmp[a_id][b_id] == masked_value or adj_matrix_tmp[a_id][b_id] > fill_value:
                        adj_matrix_tmp[a_id][b_id] = fill_value
            return adj_matrix_tmp

        adj_matrix_ar = masked_fill_matrix(adj_matrix, -1, self.args["max_edge_length"])
        assert len(input_ids_ar) == len(attn_mask_ar) == self.args["max_input_length"] == len(input_node_ids_ar) == len(
            input_edge_ids_ar)
        assert len(decoder_label_ids) == len(decoder_attn_mask) == self.args["max_output_length"]
        input_ids_ar = torch.LongTensor(input_ids_ar)
        attn_mask_ar = torch.LongTensor(attn_mask_ar)
        decoder_label_ids = torch.LongTensor(decoder_label_ids)
        decoder_attn_mask = torch.LongTensor(decoder_attn_mask)
        input_node_ids_ar = torch.LongTensor(input_node_ids_ar)
        input_edge_ids_ar = torch.LongTensor(input_edge_ids_ar)
        node_length_ar = torch.LongTensor([node_length_ar])
        edge_length_ar = torch.LongTensor([edge_length_ar])
        adj_matrix_ar = torch.LongTensor(adj_matrix_ar)
        entity_constrain_list = torch.LongTensor(np_constrain_entity)
        # constrain_prompt_list = torch.Tensor(np_constrain_predicate)
        return input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, \
               input_node_ids_ar, input_edge_ids_ar, node_length_ar, edge_length_ar, adj_matrix_ar, \
               entity_constrain_list, constrain_prompt_list
