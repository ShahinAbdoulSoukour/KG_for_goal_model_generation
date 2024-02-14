import os
import time

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PhrasalConstraint, T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, \
    AutoModelForSequenceClassification
from transformers.utils import logging

from g2t_generator.constrain_generate_util import generate
from g2t_generator.data import WebNLGDataset, WebNLGDataLoader


def generate_g2t_data(triples: list[tuple[str, str, str]]):
    triples_grouped_by_subject = dict()
    for triple in triples:
        if triple[0] in triples_grouped_by_subject:
            triples_grouped_by_subject[triple[0]].append([triple[1], triple[2]])
        else:
            triples_grouped_by_subject[triple[0]] = [[triple[1], triple[2]]]

    processed_data = [{
        "id": "Id1",
        "kbs": {
            f"W{i}": [
                subject,
                subject,
                [
                    [
                        pred_obj[0],
                        pred_obj[1]
                    ]
                    for pred_obj in pred_obj_list
                ]
            ]
            for i, (subject, pred_obj_list) in enumerate(triples_grouped_by_subject.items())},
        "text": [
            "",
            ""
        ]}]

    return processed_data


def g2t_generator(triples: list[tuple[str, str, str]], model, tokenizer):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    processed_data = generate_g2t_data(triples)

    args = {
        "model_name": "t5-base",
        "tokenizer_path": "t5-base",
        "dataset": "webnlg20",
        "append_another_bos": True,
        "clean_up_spaces": False,
        "predict_batch_size": 1,
        "max_input_length": 256,
        "max_output_length": 256,
        "max_node_length": 50,
        "max_edge_length": 60,
        "num_workers": 1,
        "cls_start_step": 1,
        "cls_model": "",
        "cls_model_path": "",
        "cls_threshold": .5,
        "num_beams": 5,
        "length_penalty": 1.,
        "promotion_weight": .1,
        "debug": False
    }

    dataset = WebNLGDataset(args, processed_data, tokenizer, "val")
    dataloader = WebNLGDataLoader(args, dataset, "dev")

    cls_tokenizer = ""
    cls_model = ""
    if args["cls_model"] != "":
        cls_tokenizer = AutoTokenizer.from_pretrained(args["cls_model"])
        cls_model = AutoModelForSequenceClassification.from_pretrained(args["cls_model"])
        cls_state_dict = torch.load(args["cls_model_path"])
        cls_model.load_state_dict(cls_state_dict)
        cls_model.to(device)
        cls_model.eval()
    cls_start_step = args["cls_start_step"]
    cls_threshold = args["cls_threshold"]
    promotion_weight = args["promotion_weight"]

    predictions = inference(model, dataloader, tokenizer, args, cls_model, cls_tokenizer, cls_start_step, cls_threshold, promotion_weight)

    return predictions


def inference(model, dev_dataloader, tokenizer, args, cls_model, cls_tokenizer, cls_start_step, cls_threshold,
              promotion_weight, save_predictions=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    predictions = []
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # data_ref = [[x.split("?")[0].strip(" ?.").lower() for x in data_ele['text']]
    #             for data_ele in dev_dataloader.dataset.data]
    # Inference on the test set
    all_cls_score = []
    for i, batch in enumerate(dev_dataloader):
        # batch = [b.to(device) for b in batch]
        b_input_id = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_entity_constraint = batch[9].to(device)
        b_cls_prompts = batch[10]
        constrain_list = []
        for each_c in b_entity_constraint[0]:
            constrain_list.append(PhrasalConstraint([x for x in each_c.cpu().numpy().tolist() if x > 0]))
        if "thu" in args["model_name"]:
            b_node_id = batch[4].to(device)
            b_edge_id = batch[5].to(device)
            b_node_len = batch[6].to(device)
            b_edge_len = batch[7].to(device)
            b_matrix = batch[8].to(device)
            # output of T5/Bart in PLM
            if promotion_weight == 0:
                outputs = model.generate(input_ids=b_input_id, attention_mask=b_attention_mask,
                                         input_node_ids=b_node_id, input_edge_ids=b_edge_id, node_length=b_node_len,
                                         edge_length=b_edge_len, adj_matrix=b_matrix, num_beams=args["num_beams"],
                                         length_penalty=args["length_penalty"],
                                         # constraints=constrain_list if len(constrain_list) > 0 else None,
                                         max_length=args["max_output_length"], early_stopping=True, do_sample=False, )
            else:
                outputs = generate(self=model, input_ids=b_input_id, attention_mask=b_attention_mask,
                                   input_node_ids=b_node_id, input_edge_ids=b_edge_id, node_length=b_node_len,
                                   edge_length=b_edge_len, adj_matrix=b_matrix, num_beams=args["num_beams"],
                                   length_penalty=args["length_penalty"], max_length=args["max_output_length"],
                                   early_stopping=True, do_sample=False, cls_prompt_list=b_cls_prompts,
                                   decode_tokenizer=tokenizer, cls_tokenizer=cls_tokenizer, cls_threshold=cls_threshold,
                                   cls_model=cls_model, cls_start_step=cls_start_step,
                                   cls_promotion_weight=promotion_weight)
        else:
            # output of T5/Bart in PLM
            if promotion_weight == 0:
                outputs = model.generate(input_ids=b_input_id, attention_mask=b_attention_mask,
                                         num_beams=args["num_beams"], length_penalty=args["length_penalty"],
                                         # constraints=constrain_list if len(constrain_list) > 0 else None,
                                         max_length=args["max_output_length"], early_stopping=True, do_sample=False)
            else:
                outputs = generate(self=model, input_ids=b_input_id, attention_mask=b_attention_mask,
                                   num_beams=args["num_beams"], length_penalty=args["length_penalty"],
                                   # constraints=constrain_list if len(constrain_list) > 0 else None,
                                   max_length=args["max_output_length"], early_stopping=True, do_sample=False,
                                   cls_prompt_list=b_cls_prompts, decode_tokenizer=tokenizer,
                                   cls_tokenizer=cls_tokenizer, cls_threshold=cls_threshold, cls_model=cls_model,
                                   cls_start_step=cls_start_step, cls_promotion_weight=promotion_weight)
        # Convert ids to tokens
        ref_id = 0
        for input_, output in zip(batch[0], outputs):
            # trunk content after eos and before bos, especially bart v4 starts with eos
            if tokenizer.bos_token_id is not None and tokenizer.bos_token_id in output:
                start_idx = (output == tokenizer.bos_token_id).nonzero(as_tuple=True)[0][0]
                output = output[start_idx:]
            if tokenizer.eos_token_id in output:
                stop_idx = (output == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0] + 1
                output = output[: stop_idx]
            pred = tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=args["clean_up_spaces"])
            if '[EOS]' in pred:
                pred = pred[: pred.index('[EOS]')]
            if cls_model != "":
                cls_input = [each_cls_prompt[0] + pred.strip(".? ") for each_cls_prompt in b_cls_prompts]
                tokenized_cls = cls_tokenizer(cls_input, truncation=True)
                tokenized_cls["input_ids"] = pad_sequence(
                    [torch.LongTensor(x) for x in tokenized_cls["input_ids"]],
                    padding_value=cls_tokenizer.pad_token_id, batch_first=True).to(device)
                tokenized_cls["attention_mask"] = pad_sequence(
                    [torch.LongTensor(x) for x in tokenized_cls["attention_mask"]], padding_value=0,
                    batch_first=True).to(device)
                cls_output = cls_model(input_ids=tokenized_cls["input_ids"],
                                       attention_mask=tokenized_cls["attention_mask"])
                softmax_cls = torch.softmax(cls_output.logits, dim=1, )
                all_cls_score.extend([float(x[1]) for x in softmax_cls])
                # print([float(x[1]) for x in softmax_cls])
                # predictions.append(pred)
                # if len(b_cls_prompts) > 1:
                #     print(b_cls_prompts)
                #     print(pred)
                # print(len(output), len(pred.split()))
            predictions.append(pred.split("?")[0].strip(" .").lower())
            ref_id += 1
    # Save the generated results
    if save_predictions:
        save_path = os.path.join(args["output_dir"], "{}predictions.txt".format(args["prefix"]))
        with open(save_path, "w") as f:
            for pred in predictions:
                f.write(pred + '\n')
        print("Saved prediction in {}".format(save_path))
    return predictions


if __name__ == "__main__":
    torch_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.set_verbosity_error()
    test_data = [("Flood", "Causes", "Property damage"),
                 ("Railroad damage", "is a type of", "Property damage"),
                 ("Flood", "Causes", "Traffic disruption"),
                 ("River overflow", "Causes", "Flood")]

    default_model = T5ForConditionalGeneration.from_pretrained("Inria-CEDAR/WebNLG20T5B").to(torch_device)
    default_tokenizer = T5Tokenizer.from_pretrained("t5-base")

    for i in range(5):
        start_time = time.time()
        prediction = g2t_generator(test_data, model=default_model, tokenizer=default_tokenizer)[0]

        print(prediction)
        print("--- %s seconds ---" % (time.time() - start_time))
