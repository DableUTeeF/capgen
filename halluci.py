import os
from typing import Optional
import torch
import argparse
from data import KarpathyDataset, alter_eng, OrderedSet
import json

from transformers import (
    AutoTokenizer,
    ViTImageProcessor,
    AutoConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoModel,
    AutoModelForTokenClassification
)
from PIL import Image
import numpy as np
import evaluate
import nltk
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers.models.gpt2.modeling_gpt2 import GPT2ForTokenClassification
from transformers.trainer_callback import ProgressCallback
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, TokenClassifierOutput
from torch.nn import CrossEntropyLoss
import spacy


@classmethod
def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
):
    kwargs_encoder = {
        argument[len("encoder_"):]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
    }
    kwargs_decoder = {
        argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
    }
    for key in kwargs_encoder.keys():
        del kwargs["encoder_" + key]
    for key in kwargs_decoder.keys():
        del kwargs["decoder_" + key]
    encoder = kwargs_encoder.pop("model", None)
    if encoder is None:
        if encoder_pretrained_model_name_or_path is None:
            raise ValueError(
                "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                "to be defined."
            )
        if "config" not in kwargs_encoder:
            encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
            )
            if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                encoder_config.is_decoder = False
                encoder_config.add_cross_attention = False
            kwargs_encoder["config"] = encoder_config
        encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)
    decoder = kwargs_decoder.pop("model", None)
    if decoder is None:
        if decoder_pretrained_model_name_or_path is None:
            raise ValueError(
                "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                "to be defined."
            )
        if "config" not in kwargs_decoder:
            decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
            )
            if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                decoder_config.is_decoder = True
                decoder_config.add_cross_attention = True
            kwargs_decoder["config"] = decoder_config
        decoder = AutoModelForTokenClassification.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
    config.tie_word_embeddings = False
    return cls(encoder=encoder, decoder=decoder, config=config)


def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}
    kwargs_decoder = {
        argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
    }
    if encoder_outputs is None:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_encoder,
        )
    elif isinstance(encoder_outputs, tuple):
        encoder_outputs = BaseModelOutput(*encoder_outputs)
    encoder_hidden_states = encoder_outputs[0]
    if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
    ):
        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
    encoder_attention_mask = None
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        use_cache=use_cache,
        past_key_values=past_key_values,
        return_dict=return_dict,
        **kwargs_decoder,
    )
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        logits = decoder_outputs.logits.reshape(-1, 2)
        loss = loss_fct(logits, labels.reshape(-1))
    if not return_dict:
        if loss is not None:
            return (loss,) + decoder_outputs + encoder_outputs
        else:
            return decoder_outputs + encoder_outputs
    return Seq2SeqLMOutput(
        loss=loss,
        logits=decoder_outputs.logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )


def gpt_fwd(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = transformer_outputs[0]
    hidden_states = self.dropout(hidden_states)
    logits = self.classifier(hidden_states)

    loss = None
    if labels is not None:
        labels = labels.to(logits.device)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
        output = (logits,) + transformer_outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return TokenClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )


def on_log(self, args, state, control, logs=None, **kwargs):
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)


def find_change(a, b):
    intersec = list(OrderedSet(a) & OrderedSet(b))
    index = 0
    num_del = 0
    no_index = []
    while b and a:
        try:
            ranks = [b.index(i) + a.index(i) for i in intersec]
            min_intersec = intersec[np.argmin(ranks)]
            intersec[0] = min_intersec
        except:
            pass
        if not intersec or b[index] != intersec[0]:
            del b[index]
            no_index.append(index + num_del)
            num_del += 1
        if not intersec or a[index] != intersec[0]:
            del a[index]
        if len(a) == 0 or len(b) == 0:
            continue
        if a and (a[index] == b[index]):
            del a[index], b[index]
            num_del += 1
        intersec = list(OrderedSet(a) & OrderedSet(b))
    return no_index


def tokenization_fn(captions, max_target_length=120):
    """Run tokenization on captions."""
    labels = tokenizer(captions,
                       padding="max_length",
                       max_length=max_target_length,
                       return_tensors="pt",
                       truncation=True).input_ids

    return labels


def feature_extraction_fn(image_paths):
    images = []
    # print(image_paths, flush=True)
    for image_file in image_paths:
        # print(image_file, flush=True)
        images.append(Image.open(image_file).convert('RGB'))

    encoder_inputs = feature_extractor(images=images, return_tensors="pt")

    return encoder_inputs.pixel_values


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def collate_fn(batch):
    images = []
    captions = []
    labels = []

    for obj in batch:
        images.append(obj[0])
        cap1 = obj[1]
        cap2 = alter_eng(
            cap1,
            nlp,
            alter_lemma,
            0.5
        )
        tk1 = tokenizer(cap1, return_tensors="pt").input_ids
        tk2 = tokenizer(cap2, return_tensors="pt").input_ids
        changes = find_change(tk1[0].tolist(), tk2[0].tolist())
        label = torch.zeros(tk2.shape[1], dtype=torch.long)
        label[changes] = 1
        captions.append(tk2[0])
        labels.append(label)
    images = feature_extraction_fn(images)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True)

    model_inputs = {'labels': labels, 'pixel_values': images, 'decoder_input_ids': captions}
    return model_inputs


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # print(preds.shape,labels.shape,(preds == labels).mean())
    return {'accuracy': (preds == labels).mean()}


def preprocess_logits_for_metrics(logits, labels):
    # print(logits[0].size(), labels.size())
    return logits[0].argmax(axis=2), labels


if __name__ == '__main__':
    # alter_lemma = json.load(open('/project/lt200203-aimedi/peune/out.json'))
    alter_lemma = {
        "NOUN": ['cat', 'dog', 'man'],
        "VERB": ['sit', 'run', 'walk'],
        "ADP": ['on', 'under']
    }

    ProgressCallback.on_log = on_log
    VisionEncoderDecoderModel.forward = forward
    GPT2ForTokenClassification.forward = gpt_fwd
    VisionEncoderDecoderModel.from_encoder_decoder_pretrained = from_encoder_decoder_pretrained
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type=str)
    parser.add_argument('--max_target_length', type=int, default=50)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--gradient_steps', type=int, default=4)
    parser.add_argument('--text_decode_model', type=str, default='gpt2')
    parser.add_argument('--logdir', type=str, default='/tmp/logs')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--encoder_freeze', action='store_true')
    args = parser.parse_args()
    expname = args.expname + f'_{args.lr:.1e}_{args.bs}'
    print(expname, flush=True)
    target_modules = ['enc_to_dec_proj', "q_proj", "v_proj", ]
    tokenizer = AutoTokenizer.from_pretrained(args.text_decode_model, trust_remote_code=True)
    nlp = spacy.load('en_core_web_sm')
    train_set = KarpathyDataset(
        '/media/palm/BiggerData/capgen/annotations/karpathy_train.json',
        '/media/palm/data/coco/images'
    )
    valid_set = KarpathyDataset(
        '/media/palm/BiggerData/capgen/annotations/karpathy_val.json',
        '/media/palm/data/coco/images'
    )
    print(len(train_set), flush=True)
    print(len(valid_set), flush=True)

    logdir = os.path.join(args.logdir, expname)
    rouge = evaluate.load('rouge')
    if 'gpt' in args.text_decode_model.lower():
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    base_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained('google/vit-base-patch16-224', args.text_decode_model)
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.decoder_start_token_id = tokenizer.bos_token_id
    base_model.config.pad_token_id = tokenizer.pad_token_id
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    if args.encoder_freeze:
        for param in base_model.encoder.parameters():
            param.requires_grad = False

    training_args = Seq2SeqTrainingArguments(
        output_dir='/tmp/t',
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.gradient_steps,
        per_device_eval_batch_size=1,
        learning_rate=args.lr,
        logging_steps=100,
        # max_steps=conf.max_steps,
        num_train_epochs=12,
        # report_to=conf.log_with,
        save_steps=5000,
        save_total_limit=1,
        logging_dir=logdir,
        warmup_steps=1000,
        warmup_ratio=1e-3,
        lr_scheduler_type='cosine',
        optim='adamw_torch',
        weight_decay=0.05,
        # bf16=True,
        remove_unused_columns=True,
        gradient_checkpointing=False,
        run_name=expname,
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
        evaluation_strategy="epoch",
        dataloader_num_workers=0,
        # eval_steps=50000,
        save_safetensors=False,
        generation_max_length=50

    )

    trainer = Seq2SeqTrainer(
        model=base_model,
        train_dataset=train_set,
        eval_dataset=valid_set,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collate_fn,
    )
    trainer.train(resume_from_checkpoint=args.resume)
