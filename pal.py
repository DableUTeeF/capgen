from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, PaliGemmaConfig, AutoConfig, AutoImageProcessor, AutoTokenizer, CamembertModel
import torch
from PIL import Image
from transformers.modeling_utils import ModuleUtilsMixin


def get_extended_attention_mask(
        self, attention_mask, input_shape, device = None, dtype = None
):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if dtype is None:
        dtype = self.dtype

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 4:
        attention_mask = attention_mask[:, 0]
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder:
            extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                input_shape, attention_mask, device
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


if __name__ == '__main__':
    PaliGemmaProcessor.image_processor_class = "AutoImageProcessor"
    PaliGemmaProcessor.tokenizer_class = "AutoTokenizer"
    CamembertModel.get_extended_attention_mask = get_extended_attention_mask

    cfg = PaliGemmaConfig()
    cfg.text_config = AutoConfig.from_pretrained('clicknext/phayathaibert', is_decoder=True)
    cfg.text_config.num_key_value_heads = 1
    cfg.vision_config = AutoConfig.from_pretrained('google/vit-base-patch16-224-in21k')
    cfg.vision_config.projection_dim = 768
    model = PaliGemmaForConditionalGeneration(cfg)
    image_proc = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    image_proc.image_seq_length = 6
    tokenizer = AutoTokenizer.from_pretrained('clicknext/phayathaibert')
    tokenizer.add_tokens(['<image>'], special_tokens=True)
    processor = PaliGemmaProcessor(
        image_proc,
        tokenizer,
    )

    image = Image.open('/mnt/d/work/coco/images/val2017/000000237864.jpg')
    prompt = "caption es"
    model_inputs = processor(text=prompt, images=image, return_tensors="pt")
    input_len = model_inputs["input_ids"].shape[-1]
    # model_inputs['attention_mask'] = model_inputs['attention_mask'][0]

    embed_layer = model.get_input_embeddings()
    w = torch.cat([embed_layer.weight, embed_layer.weight[0:1]], 0)
    embed_layer.weight = torch.nn.Parameter(w)
    model.set_input_embeddings(embed_layer)

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        print(decoded)

    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = True


