from transformers import AutoModelForCausalLM, Blip2VisionModel, VisionEncoderDecoderModel, AutoConfig, Blip2VisionConfig, AutoModel, Blip2ForConditionalGeneration


AutoConfig.register("blip_2_vision_model", Blip2VisionConfig)
AutoModel.register(Blip2VisionConfig, Blip2VisionModel)
if __name__ == '__main__':
    model = Blip2VisionModel.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
    base_model = VisionEncoderDecoderModel(
        None,
        model,
        AutoModelForCausalLM.from_pretrained(
            'gpt2',
            is_decoder=True,
            add_cross_attention=True
        )
    )
    print()
