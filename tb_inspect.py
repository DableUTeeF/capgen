from tensorboard.backend.event_processing import event_accumulator


ea = event_accumulator.EventAccumulator(
    f'/home/palm/PycharmProjects/capgen/log2/encoder_sizes/gpt2_convnext-atto_8_5e-05_thai/events.out.tfevents.1720641556.x1000c2s1b0n0.1268.0',
    size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE
    # size_guidance={
    #     # event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    #     # event_accumulator.IMAGES: 4,
    #     # event_accumulator.AUDIO: 4,
    #     # event_accumulator.SCALARS: 0,
    #     # event_accumulator.HISTOGRAMS: 1,
    # }
)
ea.Reload()
print(ea.Tags())
print(len(ea.Scalars('train/loss')))
print(ea.Scalars('train/loss')[0].value)
print(ea.Scalars('train/loss')[-1].value)
