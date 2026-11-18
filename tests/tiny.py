"""Toy-sized Gemma 4 text config: the real 26B config with the dimensions shrunk."""

from transformers import AutoConfig

LAYERS, EXPERTS, TOP_K, VOCAB = 3, 16, 4, 512


def tiny_text_config():
    text = AutoConfig.from_pretrained("google/gemma-4-26B-A4B-it").text_config
    d = text.to_dict()
    d.pop("per_layer_config", None)  # derived from layer_types; rebuilt by the constructor
    d.update(
        hidden_size=64,
        intermediate_size=48,
        moe_intermediate_size=16,
        num_hidden_layers=LAYERS,
        layer_types=text.layer_types[3 : 3 + LAYERS],  # includes one full_attention layer
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        num_experts=EXPERTS,
        top_k_experts=TOP_K,
        vocab_size=VOCAB,
        vocab_size_per_layer_input=VOCAB,
    )
    return type(text)(**d)
