# EasyGPTQ
A simple example of using GPTQ
# Papers
GPTQ ---> [``GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers``](https://arxiv.org/abs/2210.17323)

QuaRot ---> [``QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs``](https://arxiv.org/abs/2404.00456)

# Model + GPTQ + Inference after Quantization
```py
gptq_args = args_utils.parser_gen()  # Parse command-line arguments
gpt_model = prepare_model(args)  # Prepare the GPT model
gptq_fwrd(gpt_model, train_loader, dev, gptq_args)  # Perform GPTQ forward pass
gpt_model = prepare_model(args)  # Reload the model
model_path = './updated_model_weights.pt'  # Specify the path for quantized weights
gpt_model.load_state_dict(torch.load(model_path), strict=False)  # Load weights, ignoring KV cache

# Execute inference
gpt_model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation for inference
    for input_data in inference_loader:  # Iterate over the inference data
        output = gpt_model(input_data)  # Get the model's predictions
        # Process the output as needed (e.g., convert to text, save results)
```


# Weights to be quantized definition
```py
sequential = [
    ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],  # Attention projection weights
    ['self_attn.o_proj.module'],  # Output projection weights
    ['mlp.up_proj.module', 'mlp.gate_proj.module'],  # MLP upward projection weights
    ['mlp.down_proj.module']  # MLP downward projection weights
]
```

# Parameters
Defined inside args_utils
For example, w_bit can be 16/8/4 bits

# Prefill tokens
Execute prefill inference to provide calibration data for GPTQ

# Usage
```bash
python main.py
```

# Acknowledgment
The code is based on [``QuaRot``](https://github.com/spcl/QuaRot).
