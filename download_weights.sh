#!/bin/bash

set -e

# Prompt for HuggingFace token if not set
if [ -z "$HF_TOKEN" ]; then
    if [ -t 0 ]; then
        read -s -p "Enter your Hugging Face API token (input will not be visible): " HF_TOKEN
        echo
        export HF_TOKEN
    fi
fi

HF_HEADER=""
if [ -n "$HF_TOKEN" ]; then
    HF_HEADER="Authorization: Bearer $HF_TOKEN"
fi

mkdir -p model

# Virchow
echo "Downloading Virchow model from Hugging Face..."
curl -L -H "$HF_HEADER" https://huggingface.co/paige-ai/Virchow/blob/main/config.json $HF_AUTH -o model/virchow-config.json
curl -L -H "$HF_HEADER" https://huggingface.co/paige-ai/Virchow/blob/main/pytorch_model.bin $HF_AUTH -o model/virchow.pth
echo "Done."
echo ""

# biogpt
mkdir -p model/biogpt
echo "Downloading BioGPT model from Hugging Face..."

python3 - <<EOF
from transformers import BioGptTokenizer, BioGptForCausalLM

model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

model.save_pretrained("model/biogpt")
tokenizer.save_pretrained("model/biogpt")
EOF
echo "Done."
echo ""

# PRISM
echo "Downloading PRISM model from Hugging Face..."
curl -L -H "$HF_HEADER" https://huggingface.co/paige-ai/Prism/resolve/main/model.safetensors -o model/prism.pth
echo "Done."
echo ""

echo ""
echo "✅ All model weights downloaded and organized."