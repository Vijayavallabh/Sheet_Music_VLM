# Knowledge Distillation for Vision-Language Models

This project implements knowledge distillation techniques to compress the Sujet-Finance-8B model into a smaller, more efficient model suitable for edge devices, while maintaining performance capabilities.

## Project Overview

Knowledge distillation is a compression technique where a small model (student) is trained to mimic a larger model (teacher). This project uses Unsloth for optimization and implements a multi-phase training approach to create an efficient version of the original vision-language model.

### Key Features

- **Model Compression**: Distill knowledge from the large pretrained Sujet-Finance-8B model into a smaller, more efficient model
- **LoRA Adaptation**: Efficient fine-tuning using Low-Rank Adaptation
- **Multi-Phase Training**: Progressive training strategy across three specialized phases
- **Interactive Demo**: Gradio-based interface for testing the model
- **Unsloth Optimization**: Leveraging Unsloth for faster training and inference

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Knowledge_Distillation_VLM.git
cd Knowledge_Distillation_VLM

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- **main.py**: Main execution flow and Gradio demo interface
- **config.py**: Configuration settings for model paths, hyperparameters, etc.
- **model.py**: Implementation of model training with Unsloth optimization
- **data_manage.py**: Data preparation and formatting
- **inference.py**: Model inference handler for production use
- **requirements.txt**: Project dependencies

## Training Pipeline

The training process is divided into three phases:

1. **Phase 1 (Alignment)**: Basic alignment training of the student model
2. **Phase 2 (Information Extraction)**: Training the model on information extraction tasks
3. **Phase 3 (Analysis)**: Fine-tuning for in-depth content analysis capabilities

Each phase builds upon the previous one, with the adapters from earlier phases serving as the starting point for later phases.

## Configuration

The project uses a centralized configuration class in `config.py`. Key parameters include:

```python
MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
DATASET_ID = "MYTH-Lab/NOTA-dataset"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32
```

## Usage

### Training

To run the full training pipeline:

```bash
python main.py
```

This will execute all three training phases sequentially and launch the Gradio demo interface.

### Interactive Demo

After training, an interactive Gradio demo is automatically launched that allows you to test the model:

1. Upload an image
2. Enter a text prompt
3. View the model's response

## Model Optimization

This project uses several optimization techniques:

- **Knowledge Distillation**: Transferring knowledge from teacher to student
- **LoRA Fine-tuning**: Efficient adaptation with minimal parameters
- **Quantization**: 4-bit quantization for reduced memory footprint
- **Unsloth Acceleration**: Specialized optimization for faster training and inference

## Dependencies

The main dependencies include:

- torch (>=2.0.0)
- transformers (>=4.30.0)
- accelerate (>=0.21.0)
- unsloth (>=0.3.0)
- datasets (>=2.10.0)
- streamlit (>=1.25.0)
- gradio

## License

[Include your license information here]

## Credits

- Inspired by research on Vision-Language Models and knowledge distillation
- Builds upon Unsloth's optimization framework
- Uses the NOTA dataset for music score analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
