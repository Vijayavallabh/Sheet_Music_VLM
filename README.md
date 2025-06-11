# NotaGPT with Nemotron-Nano-VL & Unsloth: The AI Music Theorist

## Project Overview

This project implements knowledge distillation techniques to compress the Llama-3.1-Nemotron-Nano-VL-8B model into a smaller, more efficient model specialized for music score analysis. By applying the methodology from the NOTA (Notation Analysis) paper, we create an AI capable of understanding and analyzing musical notation in images.

### Key Features

- **Music Score Analysis**: Transcribe and analyze sheet music from images
- **LoRA Fine-tuning**: Efficient adaptation of the Nemotron-Nano-VL model for music domain
- **Multi-Phase Training**: Progressive training across alignment, information extraction, and analysis phases
- **Interactive Demo**: Gradio-based interface for testing music score analysis
- **Unsloth Optimization**: Leveraging Unsloth for faster training and inference

## Technical Approach

Our implementation follows a three-phase training approach:

1. **Alignment Phase**: Basic alignment of the VLM to music notation concepts
2. **Information Extraction**: Training the model to extract specific information from music scores
3. **Analysis Phase**: Advanced training for in-depth music theory analysis capabilities
4. **Optimization**: Apply quantization and Unsloth optimization techniques
5. **Evaluation**: Comprehensive testing across music analysis metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/username/Knowledge_Distillation_VLM.git
cd Knowledge_Distillation_VLM

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- **main.py**: Main execution flow and Gradio demo interface
- **config.py**: Configuration settings for model paths and hyperparameters
- **model.py**: Implementation of model training with Unsloth optimization
- **data_manage.py**: Music notation dataset preparation and formatting
- **inference.py**: Model inference handler for interacting with music scores
- **requirements.txt**: Project dependencies

## Configuration

The project uses a centralized configuration class in `config.py`. Key parameters include:

```python
MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
DATASET_ID = "MYTH-Lab/NOTA-dataset"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

## Usage

To run the full three-phase training pipeline and launch the demo:

```bash
python main.py
```

The training process will:
1. Train Phase 1: Alignment with musical notation
2. Train Phase 2: Information extraction from scores
3. Train Phase 3: Advanced music analysis capabilities
4. Launch the Gradio demo interface

## Gradio Demo Features

The interactive demo allows users to:

1. Upload images of music scores
2. Ask questions or give commands about the music
3. Receive detailed analyses including:
   - Music transcription in ABC notation
   - Key signature identification
   - Rhythmic structure analysis
   - Harmonic analysis
   - Performance suggestions

## Example Prompts

- "Transcribe the musical notation in this image into ABC format."
- "What is the key signature of this piece?"
- "Analyze the rhythmic structure and complexity of this music."
- "Identify any unusual harmonic progressions in this score."
- "Suggest articulation and dynamics for expressive performance."

## Dependencies

Core dependencies include:

- torch (>=2.0.0)
- transformers (>=4.30.0)
- accelerate (>=0.21.0)
- unsloth (>=0.3.0)
- gradio
- datasets (>=2.10.0)
- huggingface-hub (>=0.19.0)
- einops (>=0.6.0)
- open-clip-torch (>=2.20.0)

## Future Work

- Enhance ABC notation transcription accuracy
- Add support for more complex music theory concepts
- Implement real-time audio playback of transcribed scores
- Expand multi-language music notation support
- Improve performance on handwritten scores

## License

[Include your license information here]

## Acknowledgements

- Built on the methodology from the NOTA paper
- Utilizes NVIDIA's Llama-3.1-Nemotron-Nano-VL model
- Optimized with Unsloth framework
- Uses the MYTH-Lab/NOTA-dataset for music notation analysis
