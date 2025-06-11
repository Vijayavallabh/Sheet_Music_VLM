class Config:
    MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
    DATASET_ID = "MYTH-Lab/NOTA-dataset"
    MAX_SEQ_LENGTH = 2048
    PHASE_1_SAMPLE_SIZE = 15000
    PHASE_2_SAMPLE_SIZE = 5000
    PHASE_3_SAMPLE_SIZE = 5000
    PHASE_1_OUTPUT_DIR = "./results/unsloth_nemotron_nota_phase1_alignment"
    PHASE_2_OUTPUT_DIR = "./results/unsloth_nemotron_nota_phase2_IE"
    PHASE_3_OUTPUT_DIR = "./results/unsloth_nemotron_nota_phase3_analysis"
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]