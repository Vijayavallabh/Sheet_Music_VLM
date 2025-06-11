import gradio as gr
from datasets import load_dataset
from unsloth import FastLanguageModel
from config import Config
from data_manage import DataManager
from model import ModelTrainer
from inference import InferenceHandler


def build_gradio_demo(handler):
    """Builds and returns a Gradio demo interface."""
    demo_title = "NotaGPT with Nemotron-Nano-VL & Unsloth: The AI Music Theorist"
    demo_description = """
    <p style='text-align: center;'>An implementation of the NOTA paper's methodology, fine-tuned on <b>NVIDIA's Llama-3.1-Nemotron-Nano-VL-8B-V1</b> and highly optimized with <b>Unsloth</b>.</p>
    <p style='text-align: center;'>Upload an image of a musical score and try one of the example prompts!</p>
    """
    
    try:
        demo_dataset = load_dataset(Config.DATASET_ID, split="test_analysis", streaming=True).take(1)
        example_data = next(iter(demo_dataset))
        example_image = example_data['image']
    except Exception:
        example_image = None
        print("Could not fetch example image for Gradio.")

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"<h1 style='text-align: center;'>{demo_title}</h1>")
        gr.Markdown(demo_description)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Music Score")
                prompt_input = gr.Textbox(label="Your Question or Command", placeholder="e.g., 'Transcribe this music.'")
                submit_button = gr.Button("Analyze Score", variant="primary")
            with gr.Column(scale=1):
                output_text = gr.Textbox(label="NotaGPT's Response", lines=15, interactive=False)
                
        gr.Examples(
            examples=[
                [example_image, "Transcribe the musical notation in this image into ABC format."],
                [example_image, "What is the key signature of this piece?"],
                [example_image, "Analyze the rhythmic structure and complexity of this music."],
            ] if example_image else [],
            inputs=[image_input, prompt_input]
        )

        submit_button.click(fn=handler.predict, inputs=[image_input, prompt_input], outputs=output_text)
    
    return demo


def main():
    config = Config()
    
    _, initial_tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_ID, max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=None, load_in_4bit=True
    )
    
    data_manager = DataManager(initial_tokenizer)
    model_trainer = ModelTrainer(config)

    ds_phase1 = data_manager.get_dataset_for_phase("train_alignment", config.PHASE_1_SAMPLE_SIZE)
    if ds_phase1:
        model_trainer.train_phase(1, ds_phase1, config.PHASE_1_OUTPUT_DIR)

    ds_phase2 = data_manager.get_dataset_for_phase("train_IE", config.PHASE_2_SAMPLE_SIZE)
    if ds_phase2:
        model_trainer.train_phase(2, ds_phase2, config.PHASE_2_OUTPUT_DIR, input_adapter_path=config.PHASE_1_OUTPUT_DIR)
      
    ds_phase3 = data_manager.get_dataset_for_phase("train_analysis", config.PHASE_3_SAMPLE_SIZE)
    if ds_phase3:
        model_trainer.train_phase(3, ds_phase3, config.PHASE_3_OUTPUT_DIR, input_adapter_path=config.PHASE_2_OUTPUT_DIR)

    inference_handler = InferenceHandler(config.PHASE_3_OUTPUT_DIR, config)
    demo = build_gradio_demo(inference_handler)
    
    print("\n--- Launching Gradio Demo ---")
    demo.launch(share=True, debug=True)

if __name__ == "__main__":
    main()