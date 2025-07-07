import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    revision="fp16"
).to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate image
def generate(prompt, size):
    width, height = map(int, size.split("x"))
    image = pipe(prompt, height=height, width=width).images[0]
    return image

# Modern styled Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 ArtifyAI - Create Stunning AI Art")
    gr.Markdown("Enter your imagination as a prompt below. Choose a size, click Generate, and download your masterpiece.")

    with gr.Row():
        prompt = gr.Textbox(label="📝 Prompt", placeholder="e.g. A futuristic city in sunset", lines=2)
        size = gr.Dropdown(choices=["512x512", "640x640", "768x768"], value="512x512", label="🖼️ Image Size")

    with gr.Row():
        btn = gr.Button("🚀 Generate Art")
        output = gr.Image(label="Generated Art").style(height=512)
        download_btn = gr.Button("📥 Download Image", visible=False)

    generated_image = gr.State()

    def process(prompt, size):
        image = generate(prompt, size)
        return image, image, gr.update(visible=True)

    btn.click(fn=process, inputs=[prompt, size], outputs=[output, generated_image, download_btn])

    def download_img(image):
        return gr.File.update(value=image, visible=True)

    download_btn.click(fn=download_img, inputs=[generated_image], outputs=[download_btn])

demo.launch()
