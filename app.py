import streamlit as st
import torch
from torchvision.utils import make_grid, save_image
from io import BytesIO
from model import Generator   # ✅ Import from model.py

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="GAN Human Face Generator",
    page_icon="🧑",
    layout="wide"
)

# -----------------------------
# Load Generator
# -----------------------------
@st.cache_resource
def load_generator():
    latent_dim = 100
    device = torch.device('cpu')
    generator = Generator(latent_dim=latent_dim, channels=3).to(device)
    checkpoint = torch.load("checkpoints/checkpoint_epoch_680.pt", map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    return generator

generator = load_generator()

# -----------------------------
# UI Layout
# -----------------------------
st.title("🧑 AI Human Face Generator")
st.markdown(
    """
    Generate realistic **human face images** using a pretrained **GAN model**.  
    Use the controls below to customize your grid.  
    """
)

col1, col2 = st.columns([1, 2])

with col1:
    grid_size = st.selectbox("🔢 Select Grid Size", [4, 6, 8], index=2)
    seed = st.number_input("🎲 Random Seed (for reproducibility)", min_value=0, max_value=9999, value=42, step=1)
    generate = st.button("✨ Generate Faces")

with col2:
    if generate:
        num_images = grid_size * grid_size
        latent_dim = 100

        # Set random seed for reproducibility
        torch.manual_seed(seed)

        noise = torch.randn(num_images, latent_dim, 1, 1)

        with st.spinner("🧪 Generating faces..."):
            with torch.no_grad():
                fake_images = generator(noise)
                fake_images = (fake_images + 1) / 2

            grid_image = make_grid(fake_images, nrow=grid_size, padding=2, normalize=True)

            buffer = BytesIO()
            save_image(grid_image, buffer, format="PNG")
            buffer.seek(0)

            st.image(buffer, caption=f"Generated {grid_size}x{grid_size} Faces", use_container_width=True)

            # Add download button
            st.download_button(
                label="💾 Download Image Grid",
                data=buffer,
                file_name=f"generated_faces_{grid_size}x{grid_size}.png",
                mime="image/png"
            )
