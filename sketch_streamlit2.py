import os
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import open_clip
import base64

st.set_page_config(layout="wide")
# Path
CLASS_NAMES = sorted([d for d in os.listdir('dataset') if os.path.isdir(f'dataset/{d}')])

# Initialize the model
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model.eval(), preprocess

model, preprocess = load_model()

# Load the embedding data.
@st.cache_data
def load_embeddings():
    return np.load("animal_embeddings.npy"), np.load("animal_labels.npy")

all_embeddings, all_labels = load_embeddings()

def get_embedding(image):
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        return model.encode_image(image_tensor).squeeze().numpy()

def cosine_similarity(query_vec, target_vecs):
    dot_product = np.dot(target_vecs, query_vec)
    norm_product = np.linalg.norm(target_vecs, axis=1) * np.linalg.norm(query_vec)
    return dot_product / norm_product

def load_oracle_bone(class_name):
    """Oracle Bone"""
    base_path = f"Oracle_Bone"
    try:
        oracle_img = Image.open(f"{base_path}/{class_name}.jpg")
    except FileNotFoundError:
        oracle_img = Image.new('RGB', (300, 200), color='gray')
    
    return oracle_img

def load_real_animal(class_name):
    base_path = f"real"
    try:
        real_img = Image.open(f"{base_path}/{class_name}.png")
    except FileNotFoundError:
        real_img = Image.new('RGB', (200, 200), color='gray')
    
    return real_img


def get_image_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpg;base64,{encoded_string}"

def main():
    
    # Get the Base64 encoding of the image.
    image_path ="background2.png" 
    image_base64 = get_image_base64(image_path)
    
    # Use HTML and CSS to set the image background. padding-top: 50px;
    st.markdown(
    f"""
    <style>
    .bg {{
        height: 700px;
        background-image: url('{image_base64}');
        background-size: cover;
        background-position: right;
        color: #fbf8ef;
        font-size: 100px; 
        font-weight: bold;
        display: flex;
        flex-direction: column;
        padding: 0;  
        margin: 0;
    }}

    .bg h1 {{
        font-size: 90px !important;
    }}
    .bg h2 {{
        font-size: 40px !important;  
    }}
    .bg h3 {{
        font-size: 30px !important;  
    }}
    
    </style>
    <div class="bg">
        <h1>Oracle Bone Script - 12 Zodiac Animals</h1>
        <h2>Oracle bone inscriptions refer to the texts recorded on tortoise shells and animal bones for divination purposes during the Shang and Zhou dynasties in ancient China.</h2>
        <h3>12 Zodiac Animals: Rat, Ox, Tiger, Rabbit, Dragon, Snake, Horse, Goat, Monkey, Rooster, Dog, Pig</h4>
    </div>
    """,
    unsafe_allow_html=True
)

    # Add a full-screen background #cab797, #d8c997
    st.markdown(
        """
        <style>
       
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom, #c7c997, #c7c997);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col_left, col_right = st.columns([1,2])
    
    with col_left:
        st.markdown(
        "<p style='color: #4c3e26;font-size: 25px; font-family: Arial, font-weight: bold;;'><strong>Draw your sketch below to get started:</strong></p>", 

        unsafe_allow_html=True
    )
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=2,
            stroke_color="#000000",
            background_color="#ffffff",
            height=400,
            width=500,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        # When the canvas contains drawing content, display the "Predict" button below.
        if canvas_result.image_data is not None:
            if st.button("Predict"):
                # Get the plotted image and convert it to a PIL Image.
                sketch_image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("RGB")
                
                with st.spinner("predict..."):
                    #Get embedding vectors and calculate cosine similarity.
                    query_embedding = get_embedding(sketch_image)
                    similarities = cosine_similarity(query_embedding, all_embeddings)
                    
                    class_scores = {}
                    for cls_idx, cls_name in enumerate(CLASS_NAMES):
                        mask = (all_labels == cls_idx)
                        class_scores[cls_name] = np.mean(similarities[mask])
                    
                    # Sort to obtain the highest scoring category.
                    sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
                    best_class = sorted_classes[0][0]
                    
                    # Store the prediction results in session_state for display in the right column.
                    st.session_state.prediction = {
                        "best_class": best_class,
                        "sorted_classes": sorted_classes
                    }
    with col_right:
        st.markdown("<div style='margin-left:5cm'>", unsafe_allow_html=True)
        if "prediction" in st.session_state:
            pred = st.session_state.prediction
            best_class = pred["best_class"]
            sorted_classes = pred["sorted_classes"]
            
            st.success(f"Identify Result: {best_class}")
            
            # 使用一个容器来显示 Oracle Bone 和 Real Animal 的图片，而不是嵌套列
             # 使用 st.columns 创建两个子列，使得图片可以并排显示
            image_cols = st.columns([1, 1, 1])
            with image_cols[0]:
                st.markdown(
                      "<h4 style='color: #4c3f27;'>Oracle Bone Script</h4>",
                        unsafe_allow_html=True)
                oracle_img = load_oracle_bone(best_class)
                st.image(oracle_img, width=200)
                st.caption(f"「{best_class}」 Oracle Bone")
            with image_cols[1]:
                st.markdown(
                    "<h4 style='color: #4c3f27;'>Real-Life Images of Animals</h4>",
                    unsafe_allow_html=True)
                real_img = load_real_animal(best_class)
                st.image(real_img, width=150)
                st.caption(f"{best_class} Real Animal")
            
            with image_cols[2]:
                st.markdown("<h4 style='color: #4c3f27;'>Description</h4>",
                    unsafe_allow_html=True)
    
                if best_class == "03":
                    st.markdown("**Rat**: In oracle bone script, the character for 'rat' depicts a pointed mouth, sharp teeth, a hunched back, short legs, a long tail, and remnants of stolen food beside it.")
                elif best_class == "08":
                    st.markdown("**Snake**: In oracle bone script, the character for 'snake' features a triangular head at the top and a curved body below, resembling the form of a worm.")
                elif best_class == "04":
                    st.markdown("**Ox**: In oracle bone script, the character for 'ox' resembles the head of an ox, highlighting its pair of curved and robust horns.")
                elif best_class == "05":
                    st.markdown("**Tiger**: In oracle bone script, the character for 'tiger' depicts its head facing upward and tail downward, with a wide-open mouth, sharp teeth and claws, stripes on its body, and a strong, curved tail, vividly illustrating the majestic and fierce image of a tiger.")
                elif best_class == "06":
                    st.markdown("**Rabbit**: In oracle bone script, the character for 'rabbit' depicts a crouching rabbit with its head facing upward, short legs, a short tail, long ears, and a plump, round body.")
                elif best_class == "07":
                    st.markdown("**Dragon**: In oracle bone script, the character for 'dragon' portrays a beast's head with a serpent's body, adorned with horns on its head, scales on its body, and a long, trailing tail.")
                elif best_class == "09":
                    st.markdown("**Horse**: In oracle bone script, the character for 'horse' is depicted in a side view, complete with a mane on the head and neck, and a representation of the head, tail, and all four legs.")                
                elif best_class == "10":
                    st.markdown("**Sheep**: In oracle bone script, the character for 'sheep' illustrates a frontal view of a sheep's head, with two horns curving downward and nostrils forming a V-shape at the tip of the nose.")
                elif best_class == "11":
                    st.markdown("**Monkey**: In oracle bone script, the character for 'monkey' depicts a monkey standing sideways, with round, beady eyes, a furry tail, and a limb on each side of its body.")
                elif best_class == "12":
                    st.markdown("**Rooster**: In oracle bone script, the character for 'rooster' resembles the shape of a rooster, featuring a high crest, a long beak, and an elongated tail.")
                elif best_class == "01":
                    st.markdown("**Dog**: In oracle bone script, the character for 'dog' features a slender body and a long tail, highlighting the characteristic curled tail of a dog.")              
                elif best_class == "02":
                    st.markdown("**Pig**: In oracle bone script, the character for 'pig' outlines the shape of a pig, with a protruding belly, a short drooping tail, and stubby legs.")
                
                # Show other results 

                label_dict = {
                    "01": "Dog",
                    "02": "Pig",
                    "03": "Rat",
                    "04": "Ox",
                    "05": "Tiger",
                    "06": "Rabbit",
                    "07": "Dragon",
                    "08": "Snake",
                    "09": "Horse",
                    "10": "Sheep",
                    "11": "Monkey",
                    "12": "Rooster",
                }
            st.markdown("<h4 style='color: #4c3f27;'>Other Results</h4>",
                    unsafe_allow_html=True)
            
            cols = st.columns(3)
            for i, (cls_name, score) in enumerate(sorted_classes[1:4]):
                with cols[i]:

                    st.markdown(
                    """
                    <style>
                    .stProgress > div > div > div > div {
                        background-color: #8B4513;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                    )

                    st.markdown(f"**{label_dict[cls_name]}**")
                    o_img = load_oracle_bone(cls_name)
                    st.image(o_img, width=100)
                    st.progress(score.item())
                    st.caption(f"{cls_name} ({score*100:.1f}%)")


if __name__ == "__main__":
    main()
