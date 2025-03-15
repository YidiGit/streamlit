import os
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import open_clip
import base64

# 配置路径
CLASS_NAMES = sorted([d for d in os.listdir('dataset') if os.path.isdir(f'dataset/{d}')])

# 初始化模型
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model.eval(), preprocess

model, preprocess = load_model()

# 加载嵌入数据
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
    """加载甲骨文图片"""
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

    image_path ="background.jpg"  # 确保图片路径正确
    image_base64 = get_image_base64(image_path)

    # 使用 HTML 和 CSS 来设置图片背景
    st.markdown(
        f"""
        <style>
        .bg {{
            # width: 100%;
            # height: 700px;
            background-image: url('{image_base64}');
            background-size: cover;
            background-position: center;
            # opacity: 0.5;
            color: white;
            padding: 50px;
            text-align: left;
            font-size: 50px;
            font-weight: bold;
        }}
        </style>
        <div class="bg">
            <h1>Chinese Characters Are Drawn - Oracle Bone Script</h1>
            <h3>Oracle Bone 12 Zodiac Animals Script</h3>
            <h4>12 Zodiac Animals; Rat, Ox, Tiger, Rabbit, Dragon, Snake, Horse, Goat, Monkey, Rooster, Dog, Pig</h4>
           
        </div>
        """,
        unsafe_allow_html=True
    )

    # st.title("Chinese Characters Are Drawn - Oracle Bone Script")
    # st.subheader("Oracle Bone 12 Zodiac Animals Script")
    # st.markdown("12 Zodiac Animals; Rat, Ox, Tiger, Rabbit, Dragon, Snake, Horse, Goat, Monkey, Rooster, Dog, Pig")
    # st.write("Draw your sketch below to get started:")

    # 添加全屏背景样式（在main函数最前面添加）
    st.markdown(
        """
        <style>
       
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom, #cab797, #d8c997);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1 , col_left, col_right = st.columns([1,5,4])
    
    with col_left:
        st.write("**Draw your sketch below to get started:**")
        canvas_result = st_canvas(
            fill_color="#ffffff",
            stroke_width=2,
            stroke_color="#000000",
            background_color="#ffffff",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        # 当画布有绘图内容时，在下方显示 Predict 按钮
        if canvas_result.image_data is not None:
            if st.button("Predict"):
                # 获取绘图图像并转换为 PIL Image
                sketch_image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("RGB")
                
                with st.spinner("predict..."):
                    # 获取嵌入向量并计算余弦相似度
                    query_embedding = get_embedding(sketch_image)
                    similarities = cosine_similarity(query_embedding, all_embeddings)
                    
                    class_scores = {}
                    for cls_idx, cls_name in enumerate(CLASS_NAMES):
                        mask = (all_labels == cls_idx)
                        class_scores[cls_name] = np.mean(similarities[mask])
                    
                    # 排序得到得分最高的类别
                    sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
                    best_class = sorted_classes[0][0]
                    
                    # 将预测结果存储到 session_state 中，便于右侧列显示
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
            
            st.success(f"Identify Result;{best_class}")
            
            # 左右并排显示 oracle 骨刻图与真实动物图
            col_oracle, col_real = st.columns(2)
            oracle_img = load_oracle_bone(best_class)
            real_img = load_real_animal(best_class)
            
            with col_oracle:
                st.markdown("#### Oracle Bone")
                st.image(oracle_img, width=200)
                st.caption(f"「{best_class}」Oracle Bone")
                
            with col_real:
                st.markdown("#### Real Animal")
                st.image(real_img, width=200)
                st.caption(f"{best_class} Real Animal")
            
            # 在下方显示其他预测结果
            st.markdown("### Other Results")
            cols = st.columns(3)
            for i, (cls_name, score) in enumerate(sorted_classes[1:4]):
                with cols[i]:
                    o_img = load_oracle_bone(cls_name)
                    st.image(o_img, width=100)
                    st.progress(score.item())
                    st.caption(f"{cls_name} ({score*100:.1f}%)")

# st.image("background.jpg", 
#              width=500,  # 可以根据需要调整图片显示宽度
#     )

    # Creat canvas
    # canvas_result = st_canvas(
    #     fill_color="#ffffff",
    #     stroke_width=2,
    #     stroke_color="#000000",
    #     background_color="#ffffff",
    #     height=300,
    #     width=300,
    #     drawing_mode="freedraw",
    #     key="canvas",
    # )

    # if canvas_result.image_data is not None:
    #     sketch_image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("RGB")
        
    #     if st.button("Predict"):
    #         with st.spinner("predict..."):
    #             # Get embedding
    #             query_embedding = get_embedding(sketch_image)
                
    #             similarities = cosine_similarity(query_embedding, all_embeddings)
                
    #             class_scores = {}
    #             for cls_idx, cls_name in enumerate(CLASS_NAMES):
    #                 mask = (all_labels == cls_idx)
    #                 class_scores[cls_name] = np.mean(similarities[mask])
                
    #             # Getting the best match
    #             sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    #             best_class = sorted_classes[0][0]
                
    #             # show result
    #             st.success(f"Identify Result：{best_class}")
                
    #             # show image
    #             col1, col2 = st.columns(2)
    #             # col1= st.columns(1)
    #             oracle_img = load_oracle_bone(best_class)
    #             real_img = load_real_animal(best_class)
                
    #             with col1:
    #                 st.markdown("#### oracle bone")
    #                 st.image(oracle_img, width=200)
    #                 st.caption(f"「{best_class}」Oracle Bone")
                
    #             with col2:
    #                 st.markdown("#### Real Animal")
    #                 st.image(real_img, width=200)
    #                 st.caption(f"{best_class}Real Animal")
                
    #             # show other result
    #             st.markdown("### Other Results")
    #             cols = st.columns(3)
    #             for i, (cls_name, score) in enumerate(sorted_classes[1:4]):
    #                 with cols[i]:
    #                     o_img= load_oracle_bone(cls_name)
    #                     st.image(o_img, width=100)
    #                     #st.progress(score)
    #                     #st.progress(float(score))
    #                     st.progress(score.item())
    #                     st.caption(f"{cls_name} ({score*100:.1f}%)")

if __name__ == "__main__":
    main()
