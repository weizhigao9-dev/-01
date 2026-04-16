import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import face_processor

# 设置页面配置
st.set_page_config(page_title="人脸检测与识别系统", page_icon="👤", layout="wide")

st.title("👤 基于 face_recognition 的人脸检测与识别")
st.markdown("上传一张包含人脸的图片，系统将自动检测人脸位置并提取特征。如果在左侧录入了已知人脸，系统还能尝试识别他们！")

# --- 侧边栏：准备人脸库（可选识别功能） ---
st.sidebar.header("📁 准备已知人脸库")
st.sidebar.markdown("上传单人清晰照片作为比对基准（可选）：")

# 使用 session_state 保存已知人脸编码，避免重新加载页面时丢失
if 'known_faces' not in st.session_state:
    st.session_state['known_faces'] = {}

uploaded_refs = st.sidebar.file_uploader("上传参考人脸", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_refs:
    for ref_file in uploaded_refs:
        name = ref_file.name.split('.')[0] # 使用文件名作为人名
        if name not in st.session_state['known_faces']:
            ref_image = Image.open(ref_file).convert('RGB')
            ref_array = np.array(ref_image)
            _, encodings = face_processor.get_face_data(ref_array)
            if encodings:
                st.session_state['known_faces'][name] = encodings[0]
                st.sidebar.success(f"已录入: {name}")
            else:
                st.sidebar.error(f"未在 {name} 中检测到人脸")

if st.session_state['known_faces']:
    st.sidebar.write("当前已知人员：", list(st.session_state['known_faces'].keys()))
    if st.sidebar.button("清空人脸库"):
        st.session_state['known_faces'] = {}
        st.rerun()

# --- 主界面：上传待检测图片 ---
st.header("🔍 检测与识别区")
target_file = st.file_uploader("请上传需要检测的图片 (支持合照)", type=['jpg', 'jpeg', 'png'])

if target_file is not None:
    # 1. 读取并显示原始图片
    image = Image.open(target_file).convert('RGB')
    image_array = np.array(image)
    
    with st.spinner("正在检测人脸并提取 128 维特征..."):
        # 2. 调用检测模块
        locations, encodings = face_processor.get_face_data(image_array)
        
        if not locations:
            st.warning("未能在图片中检测到人脸，请换一张图片尝试。")
        else:
            st.success(f"检测成功！共发现 {len(locations)} 张人脸。")
            
            # 3. 如果有已知人脸库，进行识别比对
            names = face_processor.recognize_faces(encodings, st.session_state['known_faces'])
            
            # 4. 在图片上绘制框和标签 (使用 Pillow)
            draw = ImageDraw.Draw(image)
            for (top, right, bottom, left), name in zip(locations, names):
                # 绘制红色边框 (线宽3)
                draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)
                
                # 绘制标签背景和文字
                text_width, text_height = draw.textbbox((0, 0), name)[2:] # 获取文字大致大小
                draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill="red", outline="red")
                draw.text((left + 6, bottom - text_height - 5), name, fill="white")
            
            # 5. 展示结果
            col1, col2 = st.columns(2)
            with col1:
                st.image(target_file, caption="原始图片", use_column_width=True)
            with col2:
                st.image(image, caption="检测与识别结果", use_column_width=True)
                
            # 6. 展示提取的特征信息（满足作业“提取128维”的展示要求）
            with st.expander("查看人脸特征数据 (128维编码)"):
                for i, (name, encoding) in enumerate(zip(names, encodings)):
                    st.write(f"**人脸 {i+1} ({name})** 特征向量前 5 维: {encoding[:5]}...")
