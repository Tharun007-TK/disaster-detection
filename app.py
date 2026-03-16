import streamlit as st
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import plotly.express as px
from PIL import Image as PILImage
import cv2

# Import from the pipeline
from damage_assessment_prototype import (
    load_tif, load_model, load_sam, classify_buildings,
    render_gt_overlay, render_pred_overlay, find_pairs,
    CLASS_COLORS, CLASS_NAMES
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Damage Assessment System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Dark industrial theme */
    .stApp {
        background-color: #0d1117;
        color: #e6e6e4;
    }
    div[data-testid="stSidebar"] {
        background-color: #141921 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #e6e6e4;
    }
    .css-1d391kg, .css-1dp5vir {
        background-color: #1a1f2e;
    }
    .status-dot {
        height: 12px;
        width: 12px;
        background-color: #97C459;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #97C459;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(151, 196, 89, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(151, 196, 89, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(151, 196, 89, 0); }
    }
    .title-mono {
        font-family: monospace;
        color: #e6e6e4;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .subtitle {
        color: #888780;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHING MODEL & SAM ---
@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_sam():
    return load_sam()

# Load models implicitly
model = get_model()
sam = get_sam()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="title-mono">DAMAGE ASSESSMENT SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Multimodal WSL · SAR + Optical · v0.1-proto</div>', unsafe_allow_html=True)
    st.markdown('<div style="display: flex; align-items: center; margin-bottom: 20px;"><span class="status-dot"></span><span style="color:#97C459; font-family: monospace;">System Online</span></div>', unsafe_allow_html=True)
    
    mode = st.radio("Mode", ["Single Inference", "Batch Run"], label_visibility="collapsed")
    
    st.markdown("---")
    
    pre_file = None
    post_file = None
    target_file = None
    batch_dir = "data/pre-event"
    max_pairs = 10
    run_assessment = False
    run_batch = False
    
    if mode == "Single Inference":
        pre_file = st.file_uploader("SAR — Sentinel-1 GRD (Pre-event)", type=["tif", "tiff"])
        post_file = st.file_uploader("OPT — Sentinel-2 L2A (Post-event)", type=["tif", "tiff"])
        target_file = st.file_uploader("Ground Truth mask (Optional)", type=["tif", "tiff"])
        
        can_run = pre_file is not None and post_file is not None
        run_assessment = st.button("Run Assessment", disabled=not can_run, use_container_width=True)
        
    else:
        batch_dir = st.text_input("Data folder path", value="data")
        max_pairs = st.number_input("Max pairs to process", min_value=1, max_value=500, value=10)
        run_batch = st.button("Run Batch", use_container_width=True)

# --- HELPER FUNCTIONS ---
def save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        f.write(uploaded_file.getbuffer())
        return Path(f.name)

def get_color_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

COLORS_HEX = [get_color_hex(c) for c in CLASS_COLORS]

# --- MAIN LOGIC ---
if run_assessment and mode == "Single Inference":
    try:
        with st.spinner("Processing Imagery & Running Inference..."):
            # Setup temp files
            pre_path = save_uploaded_file(pre_file)
            post_path = save_uploaded_file(post_file)
            target_path = save_uploaded_file(target_file)
            
            # Load images
            pre_rgb = load_tif(pre_path)
            post_rgb = load_tif(post_path)
            gt_mask = load_tif(target_path, is_target=True) if target_path else None
            
            # Match sizes
            if pre_rgb.shape != post_rgb.shape:
                post_rgb = cv2.resize(post_rgb, (pre_rgb.shape[1], pre_rgb.shape[0]), interpolation=cv2.INTER_AREA)
            
            # Sub-step Overlays
            gt_overlay = render_gt_overlay(post_rgb, gt_mask)
            
            # Run inference
            results = classify_buildings(model, sam, pre_rgb, post_rgb)
            pred_overlay = render_pred_overlay(post_rgb, results)
            
            # Stats
            dist = Counter(r["class_name"] for r in results)
            dominant_class = dist.most_common(1)[0][0] if dist else "N/A"
            total_b = len(results)
            mean_conf = np.mean([r["confidence"] for r in results]) if results else 0
            
            # Prepare DF
            rows = []
            for r in results:
                rows.append({
                    "building_id": len(rows),
                    "predicted_class": r["class_name"],
                    "confidence": round(r["confidence"], 4),
                    "score_no_damage": round(r["scores"][0], 4),
                    "score_minor": round(r["scores"][1], 4),
                    "score_major": round(r["scores"][2], 4),
                    "score_destroyed": round(r["scores"][3], 4),
                    "area_px": r["mask"]["area"],
                    "bbox_x1": r["bbox"][0], "bbox_y1": r["bbox"][1],
                    "bbox_x2": r["bbox"][2], "bbox_y2": r["bbox"][3]
                })
            df = pd.DataFrame(rows)
            
            # UI
            st.session_state['inference_done'] = True
            st.session_state['event_name'] = pre_file.name
            st.session_state['img_size'] = f"{pre_rgb.shape[1]}x{pre_rgb.shape[0]}"
            st.session_state['total_b'] = total_b
            st.session_state['dominant_class'] = dominant_class
            st.session_state['pre_rgb'] = pre_rgb
            st.session_state['post_rgb'] = post_rgb
            st.session_state['gt_overlay'] = gt_overlay
            st.session_state['pred_overlay'] = pred_overlay
            st.session_state['df'] = df
            st.session_state['has_gt'] = target_file is not None
            st.session_state['mean_conf'] = mean_conf
            st.session_state['dist'] = dist
            
            # Cleanup temp files
            for p in filter(None, [pre_path, post_path, target_path]):
                p.unlink()
                
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")

elif run_batch and mode == "Batch Run":
    pass # Will implement Batch later in session state or directly

# --- RENDER RESULTS ---
if mode == "Single Inference" and st.session_state.get('inference_done', False):
    
    # Header Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Event Name", st.session_state['event_name'])
    c2.metric("Image Size", st.session_state['img_size'])
    c3.metric("Buildings Detected", st.session_state['total_b'])
    c4.metric("Dominant Damage Class", st.session_state['dominant_class'])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Imagery & Overlay", "Per-Building Stats"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.image(PILImage.fromarray(st.session_state['pre_rgb']), use_container_width=True)
            st.caption("Pre-event · Optical")
        with col2:
            st.image(PILImage.fromarray(st.session_state['post_rgb']), use_container_width=True)
            st.caption("Post-event · SAR")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            if st.session_state['has_gt']:
                st.image(PILImage.fromarray(st.session_state['gt_overlay']), use_container_width=True)
                st.caption("Ground Truth (target mask)")
            else:
                st.markdown("""
                <div style="height: 300px; background-color: #1a1f2e; display: flex; align-items: center; justify-content: center; border: 1px dashed #3f3f46; border-radius: 4px;">
                    <span style="color: #888780;">No ground truth available</span>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Ground Truth (target mask)")
        with col4:
            st.image(PILImage.fromarray(st.session_state['pred_overlay']), use_container_width=True)
            st.caption("Model Prediction")

    with tab2:
        df = st.session_state['df']
        dist = st.session_state['dist']
        total_b = st.session_state['total_b']
        
        no_dmg = dist.get("No Damage", 0)
        maj_dest = dist.get("Major Damage", 0) + dist.get("Destroyed", 0)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Buildings", total_b)
        c2.metric("No Damage", f"{no_dmg} ({(no_dmg/total_b*100):.1f}%)" if total_b else "0")
        c3.metric("Major/Destroyed", f"{maj_dest} ({(maj_dest/total_b*100):.1f}%)" if total_b else "0")
        c4.metric("Mean Confidence", f"{st.session_state['mean_conf']:.2f}")
        
        if not df.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            # Create a plotly bar chart
            dist_df = pd.DataFrame([{"Class": k, "Count": dist.get(k, 0)} for k in CLASS_NAMES])
            fig = px.bar(dist_df, x="Class", y="Count", color="Class", 
                         color_discrete_sequence=COLORS_HEX, title="Damage Class Distribution")
            fig.update_layout(paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e", font_color="#e6e6e4")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Damage Results Data")
            
            def color_classes(val):
                if val == "No Damage": return f'color: {COLORS_HEX[0]}; font-weight: bold'
                if val == "Minor Damage": return f'color: {COLORS_HEX[1]}; font-weight: bold'
                if val == "Major Damage": return f'color: {COLORS_HEX[2]}; font-weight: bold'
                if val == "Destroyed": return f'color: {COLORS_HEX[3]}; font-weight: bold'
                return ''
                
            styled_df = df.style.map(color_classes, subset=['predicted_class'])
            st.dataframe(styled_df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV Report", csv, "damage_results.csv", "text/csv")

elif run_batch and mode == "Batch Run":
    try:
        with st.spinner("Processing Batch..."):
            pairs = find_pairs()
            pairs = pairs[:max_pairs]
            if not pairs:
                st.error("No image pairs found.")
            else:
                progress_bar = st.progress(0)
                all_rows = []
                
                tabs = st.tabs(["Imagery & Overlay", "Per-Building Stats", "Batch Results"])
                
                with tabs[2]:
                    st.markdown("### Batch Processing Results")
                    
                    for i, pair in enumerate(pairs):
                        pre_rgb = load_tif(pair["pre"])
                        post_rgb = load_tif(pair["post"])
                        if pre_rgb.shape != post_rgb.shape:
                            post_rgb = cv2.resize(post_rgb, (pre_rgb.shape[1], pre_rgb.shape[0]), interpolation=cv2.INTER_AREA)
                        
                        results = classify_buildings(model, sam, pre_rgb, post_rgb)
                        pred_overlay = render_pred_overlay(post_rgb, results)
                        
                        dist = Counter(r["class_name"] for r in results)
                        
                        with st.expander(f"Processed: {pair['stem']} | Buildings: {len(results)}"):
                            st.write(" | ".join([f"{k}: {v}" for k, v in dist.items()]))
                            st.image(PILImage.fromarray(pred_overlay), caption="Prediction Overlay", use_container_width=True)
                            
                        # aggregate stats
                        for r in results:
                            all_rows.append({
                                "image": pair["stem"],
                                "predicted_class": r["class_name"]
                            })
                            
                        progress_bar.progress((i + 1) / len(pairs))
                
                if all_rows:
                    batch_df = pd.DataFrame(all_rows)
                    summary = batch_df.groupby(["image", "predicted_class"]).size().unstack(fill_value=0)
                    summary['total_buildings'] = summary.sum(axis=1)
                    st.dataframe(summary)
                    
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Combined Report", csv, "damage_report_combined.csv", "text/csv")
                    
    except Exception as e:
        st.error(f"Error during batch processing: {str(e)}")

else:
    if mode == "Single Inference":
        st.info("Upload imagery and click 'Run Assessment' to begin.")
    else:
        st.info("Configure batch settings and click 'Run Batch' to begin.")
