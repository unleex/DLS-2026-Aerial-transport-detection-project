from collections import Counter
from PIL import Image
from streamlit_image_zoom import image_zoom
from pathlib import Path
import requests
import streamlit as st

API = "http://127.0.0.1:8001"

st.title("Aerial object detection")

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)
LABELS = {
    "pedestrian": 0,
    "people": 1,
    "bicycle": 2,
    "car": 3,
    "van": 4,
    "truck": 5,
    "tricycle": 6,
    "awning-tricycle": 7,
    "bus": 8,
    "motor": 9,
}

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "results" not in st.session_state:
    st.session_state.results = []


# File upload
def clear_uploader():
    st.session_state.uploader_key += 1


with st.container():
    upload_col, clear_col = st.columns([6, 1])
    with upload_col:
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
    with clear_col:
        # Align button with the uploader
        st.text("")
        st.text("")
        st.button("Clear files", on_click=clear_uploader)


# Settings
with st.sidebar:
    st.title("Settings")

    selected_model = st.radio("Model:", ("Fast", "Pro"))

    classes_to_detect = st.pills(
        "Select objects to detect:",
        options=list(LABELS.keys()),
        selection_mode="multi",
        default=list(LABELS.keys()),
    )

    sorting_order = st.radio("Sort by:", ("Upload order", "Object count"))
    if sorting_order == "Object count":
        objects_to_sort_by = st.multiselect(
            "Select objects to count",
            options=classes_to_detect,
            default=classes_to_detect,
        )


# Run detection
if st.button("Run!"):
    progress_bar = st.progress(0)
    st.session_state.results = []

    for idx, file in enumerate(uploaded_files):
        savepath = TMP_DIR / file.name
        with open(savepath, "wb+") as f:
            f.write(file.getvalue())

        with st.spinner(f"Processing {file.name}..."):
            response = requests.post(
                f"{API}/predict",
                json={
                    "filename": str(savepath),
                    "model": selected_model,
                    "classes": [LABELS[cls] for cls in classes_to_detect],
                },
            ).json()

        st.session_state.results.append(
            {
                "original_name": file.name,
                "output_filename": response["output_filename"],
                "detected": response["detected"],
            }
        )
        progress_bar.progress((idx + 1) / len(uploaded_files))

    progress_bar.empty()

if st.session_state.results:
    display_results = st.session_state.results.copy()

    if sorting_order == "Object count":
        display_results.sort(
            key=lambda x: sum(
                Counter(x["detected"]).get(cls, 0) for cls in objects_to_sort_by
            ),
            reverse=True,
        )

    for res in display_results:
        with st.expander(label=res["original_name"], expanded=True):
            image_zoom(
                Image.open(res["output_filename"]), mode="both", keep_resolution=True
            )
