from pathlib import Path
import PIL
from io import BytesIO
import streamlit as st

# External packages
import streamlit as st

# Local Modules
import settings
import helper
import tempfile

# Setting page layout
st.set_page_config(
    page_title="Guided Navigation Using Object Detection and Lane Detection",
    #page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Guided Navigation Using Object Detection and Lane Detection")

# Sidebar
st.sidebar.header("Detection and Classification")

# Model Options
model_type = 'Detection'
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
source_vid = None

# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img:
                uploaded_image = PIL.Image.open(source_img)
                #st.image(source_img, caption="Uploaded Image",
                 #        use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img:
            if st.sidebar.button('Apply Detection'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                
                # Display the processed image
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                
                # Save the processed image to a temporary file
                processed_image = PIL.Image.fromarray(res_plotted)
                buffer = BytesIO()
                processed_image.save(buffer, format="JPEG")
                buffer.seek(0)
                
                # Provide a download button for the processed image
                st.download_button(
                    label="Download Processed Image",
                    data=buffer,
                    file_name="processed_image.jpg",
                    mime="image/jpeg"
                )

                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

# If video is selected
elif source_radio == settings.VIDEO:
    source_vid = st.sidebar.file_uploader("Choose a video...", type=("mp4", "avi", "mov", "mkv"))

    if source_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(source_vid.read())

        if st.sidebar.button('Detect Video Objects'):
            output_video_path = helper.play_uploaded_video(confidence, model, tfile.name)
            
            if output_video_path:
                # Provide a download button for the processed video
                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
            else:
                st.error("Failed to process the video. Please check the input file and try again.")

else:
    st.error("Please select a valid source type!")
