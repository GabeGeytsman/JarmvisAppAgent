import os
import tempfile
import time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
from PIL import Image
import io
import base64
import torch
import pandas as pd
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Auto-detect device (GPU if available, otherwise CPU)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize models (only loaded once at startup)
yolo_model_path = 'weights/icon_detect_v1_5/model_v1_5.pt'
caption_model_name = 'florence2'
caption_model_path = 'weights/icon_caption_florence'

som_model = get_yolo_model(model_path=yolo_model_path)
som_model.to(device)

caption_model_processor = get_caption_model_processor(
    model_name=caption_model_name,
    model_name_or_path=caption_model_path,
    device=device
)

@app.post("/process_image/")
async def process_image(
    file: UploadFile = File(...),
    box_threshold: float = Form(0.05), # Box Threshold
    iou_threshold: float = Form(0.1),  # IOU Threshold
    imgsz_component: int = Form(640)  # Icon Detect Image Size
):
    try:
        
        # Save uploaded file to temporary path
        contents = await file.read()
        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, file.filename)

        with open(temp_image_path, 'wb') as f:
            f.write(contents)
        
        
        image = Image.open(temp_image_path).convert('RGB')
        start_time = time.time()
        # OCR detection
        ocr_bbox_rslt, _ = check_ocr_box(
            temp_image_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.5},
            use_paddleocr=True
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        # Generate labeled image and parsed content
        draw_bbox_config = {
            'text_scale': 0.8 * (max(image.size) / 3200),
            'text_thickness': max(int(2 * (max(image.size) / 3200)), 1),
            'text_padding': max(int(3 * (max(image.size) / 3200)), 1),
            'thickness': max(int(3 * (max(image.size) / 3200)), 1),
        }
        
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            temp_image_path,
            som_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=iou_threshold,
            scale_img=False,
            batch_size=128,
            imgsz=imgsz_component
        )
        elapsed_time = time.time() - start_time
        # Delete temporary files
        os.remove(temp_image_path)
        os.rmdir(temp_dir)
        

        # Return labeled image and parsed content
        image_bytes = base64.b64decode(dino_labeled_img)
        labeled_image = io.BytesIO(image_bytes)

        # Convert parsed content to DataFrame -> JSON
        df = pd.DataFrame(parsed_content_list)
        df['ID'] = range(len(df))
        parsed_content_json = df.to_dict(orient="records")

        # Base64 encode the image
        encoded_image = base64.b64encode(labeled_image.getvalue())
        

        return {
            "status": "success",
            "parsed_content": parsed_content_json,
            "labeled_image": encoded_image,
            "e_time": elapsed_time  # Return elapsed time
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# nohup fastapi run omni.py --port 8000 > ../logfile_omni.log 2>&1