"""
FastAPI приложение для детекции переломов
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import Dict, Any
import os
import logging
from datetime import datetime

from model import get_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bone Fracture Detection API",
    description="API для детекции переломов костей на рентгеновских снимках",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")

#Инициализация модели
@app.on_event("startup")
async def startup_event():
    """Загрузка модели при запуске сервера"""
    
    logger.info("Запуск Bone Fracture Detection API")
    try:
        get_detector(MODEL_PATH)
        logger.info("Модель загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "service": "Bone Fracture Detection API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "/": "Информация о сервисе",
            "/health": "Проверка статуса",
            "/predict": "POST - Загрузить изображение для детекции",
            "/docs": "Swagger документация",
            "/redoc": "ReDoc документация"
        }
    }


@app.get("/health")
async def health_check():
    """Проверка сервиса"""
    try:
        detector = get_detector()
        return {
            "status": "healthy",
            "model_loaded": detector is not None,
            "device": detector.device,
            "classes": detector.class_names
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )
        
    
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Эндпоинт для детекции переломов
    
    Args:
        file: Загруженное изображение (JPEG, PNG)
    
    Returns:
        JSON с результатами детекции
    """
    try:
        content_type = file.content_type or ""
        filename = file.filename or ""
   
        is_image = content_type.startswith('image/') or \
                   any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp'])
        
        if not is_image:
            raise HTTPException(
                status_code=400, 
                detail="Файл должен быть изображением (JPG, JPEG, PNG, BMP)"
            )
        
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Пустой файл")
        
        detector = get_detector()
        result = detector.predict_from_bytes(contents)
        
        logger.info(
            f"Обработано изображение: {file.filename}, "
            f"размер: {len(contents)} байт, "
            f"найдено детекций: {result['num_detections']}"
        )
        
        response = {
            "success": True,
            "filename": file.filename,
            "num_detections": result["num_detections"],
            "detections": result["detections"],
            "image_size": result["image_size"],
            "message": f"Найдено {result['num_detections']} переломов" 
                       if result["num_detections"] > 0 
                       else "Переломов не обнаружено",
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@app.post("/predict-with-visualization")
async def predict_with_visualization(file: UploadFile = File(...)):
    """
    Эндпоинт для визуализации
    
    Возвращает изображение с bounding boxes
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Файл не является изображением")
        
        contents = await file.read()
        
        detector = get_detector()
        result = detector.predict_from_bytes(contents)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "num_detections": result["num_detections"],
            "detections": result["detections"],
            "annotated_image_base64": result["annotated_image_base64"],
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/model-info")
async def model_info():
    """Информация о модели"""
    detector = get_detector()
    return {
        "model_type": "YOLOv8",
        "classes": detector.class_names,
        "device": detector.device,
        "conf_threshold": detector.conf_threshold
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )