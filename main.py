from src.models.model_loader import get_model

model = get_model("convnext_base", freeze=True)
print(model)