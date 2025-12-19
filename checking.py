from src.infer import load_pipeline, predict

model, tok, l2i, i2l = load_pipeline()

text = "The patient has chest pain, was started on aspirin, and scheduled for a CT scan."
preds = predict(text, model, tok, i2l)
print(preds)
