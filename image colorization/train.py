import engine, model, data_setup, utils
import torch


NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001

train_dir = "/content/drive/MyDrive/Linnaeus 5 256X256/train"
test_dir = "/content/drive/MyDrive/Linnaeus 5 256X256/test"

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_dir, test_dir, BATCH_SIZE
)

model = model.ConvAutoencoder().to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device,
)

# Save the model with help from utils.py
utils.save_model(
    model=model,
    target_dir="models",
    model_name="model_1.pth",
)
