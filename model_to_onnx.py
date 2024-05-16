import torch
from data import DataModule
from model import BERTClass

MODEL_PATH = ".\models\model.pt"

def convert_model():
    model_path = MODEL_PATH
    model = BERTClass()
    model.load_state_dict(torch.load(model_path))     

    data_model = DataModule('bert-base-uncased')
    data_model.setup()
    input_batch = next(iter(data_model.get_train_dataloader()))
    input_sample = {
        "ids": input_batch["ids"][0].unsqueeze(0),
        "mask": input_batch["mask"][0].unsqueeze(0),
        "token_type_ids": input_batch["token_type_ids"][0].unsqueeze(0)
    }

    # Export the model
    torch.onnx.export(
        model,  # model being run
        args = (
            input_sample["ids"],
            input_sample["mask"],
            input_sample["token_type_ids"],
        ),  # model input (or a tuple for multiple inputs)
        f=".\models\model.onnx",  # where to save the model (can be a file or file-like object)
        input_names=["ids", "mask", "token_type_ids"],  # the model's input names
        output_names=["output"],  # the model's output names
    )

if __name__ == "__main__":
    convert_model()