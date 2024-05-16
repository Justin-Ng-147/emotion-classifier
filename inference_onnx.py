import torch
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from transformers import AutoTokenizer
import onnxruntime as ort
import pandas as pd
import numpy as np

class ONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.lables = ["sadness", "joy", "love", "anger", "fear", "suprise"]
        self.tokenizer =  AutoTokenizer.from_pretrained('bert-base-uncased')

    def process(self,text):
        # text.replace(r'http\S+', '', regex=True)
        # text.replace(r'[^\w\s]', '', regex=True)
        # text.replace(r'\s+', ' ', regex=True)
        # text.replace(r'\d+', '', regex=True)
        # text.lower()
        # cachedStopWords = set(stopwords.words("english"))
        # text = ' '.join([word for word in text.split() if word not in cachedStopWords])
        # text.replace(r'[^a-zA-Z\s]', '',regex=True)

        df = pd.DataFrame({'text':[text]})
        df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)
        df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
        df['text'] = df['text'].str.replace(r'\d+', '', regex=True)
        df['text'] = df['text'].str.lower()
        cachedStopWords = set(stopwords.words("english"))
        df["text"] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in cachedStopWords]))
        df['text'] = df['text'].str.replace(r'[^a-zA-Z\s]', '',regex=True)

        text = df['text'][0]

        inputs = self.tokenizer(
            text,
            max_length = 232,
            padding = 'max_length'
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # return {
        #     'ids': torch.tensor(ids,dtype=torch.long),
        #     'mask': torch.tensor(mask,dtype=torch.long),
        #     'token_type_ids': torch.tensor(token_type_ids,dtype=torch.long),
        # }
        return {
            'ids': np.expand_dims(torch.tensor(ids,dtype=torch.long),axis=0),
            'mask': np.expand_dims(torch.tensor(mask,dtype=torch.long),axis=0),
            'token_type_ids': np.expand_dims(torch.tensor(token_type_ids,dtype=torch.long),axis=0),
        }


    def predict(self, text):
        ort_inputs = self.process(text)
        ort_outs = self.ort_session.run(None, ort_inputs)
        ort_outs = torch.Tensor(np.array(ort_outs)) 
        predictions = torch.argmax(ort_outs[0],dim=1)
        # print(ort_outs)
        # print(self.lables[predictions])
        return self.lables[predictions]

if __name__ == "__main__":
    sentence = "i am feeling grouchy"
    predictor = ONNXPredictor("./models/model.onnx")
    print(f"{sentence} : {predictor.predict(sentence)}")