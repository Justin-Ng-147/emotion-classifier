import datasets
import torch
from datasets import load_dataset
import pandas as pd
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from transformers import AutoTokenizer


TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 8

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer , max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = self.data['label']
        self.text = self.data['text']

    def __len__(self):
        return len(self.data.text)

    def __getitem__(self, index):
        text = self.text[index]

        inputs = self.tokenizer(
            text,
            max_length = self.max_len,
            padding = 'max_length'
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids,dtype=torch.long),
            'mask': torch.tensor(mask,dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids,dtype=torch.long),
            'targets': torch.tensor(self.data['label'][index],dtype=torch.long)
        }
    
class DataModule():
    def __init__(self,tokenizer_name, max_len = 232):
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.training_loader = None
        self.testing_loader = None

    def prepare_data(self):
        # data = load_dataset("dair-ai/emotion","unsplit",trust_remote_code=True)
        # df = data['train'].to_pandas()
        data = load_dataset("dair-ai/emotion",trust_remote_code=True)
        df = pd.concat([data['train'].to_pandas(),data['validation'].to_pandas(),data['test'].to_pandas()],ignore_index=True)
        df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)
        df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
        df['text'] = df['text'].str.replace(r'\d+', '', regex=True)
        df['text'] = df['text'].str.lower()
        cachedStopWords = set(stopwords.words("english"))
        df["text"] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in cachedStopWords]))
        df['text'] = df['text'].str.replace(r'[^a-zA-Z\s]', '',regex=True)

        return df

    def setup(self):
        df = self.prepare_data()
        train_data = df.sample(frac=0.8,random_state=420)
        test_data = df.drop(train_data.index).reset_index(drop=True)
        train_data = train_data.reset_index(drop=True)

        train_set = EmotionDataset(train_data, self.tokenizer, self.max_len)
        test_set = EmotionDataset(test_data, self.tokenizer, self.max_len)

        train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

        test_params = {'batch_size': VAL_BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': 0
                        }

        self.training_loader = torch.utils.data.DataLoader(train_set, **train_params)
        self.testing_loader = torch.utils.data.DataLoader(test_set, **test_params)


    def get_train_dataloader(self):
        return self.training_loader
    
    def get_test_dataloader(self):
        return self.testing_loader
    
if __name__ == "__main__":
    data_model = DataModule('bert-base-uncased')
    data_model.setup()
    print(next(iter(data_model.get_train_dataloader())))
    print(next(iter(data_model.get_train_dataloader()))["ids"].shape)
