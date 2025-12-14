import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk
nltk.download("stopwords")
nltk.download('punkt') #  Recognizes that “Dr.” or “Inc.” aren’t sentence endings.
nltk.download('punkt_tab')
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)


logger=logging.getLogger('data_preprocessing')
logger.setLevel("DEBUG")


console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')


file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')


formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text:str):
    """_summary_


    Args:
        text (str): transforming all the data using nlp techniques
    """
    ps=PorterStemmer()
    text=text.lower()
    text=nltk.word_tokenize(text)
    text=[word for word in text if word.isalnum()]
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text=[ps.stem(word) for word in text]
    return " ".join(text)


def preprocess_df(df,text_column='text',target_column='target'):
    """_summary_


    Args:
        df (_type_): _description_
        text_column (str, optional): _description_. Defaults to 'text'.
        target_column (str, optional): _description_. Defaults to 'target'.


    Returns:
        _type_: returns a data frame
    """
    try:
        logger.debug('starting preprocessing for df')
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug('target column encoded')
       
       
        df=df.drop_duplicates(keep='first')
        logger.debug('duplicates removed')
       
       
        df.loc[:,text_column]=df[text_column].apply(transform_text)
        logger.debug('text transformed')
        return df
    except KeyError as e:
        logger.error('column not found: %s',e)
        raise
    except Exception as e:
        logger.error('error during text normalization: %s',e)
        raise
   
def main(text_column='text',target_column='target'):
    try:
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')
        logger.debug('data loaded properly')
       
       
        train_processed_data=preprocess_df(train_data,text_column,target_column)
        test_processed_data=preprocess_df(train_data,text_column,target_column)
       
       
        data_path=os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
       
       
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
       

