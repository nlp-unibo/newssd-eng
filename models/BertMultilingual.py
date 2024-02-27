import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

from utils import prepare_data, toLabels


class BertMultilingual:

    def __init__(self, maxSentenceLen):
        self.model = self.create_sentences_model(maxSentenceLen)
        self.maxSentenceLen = maxSentenceLen
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

    def load(self, path):
        self.model.load_weights(path)

    def predict(self, xtest):
        preds = self.model.predict(prepare_data(xtest,
                                                tokenizer=self.tokenizer,
                                                maxSentenceLen=self.maxSentenceLen)[0])
        return toLabels(preds)

    def fit(self, xtrain, ytrain, xval=None, yval=None, weightPath=None):
        x_train, y_train = prepare_data(xtrain,
                                        y=ytrain,
                                        tokenizer=self.tokenizer,
                                        maxSentenceLen=self.maxSentenceLen)
        self.model.fit(x_train,
                       y=y_train,
                       validation_data=prepare_data(xval,
                                                    y=yval,
                                                    tokenizer=self.tokenizer,
                                                    maxSentenceLen=self.maxSentenceLen),
                       batch_size=16,
                       epochs=4)
        if weightPath:
            self.model.save_weights(weightPath)

    @staticmethod
    def create_sentences_model(maxSentenceLen):
        input_ids = tf.keras.layers.Input(shape=(maxSentenceLen,), dtype=tf.int32)
        token_type_ids = tf.keras.layers.Input(shape=(maxSentenceLen,), dtype=tf.int32)
        attention_mask = tf.keras.layers.Input(shape=(maxSentenceLen,), dtype=tf.int32)
        bertModel = TFBertModel.from_pretrained("bert-base-multilingual-cased")(input_ids,
                                                                                token_type_ids=token_type_ids,
                                                                                attention_mask=attention_mask)[-1]
        out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(tf.keras.layers.Dropout(0.1)(bertModel))
        model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=out)
        model.compile(optimizer=tf.optimizers.Adam(1e-5), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        return model
