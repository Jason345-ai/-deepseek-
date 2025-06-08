import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import re

class TopicFeatureExtractor:
    """封装LDA主题特征提取的类"""
    def __init__(self, texts):
        processed_texts = [self.preprocess_text(text).split() for text in texts]
        self.dictionary = Dictionary(processed_texts)
        corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
        self.lda = LdaModel(corpus=corpus, num_topics=2, id2word=self.dictionary)
    
    def preprocess_text(self, text):
        """预处理文本：移除URL、特殊字符等"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()
    
    def get_topic_vector(self, text):
        """获取文本的主题分布特征"""
        processed_text = self.preprocess_text(text)
        bow = self.dictionary.doc2bow(processed_text.split())
        topic_dist = self.lda.get_document_topics(bow)
        vector = np.zeros(self.lda.num_topics)
        for topic_id, prob in topic_dist:
            vector[topic_id] = prob
        return vector
    
    def visualize_topics(self):
        """可视化主题关键词"""
        for topic_id in range(self.lda.num_topics):
            words = [word for word, _ in self.lda.show_topic(topic_id, topn=5)]
            weights = [weight for _, weight in self.lda.show_topic(topic_id, topn=5)]
            plt.barh(words, weights)
            plt.title(f"Topic {topic_id} Keywords")
            plt.show()

class SentimentFeatureExtractor:
    """封装BERT情感特征提取的类"""
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
    
    def get_bert_embedding(self, text):
        """获取文本的BERT嵌入特征"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

class MultimodalClassifier(nn.Module):
    """多模态分类模型"""
    def __init__(self, bert_dim=768, topic_dim=2):
        super().__init__()
        # 注意力融合层
        self.attn = nn.Sequential(
            nn.Linear(bert_dim + topic_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim + topic_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, bert_input, topic_input):
        combined = torch.cat([bert_input, topic_input], dim=1)
        weights = self.attn(combined)
        attended = combined * weights
        return self.classifier(attended)

class NewsDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_dataset(filepath):
    """加载数据集文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        data = {"text": [], "label": []}
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 2:
                data["text"].append(parts[0])
                data["label"].append(1 if parts[1].lower() == "real" else 0)
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"加载数据集 {filepath} 失败: {str(e)}")
        return None

def main():
    # 1. 设置数据集路径
    devset_path = r"C:\Users\runni\OneDrive\文档\大学\大二下\学科\云计算与大数据\楷滨—皮卡丘—单\9646418e3675e7d02d558d3dad3c18e7_87633e15c15e68e5b7f6ca618b84f351_8\新建文件夹\twitter_dataset\devset\posts.dev - 副本.txt"
    testset_path = r"C:\Users\runni\OneDrive\文档\大学\大二下\学科\云计算与大数据\楷滨—皮卡丘—单\9646418e3675e7d02d558d3dad3c18e7_87633e15c15e68e5b7f6ca618b84f351_8\新建文件夹\twitter_dataset\devset\posts.test- 副本.txt"
    
    # 2. 加载开发集和测试集
    dev_df = load_dataset(devset_path)
    test_df = load_dataset(testset_path)
    
    if dev_df is None or test_df is None:
        print("数据集加载失败，请检查文件路径和格式")
        return
    
    print(f"开发集样本数: {len(dev_df)}")
    print(f"测试集样本数: {len(test_df)}")
    
    # 3. 特征提取（仅在开发集上训练LDA）
    print("正在提取特征...")
    topic_extractor = TopicFeatureExtractor(dev_df["text"].tolist())
    sentiment_extractor = SentimentFeatureExtractor()
    
    # 提取开发集特征
    dev_df["topic_feat"] = dev_df["text"].apply(topic_extractor.get_topic_vector)
    dev_df["bert_feat"] = dev_df["text"].apply(sentiment_extractor.get_bert_embedding)
    
    # 提取测试集特征（使用开发集训练的LDA模型）
    test_df["topic_feat"] = test_df["text"].apply(topic_extractor.get_topic_vector)
    test_df["bert_feat"] = test_df["text"].apply(sentiment_extractor.get_bert_embedding)
    
    # 4. 准备数据
    X_dev = np.hstack([np.stack(dev_df["bert_feat"]), np.stack(dev_df["topic_feat"])])
    y_dev = dev_df["label"].values
    
    X_test = np.hstack([np.stack(test_df["bert_feat"]), np.stack(test_df["topic_feat"])])
    y_test = test_df["label"].values
    
    # 创建数据加载器
    train_loader = DataLoader(NewsDataset(X_dev, y_dev), batch_size=8, shuffle=True)
    test_loader = DataLoader(NewsDataset(X_test, y_test), batch_size=8)
    
    # 5. 训练模型
    model = MultimodalClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    
    print("\n开始训练...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X[:, :768], batch_X[:, 768:])
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 每个epoch后评估训练集和测试集
        model.eval()
        train_correct, test_correct = 0, 0
        with torch.no_grad():
            # 训练集准确率
            for batch_X, batch_y in train_loader:
                preds = model(batch_X[:, :768], batch_X[:, 768:])
                train_correct += (preds.argmax(1) == batch_y).sum().item()
            
            # 测试集准确率
            for batch_X, batch_y in test_loader:
                preds = model(batch_X[:, :768], batch_X[:, 768:])
                test_correct += (preds.argmax(1) == batch_y).sum().item()
        
        train_acc = train_correct / len(train_loader.dataset)
        test_acc = test_correct / len(test_loader.dataset)
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}")
    
    # 6. 最终评估和可视化
    print("\n训练完成，最终评估：")
    model.eval()
    with torch.no_grad():
        test_correct = 0
        for batch_X, batch_y in test_loader:
            preds = model(batch_X[:, :768], batch_X[:, 768:])
            test_correct += (preds.argmax(1) == batch_y).sum().item()
        print(f"最终测试集准确率: {test_correct / len(test_loader.dataset):.2%}")
    
    # 可视化主题
    topic_extractor.visualize_topics()
    
    # 示例预测
    sample_texts = [
        "4chan users identified the wrong suspect in Boston bombing",
        "NASA confirms the ISS transit during eclipse was real"
    ]
    
    print("\n示例预测：")
    for text in sample_texts:
        bert_feat = sentiment_extractor.get_bert_embedding(text)
        topic_feat = topic_extractor.get_topic_vector(text)
        input_tensor = torch.cat([
            torch.FloatTensor(bert_feat).unsqueeze(0),
            torch.FloatTensor(topic_feat).unsqueeze(0)
        ], dim=1)
        with torch.no_grad():
            prob = model(input_tensor[:, :768], input_tensor[:, 768:])
        print(f"\n文本: '{text}'")
        print(f"虚假新闻概率: {prob[0][0]:.2%}")
        print(f"真实新闻概率: {prob[0][1]:.2%}")

if __name__ == "__main__":
    main()