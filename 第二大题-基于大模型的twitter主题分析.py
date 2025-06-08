# twitter_topic_analysis_deepseek.py
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis.gensim_models
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from ollama import generate
import time
from tqdm import tqdm

# DeepSeek模型配置
MODEL_NAME = "deepseek-r1:1.5b"
MAX_RETRIES = 5
DELAY = 1

# 1. 数据准备
def load_data():
    # 模拟从Excel读取数据（实际使用时替换为pd.read_excel）
    data = {
        'post_text': [
            "Don't need feds to solve the #bostonbombing when we have #4chan!! ...",
            "PIC: Comparison of #Boston suspect Sunil Tripathi's FBI-released...",
            "I'm not completely convinced that it's this Sunil Tripathi fellow—...",
            "Brutal lo que se puede conseguir en colaboración. #4Chan analizando fotos...",
            "@ElbesoenlaLuna: 'Espectacular imag"
            "en del eclipse desde España con la...",
            "Fregenal de la Sierra RT @ElbesoenlaLuna: Espectacular imagen...",
            "“@ElbesoenlaLuna: Espectacular imagen del eclipse desde España con la...",
            "RT @ElbesoenlaLuna: Espectacular imagen del eclipse desde España...",
            "¡Beeeestial! RT @ElbesoenlaLuna: Espectacular imagen del eclipse...",
            "Espectacular imagen del eclipse desde España con la ISS de @ThierryLegault..."
        ],
        'label': ['fake', 'fake', 'fake', 'fake', 'real', 'real', 'real', 'real', 'real', 'real']
    }
    return pd.DataFrame(data)

# 2. 数据预处理
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # 清洗：去除非字母字符并转为小写
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    
    # 分词
    words = text.split()
    
    # 去停用词和短词
    stop_words = set(stopwords.words('english') + ['http', 'https', 'rt'])
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return words

# 3. LDA模型训练
def train_lda(processed_texts, num_topics=2):
    # 构建词典
    dictionary = Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    # 训练LDA模型
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10
    )
    
    return lda_model, corpus, dictionary

# 4. 可视化
def visualize(lda_model, corpus, dictionary):
    # LDAvis交互可视化
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization.html')
    
    # 词云生成
    for topic_id in range(lda_model.num_topics):
        topic_words = dict(lda_model.show_topic(topic_id, 10))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topic {topic_id} Keywords', fontsize=15)
        plt.axis("off")
        plt.savefig(f'topic_{topic_id}_wordcloud.png', bbox_inches='tight')
        plt.close()

# 5. DeepSeek模型分析（替换原GPT-4部分）
def deepseek_analysis(lda_model):
    """使用DeepSeek模型进行主题分析"""
    topics_description = ""
    for topic_id in range(lda_model.num_topics):
        topic_words = [word for word, _ in lda_model.show_topic(topic_id, 10)]
        topics_description += f"Topic {topic_id} keywords: {', '.join(topic_words)}\n"
    
    prompt = f"""请分析以下LDA主题模型结果，用中文解释每个主题可能代表的含义：
    {topics_description}
    
    要求：
    1. 分别说明每个主题的核心内容
    2. 指出关键词之间的关联性
    3. 推测这些主题可能涉及的现实事件"""
    
    for _ in range(MAX_RETRIES):
        try:
            response = generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={"temperature": 0.7}
            )
            analysis_result = response['response'].strip()
            
            with open('deepseek_analysis.txt', 'w', encoding='utf-8') as f:
                f.write(analysis_result)
            return
        except Exception as e:
            print(f"分析出错: {str(e)}")
            time.sleep(DELAY)
    
    print("警告：主题分析失败")

# 主流程
def main():
    # 1. 加载数据
    df = load_data()
    print("原始数据示例:\n", df.head())
    
    # 2. 数据预处理
    df['processed'] = df['post_text'].apply(preprocess_text)
    processed_texts = df['processed'].tolist()
    print("\n预处理示例:", processed_texts[0])
    
    # 3. 训练LDA模型
    lda_model, corpus, dictionary = train_lda(processed_texts)
    print("\n主题展示:")
    for idx, topic in lda_model.print_topics():
        print(f"Topic {idx}: {topic}")
    
    # 4. 可视化
    visualize(lda_model, corpus, dictionary)
    print("\n可视化结果已保存为: lda_visualization.html 和 topic_*_wordcloud.png")
    
    # 5. 使用DeepSeek模型分析
    deepseek_analysis(lda_model)
    print("主题分析结果已保存为: deepseek_analysis.txt")

if __name__ == "__main__":
    main()