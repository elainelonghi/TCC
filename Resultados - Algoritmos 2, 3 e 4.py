import nltk
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('punkt')      
nltk.download('wordnet')
nltk.download('omw-1.4') 
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from nltk.util import ngrams
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import RSLPStemmer
import spacy
import string
nlp = spacy.load("pt_core_news_sm")
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pio.renderers.default='browser'
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random
import re
import pandas as pd
import random
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC




# RESULTADOS

# ALGORITMO 2: PRÉ-PROCESSAMENTO COM A BASE COMPLETA
# Importando o banco de dados e verificando suas características
df = pd.read_excel(r'C:\Users\elain\OneDrive\Documents\2025\TCC\reviewsMeli_Dell.xlsx')
df.info()
print()

# Remover atributos não importantes
df = df.drop(columns=[
    'id',
    'date_created',
    'status',
    'reviewer_id',
    'valorization',   
    'buying_date',
    'relevance',
    'forbidden_words',
    'attributes',
    'media',
    'reactions',
    'attributes_variation',
    'secondary_key',
    'routing_key',
    'order_id',
    'catalog_listing',
    'condition',
    'document_version',
    'item_status',
    'coupon_redeemed',
    'earned_rewards',
    'bucket'
])
df.info()
print()

# Contar quantos comentários são nulos
nulos = df['content'].isna().sum()

# Contar quantos comentários NÃO são nulos
nao_nulos = df['content'].notna().sum()

print(f"Comentários NULOS: {nulos}")
print(f"Comentários NÃO NULOS: {nao_nulos}")
# Conclusão: Temos comentários nulos

# Renomear as colunas
df = df.rename(columns={
    'reviewable_object': 'ID Produto',
    'title': 'Título',
    'content': 'Opinião',
    'rate': 'Nota',
    'likes': 'Likes',
    'dislikes': 'Dislikes'
})
# Remover linhas com comentários nulos
df = df.dropna(subset=['Opinião'])
df.info()
print()

# Análise de frequência de tokens relevantes

# Iterar sobre cada depoimento
tokens_totais = []
stopwords_portugues = set(stopwords.words('portuguese'))
for content in df['Opinião']:
    # Tokenização do depoimento
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(content)
    # Converter para minúsculas e filtrar tokens:
    #  - Remover stopwords
    #  - Manter apenas tokens que contenham apenas letras (ignora números e pontuações)
    tokens_filtrados = [token.lower() for token in tokens
                         if token.lower() not in stopwords_portugues and token.isalpha()]
    tokens_totais.extend(tokens_filtrados)

# Contar a frequência dos tokens
frequencia = Counter(tokens_totais)

# Exibir as 100 palavras mais comuns
top_100 = frequencia.most_common(100)
print("Palavras mais comuns:",top_100)

# Gerar a nuvem de palavras com os top 100
top_100 = dict(frequencia.most_common(100))
wordcloud = WordCloud(width=800, height=400, background_color='white', random_state=100).generate_from_frequencies(top_100)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Análise de frequência de tokens relevantes após stemming
# Inicializa o stemmer para português
stemmer = RSLPStemmer()

# Dicionário para armazenar o stem e a contagem agregada
stemmed_counts = {}

for word, count in top_100.items():
    # Aplica o stemming na palavra
    stem = stemmer.stem(word)
    # Agrega a contagem para o stem correspondente
    stemmed_counts[stem] = stemmed_counts.get(stem, 0) + count

print(stemmed_counts)

# Gerar a nuvem de palavras com os top 100 após stemming
wordcloud = WordCloud(width=800, height=400, background_color='white', random_state=100).generate_from_frequencies(stemmed_counts)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Criação de colunas de tokens relevantes -- NÃO USEI NO TRABALHO
# Função para tokenização e removeção da pontuação e stopwords em cada comentário
tokens_totais = []
pontuacoes = set(string.punctuation)
stopwords_portugues = set(stopwords.words('portuguese'))
def limpar_tokens(texto):
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(str(texto))
    tokens_filtrados = [
        token.lower() for token in tokens
        if token.lower() not in stopwords_portugues
        and token not in pontuacoes
        and token.isalpha()
    ]
    return tokens_filtrados
    tokens_totais.extend(tokens_filtrados)

# Criar coluna de palavras tokenizadas
df['Tokenized'] = df['Opinião'].apply(limpar_tokens)

# Função para stemização em cada comentário
pontuacoes = set(string.punctuation)
stopwords_portugues = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()
def stemmer_tokens(texto):
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(str(texto))
    tokens_stemmed = [
        stemmer.stem(token.lower()) for token in tokens
        if token.lower() not in stopwords_portugues
        and token not in pontuacoes
        and token.isalpha()
    ]
    return tokens_stemmed

# Criar coluna de palavras após stemização
df['Stemmed'] = df['Opinião'].apply(stemmer_tokens)

# Função para Lemmatização:
def lematizar_texto(texto):
    doc = nlp(str(texto))
    return [
        token.lemma_.lower() for token in doc
        if token.is_alpha 
        and not token.is_stop 
        and not token.is_punct
    ]

# Criar colunas de palavras e frases após lemmatização
df['Lemmatized'] = df['Opinião'].apply(lematizar_texto)
df['Lemmatized_phrases'] = df['Lemmatized'].apply(lambda tokens: ' '.join(tokens))

#-------------------------------------------------------------------------------------

# ALGORITMO 3: Classificação e Mineração

# Contagem das notas
rate_counts = df["Nota"].value_counts().sort_index()
total = rate_counts.sum()

# Criar gráfico de barras
gray_scale = ['#d9d9d9', '#bdbdbd', '#969696', '#636363', '#252525'] #Tons de cinza
text_labels = [
    f"{count} ({count/total:.1%})"
    for count in rate_counts.values
]
fig = go.Figure(data=[
    go.Bar(
        x=rate_counts.index,
        y=rate_counts.values,
        marker_color=gray_scale,
        text=text_labels,
        textposition='outside',  # Exibe os rótulos acima das barras
        textfont=dict(
            size=22,       # Tamanho da fonte
            color='black', # Cor da fonte
            family='Arial' # Fonte (pode ser 'Verdana', 'Courier New', etc.)
        )
    )
])
fig.update_layout(
    xaxis_title="Notas",
    yaxis_title="Quantidade",
    template="simple_white",
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray')
)
fig.show()
# Conclusão: o campo título é definido automaticamente quando a nota é selecionada e obrigatório
# 1 = Muito Ruim
# 2 = Ruim
# 3 = Bom
# 4 = Muito Bom
# 5 = Excelente


# Tabela de palavras mais frequentes por nota
stopwords_pt = set(stopwords.words('portuguese'))
def termos_detalhados_por_nota(df, nota_col='Nota', texto_col='Opinião', top_n=5):
    resultado = []

    for nota in sorted(df[nota_col].unique()):
        comentarios = df[df[nota_col] == nota][texto_col].dropna().astype(str)

        palavras = []
        for texto in comentarios:
            tokens = nltk.word_tokenize(texto.lower(), language='portuguese')
            palavras += [t for t in tokens if t.isalpha() and t not in stopwords_pt]

        total_palavras = len(palavras)
        contagem = Counter(palavras).most_common(top_n)

        termos = [palavra for palavra, _ in contagem]
        ocorrencias = [freq for _, freq in contagem]
        proporcoes = [
            f"{(freq / total_palavras * 100):.2f}%"
            if total_palavras > 0 else "0.00%"
            for freq in ocorrencias
        ]

        resultado.append({
            'Nota de Avaliação': nota,
            'Total de Palavras (Nota)': total_palavras,
            'Termos Mais Frequentes': ', '.join(termos),
            'Ocorrências por Termo': f"[{', '.join(map(str, ocorrencias))}]",
            'Frequência Relativa por Termo (%)': f"[{', '.join(proporcoes)}]"
        })

    return pd.DataFrame(resultado)
# Gerar a tabela
tabela_detalhada = termos_detalhados_por_nota(df)
print(tabela_detalhada)

# Análise de bigramas contendo a palavra tela nas avaliações com nota 1
stopwords_pt = stopwords.words('portuguese')
df_nota_1 = df[df["Nota"] == 1]

# Extrair o texto das opiniões
corpus = df_nota_1["Opinião"].astype(str).tolist()

# Configurar o CountVectorizer para bigrams com stopwords personalizadas
vectorizer = CountVectorizer(ngram_range=(2, 2), lowercase=True, stop_words=stopwords_pt)

# Ajustar e transformar o corpus
X = vectorizer.fit_transform(corpus)

# Obter os bigrams e suas frequências
bigrams = vectorizer.get_feature_names_out()
frequencias = X.toarray().sum(axis=0)
df_bigrams = pd.DataFrame({'bigrama': bigrams, 'frequencia': frequencias})

# Filtrar os bigrams que contêm o termo "tela" e exibir por ordem de frequência
df_tela = df_bigrams[df_bigrams['bigrama'].str.contains('tela')]
df_tela = df_tela.sort_values(by='frequencia', ascending=False)
print(df_tela)

# Gerar nuvem de palavras
bigrams_dict = dict(zip(df_tela['bigrama'], df_tela['frequencia']))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bigrams_dict)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuvem de Bigramas com "tela"')
plt.show()

# ANÁLISE DE SENTIMENTOS

# Mapear notas para sentimentos (1 e 2 = Negativo e 5 = Positivo)
df_filtrado = df[df['Nota'].isin([1, 2, 5])].copy()
mapeamento_sentimento = {1: 'negativo', 2: 'negativo', 5: 'positivo'}
df_filtrado['Sentimento'] = df_filtrado['Nota'].map(mapeamento_sentimento)
def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Aplicar limpeza ao corpus
corpus = df_filtrado['Opinião'].astype(str).apply(limpar_texto).tolist()
labels = df_filtrado['Sentimento'].tolist()

# Dividir o corpus e os rótulos
corpus_train, corpus_test, y_train, y_test = train_test_split(
    corpus, labels, test_size=0.2, random_state=42
)

# Vetorização com unigramas + bigramas
vectorizer = CountVectorizer(lowercase=True, stop_words=stopwords_pt, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(corpus_train)
X_test = vectorizer.transform(corpus_test)

# Função para avaliar modelo
def avaliar_modelo(nome, modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    matriz = confusion_matrix(y_test, y_pred)
    return {
        'Modelo': nome,
        'Acurácia (%)': round(acuracia * 100, 2),
        'F1-Score': round(f1, 2),
        'Matriz de Confusão': matriz
    }

# Avaliar os três modelos
resultados = []

# Naive Bayes
modelo_nb = MultinomialNB()
resultados.append(avaliar_modelo("Naive Bayes", modelo_nb, X_train, y_train, X_test, y_test))

# Regressão Logística
modelo_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
resultados.append(avaliar_modelo("Regressão Logística", modelo_lr, X_train, y_train, X_test, y_test))

# SVM
modelo_svm = SVC(kernel='linear', class_weight='balanced')
resultados.append(avaliar_modelo("SVM", modelo_svm, X_train, y_train, X_test, y_test))

# Exibir tabela resumida
print("\n Tabela Comparativa dos Modelos:")
for r in resultados:
    print(f"\n🔹 {r['Modelo']}")
    print(f"Acurácia: {r['Acurácia (%)']}%")
    print(f"F1-Score: {r['F1-Score']}")
    print("Matriz de Confusão:")
    print(r['Matriz de Confusão'])

# Pesos das variáveis da Regressão Logística
pesos = modelo_lr.coef_[0]
termos = vectorizer.get_feature_names_out()
import pandas as pd
df_pesos = pd.DataFrame({'Termo': termos, 'Peso': pesos})

# Ordenar pelos pesos positivos (maior influência positiva)
df_positivos = df_pesos.sort_values(by='Peso', ascending=False).head(10)
print("\n🔍 Termos com maior influência positiva:")
print(df_positivos)

# Ordenar pelos pesos negativos (maior influência negativa)
df_negativos = df_pesos.sort_values(by='Peso', ascending=True).head(10)
print("\n🔍 Termos com maior influência negativa:")
print(df_negativos)



