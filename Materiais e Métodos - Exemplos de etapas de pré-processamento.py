import nltk
nltk.download('stopwords')
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.stem import RSLPStemmer

# MATERIAIS E MÉTODOS - PRÉ-PROCESSAMENTO COM EXEMPLO

# Apresentação das etapas comuns de pré-processamento utilizando exemplo da base de dados
texto = """Qualidade esperada para o preço. É dell, então é um bom produto."""

# Convertendo para minúsculas e separando as palavras usando 'split' e 'isalnum' para limpar
palavras = [palavra for palavra in texto.lower().split() if palavra.isalnum()]
print("Transformação do texto em minúsculo:",texto.lower())

# Contar a frequência das palavras e exibir as palavras mais comuns
frequencia = Counter(palavras)
print("Contagem de palavras mais frequentes:",frequencia.most_common(5))

# Tokenização usando ToktokTokenizer
tokenizer = ToktokTokenizer()
tokens = tokenizer.tokenize(texto)
print("Tokenização:", tokens)

#Stemming
stemmer = RSLPStemmer()
stems = [stemmer.stem(p) for p in palavras]
print("Normalização Morfológica (Stemming):", stems)

# Obtendo stopwords do idioma português e filtrando tokens que não estão na lista de stopwords (comparando em minúsculas)
stopwords_portugues = set(stopwords.words('portuguese'))
tokens_filtrados = [token for token in tokens if token.lower() not in stopwords_portugues and token.isalpha()]
print("Remoção de Termos Irregulares (Stopwords) e pontuação:", tokens_filtrados)
