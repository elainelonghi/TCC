import pandas as pd
import requests
import json

def consulta_reviews_produto(id_produto, pagina):
    # Faz um request para a API do mercado livre, buscando os reviews do produto indicado.
    # ATENCAO: necessario regerar o token a cada 6 horas.
    # Salva a resposta do request na variavel response, como um objeto JSON. 
    
    limite_de_reviews = 100
    offset = pagina*limite_de_reviews
    token = 'APP_USR-5756624249112321-083019-11232ffb6d9a19d6146eee038516df2c-48967266'
    headers = {'Authorization': f'Bearer {token}'}
    
    url = f'https://api.mercadolibre.com/reviews/item/{id_produto}?limit={limite_de_reviews}&offset={offset}'
    
    response = requests.request("GET", url, headers=headers).json()
    
    return response


def pre_tratamento_json(response):
    # A partir do objeto JSON, extraimos o atributo de interesse 'reviews' do json
    # E criamos um dataframe com os dados para os produtos listados
    # Extraimos o dataframe para excel
    
    review = response['reviews']
    df = pd.DataFrame.from_dict(review)
    
    df['reviewable_object'] = df['reviewable_object'].apply(lambda x: x['id'])
    return df

product_list = ['MLB4132228593','MLB4830712680','MLB5016573914','MLB3972283557','MLB4001524835','MLB5512463698','MLB4477881510',
                'MLB5107031784','MLB5107118852','MLB4166708923','MLB3609882265','MLB3871416515','MLB4590343328','MLB4908332416',
                'MLB3902275207','MLB5107038044','MLB3902530057','MLB4908356972','MLB3782028017','MLB4908320820','MLB5192934792',
                'MLB3902555023','MLB4908369882','MLB4995897798','MLB4854888140']

df_master = pd.DataFrame()

for product in product_list:
    print(product)
    
    pagina = 0
    
    while True:
        print (pagina)
        response = consulta_reviews_produto(product,pagina)
        try: 
            response['error']
            break
        except: 
            pass
        if len(response['reviews']) == 0: 
            break
        
        df = pre_tratamento_json(response)
        df_master = pd.concat([df_master,df])
        pagina = pagina + 1
    
df_master.to_excel(r'C:\Users\elain\Documents\TCC\reviewsMeli_Dell.xlsx',
                       index = False)
