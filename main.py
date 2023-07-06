from fastapi import FastAPI
import pandas as pd
from unicodedata import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from datetime import datetime
import scipy.sparse as sp


app = FastAPI()


# cargo el dataset limpio
df = pd.read_csv('movie_dataset_clean(by_alexDRandom).csv')


@app.get('/')
def index():   
    return {'message': '尺闩ﾁ七〤ㄖ ﾁ闩丂セ 闩尸讠',
           'description': 'Implementación en Render de la API con FastAPI',
          'organization': 'soyHENRY.com',
               'carreer': 'soyHenry bootcamp DATA SCIENCE',
               'student': 'Rafal Janusz Wysocki',
           'mail/github': 'raftxo.mx@gmail.com', 
           }

'''
# Esqueleto funciones PI01 DTS-12
@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma:str):
    '''Ingresas el idioma, retornando la cantidad de peliculas producidas en el mismo'''
    return {'idioma':idioma, 'cantidad':respuesta}
    
@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula:str):
    '''Ingresas la pelicula, retornando la duracion y el año'''
    return {'pelicula':pelicula, 'duracion':respuesta, 'anio':anio}

@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''
    return {'franquicia':franquicia, 'cantidad':respuesta, 'ganancia_total':respuesta, 'ganancia_promedio':respuesta}

@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo'''
    return {'pais':pais, 'cantidad':respuesta}

@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    '''Ingresas la productora, entregandote el revunue total y la cantidad de peliculas que realizo '''
    return {'productora':productora, 'revenue_total': respuesta,'cantidad':respuesta}


@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):
    ''' Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
    Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma. En formato lista'''
    return {'director':nombre_director, 'retorno_total_director':respuesta, 
    'peliculas':respuesta, 'anio':respuesta,, 'retorno_pelicula':respuesta, 
    'budget_pelicula':respuesta, 'revenue_pelicula':respuesta}

# ML
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    return {'lista recomendada': respuesta}

# =====================
'''

@app.get('/peliculas_mes/{mes}')
def peliculas_mes(mes: str):
    '''
    Retorna la cantidad de películas que se estrenaron en un mes específico.

    Args:
        mes (str): El nombre del mes para el cual se desea obtener la información. 
        Puede ser el nombre del mes en inglés o en español.

    Returns:
        dict: Un diccionario que contiene el nombre del mes (en formato capitalizado), 
        y la cantidad de películas que se estrenaron en ese mes.

    '''
    # diccionario de mapeo de meses en inglés a español
    meses_ingles = {
        'January': 'enero',
        'February': 'febrero',
        'March': 'marzo',
        'April': 'abril',
        'May': 'mayo',
        'June': 'junio',
        'July': 'julio',
        'August': 'agosto',
        'September': 'septiembre',
        'October': 'octubre',
        'November': 'noviembre',
        'December': 'diciembre'
    }
    
    # diccionario de mapeo de meses en español a inglés
    meses_espanol = {v: k for k, v in meses_ingles.items()}
    
    # convertir la columna 'release_date' a tipo fecha
    df['release_date'] = pd.to_datetime(df['release_date'])

    # obtener el nombre del mes en minúsculas
    df['mes'] = df['release_date'].dt.month_name().str.lower()
    
    # verificar si se proporcionó el mes en español
    if mes.lower() in meses_espanol:
        # filtrar el DataFrame por el mes en inglés correspondiente
        df_mes = df[df['mes'] == meses_espanol[mes.lower()].lower()]
        nombre_mes = mes.capitalize()
    else:
        # filtrar el DataFrame por el mes en inglés
        df_mes = df[df['mes'] == mes.lower()]
        nombre_mes = meses_ingles[df_mes['mes'].iloc[0]].capitalize()
    
    # obtener la cantidad de películas para ese mes
    cantidad = len(df_mes)

    
    return {'mes': nombre_mes, 'cantidad': cantidad}


@app.get('/peliculas_dia/{dia}')
def peliculas_dia(dia: str):
    '''
    Retorna la cantidad de películas que se estrenaron en un día de la semana específico.

    Args:
        dia (str): El nombre del día de la semana para el cual se desea obtener la información. 
        Puede ser el nombre del día en inglés o en español.

    Returns:
        dict: Un diccionario que contiene el nombre del día de la semana (en formato capitalizado), 
        y la cantidad de películas que se estrenaron en ese día.

    '''
    
    # diccionario de mapeo de días en inglés a español
    dias_ingles = {
        'Monday': 'lunes',
        'Tuesday': 'martes',
        'Wednesday': 'miércoles',
        'Thursday': 'jueves',
        'Friday': 'viernes',
        'Saturday': 'sábado',
        'Sunday': 'domingo'
    }
    
    # diccionario de mapeo de días en español a inglés
    dias_espanol = {unidecode(v): k for k, v in dias_ingles.items()}
    
    # convertir la columna 'release_date' a tipo fecha
    df['release_date'] = pd.to_datetime(df['release_date'])
    
    # obtener el día de la semana para cada fecha en inglés
    df['day_of_week'] = df['release_date'].dt.day_name()
    
    # filtrar el DataFrame por el día de la semana especificado en español
    df_dia = df[df['day_of_week'] == dias_espanol[unidecode(dia.lower())]]
    
    # obtener el nombre del día de la semana en formato capitalizado
    nombre_dia = dia.capitalize()

    # obtener la cantidad de películas para ese día
    cantidad = len(df_dia)

    return {'dia_semana': nombre_dia, 'cantidad': cantidad}


@app.get('/franquicia/{franquicia}')
def franquicia(franquicia):
    '''
    Retorna la cantidad de películas, ganancia total y ganancia promedio para una franquicia específica.

    Args:
        franquicia (str): El nombre de la franquicia de películas.

    Returns:
        dict: Un diccionario que contiene el nombre de la franquicia (en formato capitalizado), 
        la cantidad de películas en la franquicia,
        la ganancia total de la franquicia formateada con separadores de miles, 
        y la ganancia promedio de la franquicia formateada con separadores de miles.
    
    '''

    # filtrar las películas que pertenecen a la franquicia especificada
    peliculas_franquicia = df[df['belongs_to_collection'].str.contains(franquicia, case=False, na=False)]

    # obtener la cantidad de películas para esa franquicia
    cantidad = len(peliculas_franquicia)

    # calcular la ganancia total y el promedio de ganancia para esa franquicia
    ganancia_total = peliculas_franquicia['revenue'].sum()
    ganancia_promedio = peliculas_franquicia['revenue'].mean()

    return {'franquicia': franquicia.title(), 'cantidad': cantidad, 'ganancia_total': f'{ganancia_total:,}', 'ganancia_promedio': f'{ganancia_promedio:,}'}


@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais):
    '''
    Retorna la cantidad de películas producidas en un país específico.

    Args:
        pais (str): El nombre del país.

    Returns:
        dict: Un diccionario que contiene el nombre del país (en formato capitalizado) y 
        la cantidad de películas producidas en ese país.

    '''

    # filtrar las películas producidas en el país especificado
    peliculas_pais = df[df['production_countries'].str.contains(pais, case=False, na=False)]

    # obtener la cantidad de películas producidas en ese país
    cantidad = len(peliculas_pais)

    return {'pais': pais.title(), 'cantidad': cantidad}


@app.get('/productoras/{productora}')
def productoras(productora:str):
    '''
    Retorna la ganancia total y la cantidad de películas producidas por una productora específica.

    Args:
        productora (str): El nombre de la productora.

    Returns:
        dict: Un diccionario que contiene el nombre de la productora (en formato capitalizado), 
        la ganancia total de las películas producidas por la productora formateada con separadores de miles,
        y la cantidad de películas producidas por la productora.
    
    '''

# filtrar el DataFrame por las filas que contienen la productora especificada en la columna 'production_companies'
    filtered_df = df[df['production_companies'].str.contains(productora, case=False, na=False)]

    
    # calcular el total de 'revenue' y se cuenta el numero de peliculas 
    ganancia_total = filtered_df['revenue'].sum()
    cantidad = filtered_df.shape[0] 

    return {'productora': productora.title(), 'ganancia_total': f'{ganancia_total:,}', 'cantidad': cantidad}

@app.get('/retorno/{pelicula}')
def retorno(pelicula:str):
    """
    Retorna información sobre una película específica.

    Args:
        pelicula (str): El título de la película.

    Returns:
        dict: Un diccionario con los siguientes valores:
            - 'pelicula' (str): El título de la película en formato capitalizado.
            - 'inversion' (str): La inversión de la película formateada con separadores de miles.
            - 'ganancia' (str): La ganancia de la película formateada con separadores de miles.
            - 'retorno' (str): El retorno de la película redondeado a 2 decimales y formateado con separadores de miles.
            - 'anio' (int): El año en el que se lanzó la película.

    """
    

    # filtrar el DataFrame por el título de la película especificada
    pelicula_filtrada = df[df['title'].str.contains(pelicula, case=False, na=False)]

    # obtener los valores de inversión, ganancia, retorno y año de lanzamiento
    inversion = pelicula_filtrada['budget'].values[0]
    ganancia = pelicula_filtrada['revenue'].values[0]
    retorno = round(pelicula_filtrada['return'].values[0],2)
    anio = int(pelicula_filtrada['release_year'].values[0])
    
    return {'pelicula': pelicula.title(), 'inversion': f'{inversion:,}', 'ganancia': f'{ganancia:,}', 'retorno': f'{retorno:,}', 'anio': anio}


#ML

# hacemos un fillna para rellenar los vacios de nuevo, ya que me dio problemas cuando cargue de nuevo el dataset
df.fillna({'overview': '', 'tagline': '', 'genres': '', 'belongs_to_collection': ''}, inplace=True)

def preprocess_text(text):
    """
    Realiza el preprocesamiento de un texto dado.
    
    Args:
        text (str): El texto a preprocesar.
    
    Returns:
        str: El texto preprocesado.
    
    """

    # convierte el texto a minúsculas
    text = text.lower()

    # elimina los caracteres de puntuación
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    # aplica stemming o lematización si es necesario
    
    return text

# preprocesado de las columnas overview, tagline, genres y belong_to_collection
df['processed_text'] = df['overview'] + ' ' + df['tagline'] + ' ' + df['belongs_to_collection'] + ' ' + df['genres']
df['processed_text'] = df['processed_text'].map(preprocess_text)

# calcula TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_alpha = tfidf.fit_transform(df['processed_text'])
tfidf_matrix_beta = tfidf.fit_transform(df['title']) / 1.5
tfidf_matrix_gama = tfidf.fit_transform(df['genres']) * 1.15

tfidf_matrix = sp.hstack([tfidf_matrix_alpha, tfidf_matrix_beta, tfidf_matrix_gama]).tocsr()




@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    """
    Lleva a cabo la recomendación de películas basada en la similitud de contenido utilizando TF-IDF.

    Args:
        titulo (str): El título de la película de referencia para la cual se realizará la recomendación.

    Returns:
        list: Una lista que contiene los títulos de las películas recomendadas con mayor similitud de contenido.
    """

    movie = df[df['title'].str.contains(titulo, case=False, na=False)]
    if movie.empty:
        return [{'error': 'Movie not found'}]
    
    movie_index = movie.index[0]
    movie_vector = tfidf_matrix.getrow(movie_index)
    
    # Calcula la similitud del coseno entre la película de entrada y todas las demás películas
    cosine_similarities = linear_kernel(movie_vector, tfidf_matrix).flatten()
    
    # Obtiene los índices de las películas ordenadas por puntajes de similitud
    similar_movie_indices = cosine_similarities.argsort()[::-1]
    
    # Filtra la propia película de entrada
    similar_movie_indices = similar_movie_indices[similar_movie_indices != movie_index]
    
    # Filtra las películas con un puntaje de similitud inferior a 0.35
    high_similarity_indices = similar_movie_indices[cosine_similarities[similar_movie_indices] >= 0.2]
    
    # Obtiene los títulos de las películas con mayor similitud
    recommended_movies = df.iloc[high_similarity_indices[:5]]['title'].tolist()
    
    return recommended_movies
