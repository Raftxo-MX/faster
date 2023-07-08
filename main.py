from fastapi import FastAPI
import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from datetime import datetime
import scipy.sparse as sp


app = FastAPI()


# cargo el dataset limpio
df = pd.read_csv('./datos_limpios.csv')


@app.get('/')
def index():   
    """
    尺闩ﾁ七〤ㄖ ﾁ闩丂セ 闩尸讠<br>
    Implementación de la API con FastAPI en Render<br>
    PI01_MLops DTS-12<br>
    Rafa J.W.
    """
    return {   'message': '尺闩ﾁ七〤ㄖ ﾁ闩丂セ 闩尸讠',
           'description': 'Implementación de la API con FastAPI en Render',
          'organization': 'soyHENRY.com',
               'carreer': 'soyHenry bootcamp DATA SCIENCE',
               'project': 'PI01_MLops',
                'cohort': 'DTS-12',
               'student': 'Rafal Janusz Wysocki',
           'mail/github': 'raftxo.mx@gmail.com', 
           }

@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma:str):
    '''
    Ingresa el idioma en formato abreviado conforme la norma internacional ISO 639-1.<br> 
    Los idiomas más populares son:<br>
        "en" corresponde a inglés (English).<br>
        "fr" corresponde a francés (French).<br>
        "it" corresponde a italiano (Italian).<br>
        "ja" corresponde a japonés (Japanese).<br>
        "de" corresponde a alemán (German).<br>
        "es" corresponde a español (Spanish).<br>
        "ru" corresponde a ruso (Russian).<br>
        "hi" corresponde a hindi (Hindi).<br>
        "ko" corresponde a coreano (Korean).<br>
        "zh" corresponde a chino (Chinese).<br>
        "pl" corresponde a polaco (Polish)<br>
        "ar" corresponde a argentino viste...<br>
<br>
    La función devolverá:<br>
        Un diccionario {'idioma':idioma, 'cantidad':respuesta}.<br>

    '''
    df_filtrado=df[df['original_language']==idioma]
    cantidad=len(df_filtrado)
    respuesta = {'idioma':idioma, 'cantidad':cantidad}
  
    return respuesta

@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula:str):
    '''
    Ingresas un título de una pelicula y verás su duración en minutos y el año de estreno.
    '''
    # Creamos una lista vacía par almacenar los resultados
    resultados=[]

    # Filtramos las filas con el título especificado
    df_filtrado = df[df['title'].str.contains(pelicula, case=False)]

    # Verificamos si hay resultados
    if len(df_filtrado) == 0:
        mensaje_error = {'error': f'No se encontró ninguna película que contenga la palabra "{texto}"'}
        return mensaje_error

    # Iteramos sobre las filas filtradas y agregamos los datos solicitados en el ejercicio
    for indice, fila in df_filtrado.iterrows():
        resultado = {'titulo':fila['title'], 'duracion':fila['runtime'], 'anno':fila['release_year']}
        resultados.append(resultado)

    return resultados

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

        ['index', 'id', 'budget', 'original_language', 'overview', 'popularity',
       'release_date', 'revenue', 'runtime', 'spoken_languages', 'status',
       'tagline', 'title', 'vote_average', 'vote_count', 'return',
       'release_year', 'directed_by', 'franchise', 'produced_by',
       'produced_in', 'genres_clean']
    
    '''

    # filtrar las películas que pertenecen a la franquicia especificada
    peliculas_franquicia = df[df['franchise'].str.contains(franquicia, case=False, na=False)]

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
    peliculas_pais = df[df['produced_in'].str.contains(pais, case=False, na=False)]

    # obtener la cantidad de películas producidas en ese país
    cantidad = len(peliculas_pais)

    return {'pais': pais.title(), 'cantidad': cantidad}


@app.get('/productoras_exitosas/{productora}')
def productoras(productora:str):
    '''
    Ingresa la productora para ver su revenue total y la cantidad de películas que realizó.

    Algunas productoras famosas:  <br>


    return {'productora':productora, 'revenue_total':X, 'cantidad':Y}
    
    '''

# filtrar el DataFrame por las filas que contienen la productora especificada en la columna 'produced_by'
    filtered_df = df[df['produced_by'].str.contains(productora, case=False, na=False)]

    
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


# ML - to do

