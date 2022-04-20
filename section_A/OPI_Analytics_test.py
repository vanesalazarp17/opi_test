#!/usr/bin/env python
# coding: utf-8

# ## OPI Analytics Test
# ##### Vanessa Salazar
# 
# 
# ### Section A

# ##### Libraries

# In[74]:


get_ipython().run_line_magic('matplotlib', 'inline')
import sys

# Basic Analysis and Visualization
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import pandas_profiling
import seaborn as sns
import math
from datetime import timedelta

# Mapping
import folium
from folium import plugins
from folium.plugins import MarkerCluster


# ##### Read input dataset

# In[20]:


data = pd.read_csv("fgj_data_opitest.csv")
data.head()


# In[21]:


data.describe()


# In[22]:


df = data.copy()
df.shape


# In[23]:


df = df.dropna(subset=["fecha_hechos", "fecha_inicio"])
df.shape


# In[24]:


df["fecha_hechos"] = pd.to_datetime(df["fecha_hechos"])
df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"])
df = df.astype({"ao_hechos": int})


# In[25]:


df.drop(axis=1, columns=["tempo"], inplace=True)


# In[26]:


df = df[(df.ao_hechos >= 2010) & (df.ao_hechos < 2022)]


# In[27]:


df.shape


# #### ¿Qué pruebas identiﬁcarías para asegurar la calidad de estos datos?

# Al hablar de calidad en los datos es importante tomar como punto de partida los issues standard en términos de calidad que suelen presentarse y tratar de resolver aquellos que se apliquen a la características de los datos en análisis.
# 
# En este dataset en particular, observamos basicamente 3 tipos de datos:
# 
# - datetime (fecha_hechos, fecha_inicio)
# 
# - int64 (ao_hechos, ao_inicio, longitud, latitud)
# 
# - object (meses, delito y sus dependencias)
# 
# 
# Es importarte verificar la completitud del dataset, obviando los datos que contienen valores vacios, especialmente en columnas trascendentes como son fecha_hechos, delito, alcaldia (para este ejercicio en particular).
# 
# Además, es necesario verificar la coherencia de los datos, verificar si las fechas son correctas tanto en rangos numericos permitidos, como en formato.
# 
# Verificar que los valores de latitud y longitud correspondan a la ubicacion tambien definida (alcaldia, colonia por ejemplo).
# 
# Al existir muchos campos de texto, es importante verificar la unificacion en el vocabulario, para verificar errores que pueden darse como la duplicacion de categorias por mala escritura.
# 
# Ademas, se pueden emplear herramientas disenadas para proveer reportes de analisis de los datos incluidos en el dataset, que resultan muy utiles para emitir un diagnostico, como pandas_profiling, por ejemplo.

# #### Identiﬁca los delitos que van a la alza y a la baja en la CDMX

# Para este analsis, se consideran unicamente los delitos representativos. Para esto se podrian plantear varios criterios de seleccion de delitos, sin embargo dada la premura del ejercicio, en este caso los datos representativos se eligen bajo dos consideraciones:
# 
# 1. Que el delito incluya data de al menos los ultimos 5 años
# 2. Que el numero de ocurrencias que tienen estos delitos sea mayor a la mediana calculada sobre los datos totales.
# 
# De esta manera el numero de delitos en analisis se reduce a 110 de un total de 347.
# 
# Ademas, para determinar si un delito va a la alza o la baja, se ha calculado el pocentaje de cambio de cada delito dentro de un periodo de tiempo, para este periodo se ha tomando en consideracion los ultimos 5 anios, al ser este periodo el que incluye data mas representativa de la variacion del comportamiento de los delitos analizados.  El promedio de este porcentaje de cambio en un periodo de 5 anios, definira si el delito va a la alza o a la baja.

# In[458]:


def get_percentile(df_col):
    mean_val = df_col.mean()
    med_val = df_col.median()
    q10 = np.percentile(df_col, 10)
    return [round(mean_val), round(med_val), round(q10)]


# In[459]:


relevant_crimes = []
irrelevant_crimes = []
for delito in df["delito"].value_counts().index:
    years = df[df.delito == delito]["ao_hechos"].value_counts().sort_index().index
    if (
        (2021 in years)
        and (2020 in years)
        and (2019 in years)
        and (2018 in years)
        and (2017 in years)
    ):
        relevant_crimes.append(delito)
    else:
        irrelevant_crimes.append(delito)


# In[460]:


df = df.loc[df["delito"].isin(relevant_crimes)]


# In[461]:


metrics_crime = get_percentile(df.delito.value_counts())
metrics_crime


# In[462]:


len(df.groupby("delito")["delito"].count().sort_values())


# In[463]:


relevant_ocurrence = (
    df.delito.value_counts().loc[lambda x: x >= metrics_crime[1]].index.tolist()
)
df = df.loc[df["delito"].isin(relevant_ocurrence)]
len(relevant_ocurrence)


# In[471]:


df.delito.value_counts().sort_values()[0:30].plot(
    kind="barh", title="Tipos de delito", figsize=(8, 8)
)


# In[472]:


# plt.figure(figsize=(40, 40))
df_pct_change = pd.DataFrame()
for i, delito in enumerate(df.delito.value_counts().sort_values().index):
    # pos = i+1
    # plt.subplot(12, 10, pos)
    df_ = df[df.delito == delito]["ao_hechos"].value_counts().sort_index()
    pct_change = float(
        df[df.delito == delito]["ao_hechos"]
        .value_counts()
        .sort_index()
        .rename_axis("years")
        .to_frame(name="counts")[-5:]
        .pct_change()
        .counts.mean()
    )
    df_pct_change.at[delito, "pct_change"] = pct_change
    # df_.plot(title = delito[0:25])

df_pct_change_increasing = df_pct_change[df_pct_change["pct_change"] >= 0]
df_pct_change_decreasing = df_pct_change[df_pct_change["pct_change"] < 0]


# In[473]:


print("Lista de los crimenes en aumento, total:", df_pct_change_increasing.shape[0])
list(df_pct_change_increasing.index)  # whole df


# In[474]:


print("Lista de los crimenes en descenso, total:", df_pct_change_decreasing.shape[0])
list(df_pct_change_decreasing.index)


# #### ¿Cuál es la alcaldía que más delitos tiene y cuál es la que menos? ¿Por qué crees que sea esto?

# En el dataset en analisis existe un total de 514 alcaldias. 

# La alcaldia con el mayor numero de crimenes es:

# In[981]:


print(
    "alcaldia: ",
    df["alcaldia_hechos"].value_counts().idxmax(),
    "No. delitos:",
    df["alcaldia_hechos"].value_counts().max(),
)


# Por otro lado, existe un total de 223 alcaldias con un numero de delitos minimo, es decir 1:

# In[989]:


lower_crime_rate_alcaldias = df["alcaldia_hechos"].value_counts() == 1
lower_crime_rate_alcaldias = list(
    lower_crime_rate_alcaldias[lower_crime_rate_alcaldias == True].index
)
lower_crime_rate_alcaldias


# In[328]:


df_cuauhtemoc = df[df["alcaldia_hechos"] == "CUAUHTEMOC"]


# In[329]:


df_cuauhtemoc = df_cuauhtemoc.dropna()
m = folium.Map([39.645, -79.96], zoom_start=14)
for index, row in df_cuauhtemoc.iterrows():
    folium.CircleMarker(
        [row["latitud"], row["longitud"]],
        radius=3,
        popup=row["delito"],
        fill_color="#3db7e4",
    ).add_to(m)


# In[331]:


m


# In[1032]:


df_cuauhtemoc["delito"].value_counts().sort_values(ascending=False)[0:10].plot(
    kind="barh", title="Delitos Cuauhtemoc"
)


# In[1035]:


df_cuauhtemoc["colonia_hechos"].value_counts().head()


# In[993]:


df_cuauhtemoc["ao_hechos"].value_counts().sort_index().plot(
    kind="bar", title="Total Crime Events by Year"
)


# He realizado un breve de los delitos cometidos en la alcaldia de Cuauhtemoc, habiendo encontrado lo siguiente:
#    
#    - El crimen en la alcaldia empezo a aumentar significativamente en el año 2016.
#    
#    - Los delitos con mayor ocurrencia estan vinculados al robo (vehiculos, objetos, accesorios de vehiculos asi como tambien transeuntes)
#    
#    - Otro crimenes tambien destacados son la narcoposesion y el maltrato intrafamiliar.
#    
#    - La latitud y longitud del dataset muestra una zona con presencia de muchos comercios, la zona parece bastante turistica al estar formada por colonias que contienen sitios de interes para visitantes, pero tambien se observan centros nocturnos, prostibulos, etc.
# 
# Con esta informacion puedo suponer que Cuauhtemoc corresponde a una alcaldia de nivel socieconomico bajo donde hay mas probabilidad delincuencia debido a la presencia de comercios en el sector y de visitantes interesados en la zona, ademas del narcomenudeo que puede ser tambien causante de que delitos que causa que se considere a esta como la alcaldia de mayor delincuencia en el periodo que se analiza. 
# 
# 
# Tras una breve investigacion, he confirmado que Cuauhtemoc es efectivamente una zona de nivel socioeconomico popular/bajo, en donde se encuentran las colonias consideradas mas peligrosas en la CDMX como son Centro, Sta. Maria la Ribera, Morelos, etc. Este sector tiene problematicas como la seguridad pública, recolección de basura, hacinamiento, vivienda, desempleo, insalubridad, drogadicccion, desercion escolar etc. 
# Estos factores indudablemente inciden en la tasa de delitos alta que se puede evidenciar, ademas de justificar los delitos de violencia registrados debido a las condiciones de vida de las personas de esta alcaldia.
# 
# Por otro lado, las alcaldias con menor numero de crimenes corresponden en su mayoria a alcaldias alejadas con un numero reducido de habitantes y menor extension, lo que explica la tasa baja de delitos en estas zonas.
# 
# fuente : https://www.redalyc.org/pdf/325/32515208.pdf
# 

# #### ¿Cuáles son los delitos que más caracterizan a cada alcaldía? Es decir, delitos que suceden con mayor frecuencia en una alcaldía y con menor frecuencia en las demás.

# El analisis realizado para esta seccion, incluye graficas de cada delito Vs. el numero de ocurrencias del mismo para cada alcaldia. 
# Analiticamente se ha determinado tambien el delito con mas frecuencia para cada alcaldia, siendo predominantes las alcaldias de 'Cuauhtemoc' e 'Iztapalapa', para la mayoria de delitos analizados. Este resultado es consistente con el analizado previamente, el mismo que categorizaba a Cuauhtemoc como la alcaldia con mayor indice de delincuencia en el periodo analizado.

# In[28]:


df_crime_frequency = (
    df.groupby("delito")["alcaldia_hechos"].value_counts().reset_index(name="counts")
)
# for i, crime in enumerate(df_crime_frequency.delito.unique()):
#     pos = i + 1
#     d = df_crime_frequency[df_crime_frequency.delito == crime]
#     d.plot(
#         x="alcaldia_hechos", y="counts", figsize=(10, 4), grid=True, title=crime[0:30]
#     )
#     plt.xticks(rotation=90)


# In[325]:


table_crime_frecuency = pd.crosstab(df["delito"], df["alcaldia_hechos"]).T
table_crime_frecuency.T.plot(figsize=(40, 15))


# In[314]:


crime_rate_max = []
for column in table_crime_frecuency.columns:
    max_index = table_crime_frecuency[column].idxmax()
    max_value = table_crime_frecuency[column].max()
    col_name = column[0:30] + " - " + max_index
    crime_rate_max.append([col_name, max_value])


# In[342]:


df_alcaldia_rate = pd.DataFrame(crime_rate_max, columns=["delito", "counts"])
df_alcaldia_rate.sort_values(by=["counts"]).plot(
    x="delito", y="counts", kind="barh", figsize=(6, 20)
)


# #### Diseña un indicador que mida el nivel de “inseguridad”. Genéralo al nivel de desagregación que te parezca más adecuado (ej. manzana, calle, AGEB, etc.). Analiza los resultados ¿Encontraste algún patrón interesante? ¿Qué decisiones se podrían tomar con el indicador?

# In[47]:


def show_distribution(var_data):
    """
    This function will make a distribution (graph) and display it
    """

    # Get statistics
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]
    q25_val = np.percentile(var_data, 25)
    q99_val = np.percentile(var_data, 99)

    print(
        "Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\np25:{:.2f}\np95:{:.2f}\n".format(
            min_val, mean_val, med_val, mod_val, max_val, q25_val, q99_val
        )
    )

    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize=(15, 5))

    # Plot the histogram
    ax[0].hist(var_data, color="navy")
    ax[0].set_ylabel("Frequency")

    # Add lines for the mean, median, and mode
    ax[0].axvline(x=q25_val, color="yellow", linestyle="dashed", linewidth=2)
    ax[0].axvline(x=q99_val, color="yellow", linestyle="dashed", linewidth=2)
    ax[0].axvline(x=mean_val, color="cyan", linestyle="dashed", linewidth=2)
    ax[0].axvline(x=med_val, color="red", linestyle="dashed", linewidth=2)

    # Plot the boxplot
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel("Engagement")

    # Add a title to the Figure
    fig.suptitle("Data Distribution")

    # Show the figure
    fig.show()
    return mean_val, q99_val


def get_normalized_data(df):
    q25 = np.percentile(df.counts, 25)
    q75 = np.percentile(df.counts, 75)
    cut_off = (q75 - q25) * 3  # iqr * k
    lower = q25 - cut_off
    upper = q75 + cut_off
    df_ = df[df.counts > 1]
    df_norm = df_[df_.counts < upper]
    return df_norm


# El indicador a ser usado pretende medir el nivel de inseguridad en la CDMX que considera para su calculo el porcentaje de cambio delincuencial en cada zona. Para esto, se ha planteado un analisis de percentiles, la desagregacion se ha realizado a nivel de alcaldia. Se han planteado una categorizacion delicuencial en 4 niveles:
# 
#     - A: nivel bajo de delincuencia
#     - B: nivel medio de delincuencia
#     - C: nivel medio-alto de delincuencia
#     - D: nivel alto de delincuencia
#     
# Tras eliminar outliers se obtiene la distribucion de la muestra, y se seleccionan los percentiles 50 y 95 para definir los limites de cada rango de frecuencia de delitos. El percentil 50 definira el limite superior para el nivel bajo, el percentil 95 definira el limite superior para el nivel medio, los valores a partir del percentil 95 se considerara como nivel medio-alto de delincuencia hasta el limite superior, que vendra dado tras analizar los valores de outliers encontrados en el dataset, los valores a partir de este limite se consideraran como nivel alto de delincuencia.
# 
# Ademas, para este analisis se toma en consideracion el porcentaje de cambio, es decir el porcentaje de crecimiento o decrecimiento delincuencial en un periodo de tiempo de 5 anios. En caso de que el indice de delincuencia este en decrecimiento, se multiplicara la ocurrencia del delito por un valor k1 < 1. Por otro lado, cuando el indice de delincuencia este en crecimiento se multiplicara la ocurrencia del delito por un valor k2 >1 que viene dado por la suma de 1 + indice de crecimiento.
# 
# Este ultimo threshold, se ha fijado en un valor de 1500, dado que tras analizar los datos existe un evidente aumento en la ocurrencia de delitos a partir de un valor cercano a 1500. Estos son outliers extremos que deberian ser categorizados como nivel alto de delincuencia.
# 
# El indicador pretende mostrar el nivel de inseguridad de CDMX atraves del porcentaje que representan las alcaldias con nivel delincuencial alto y medio alto, del total de alcaldias analizadas.

# In[117]:


df_2021 = df[df['ao_hechos']==2021]


# In[118]:


df_indicador = (
    df_2021.alcaldia_hechos.value_counts().rename_axis("alcaldia").to_frame(name="counts")
)
df_indicador_norm = get_normalized_data(df_indicador)


# In[119]:


mean, q95 = show_distribution(df_indicador_norm.counts)


# In[120]:


df_pct_change_alcaldia = pd.DataFrame()
for i, alcaldia in enumerate(df.alcaldia_hechos.value_counts().index):
    pct_change_alcaldia = float(
        df[df.alcaldia_hechos == alcaldia]["ao_hechos"]
        .value_counts()
        .sort_index()
        .rename_axis("years")
        .to_frame(name="counts")[-5:]
        .pct_change()
        .counts.mean()
    )
    df_pct_change_alcaldia.at[alcaldia, "pct_change"] = pct_change_alcaldia


# In[121]:


df_indicador_ = pd.concat([df_indicador, df_pct_change_alcaldia], axis=1)
df_indicador_["pct_change"] = df_indicador_["pct_change"].fillna(0)
df_indicador_["counts"] = df_indicador_["counts"].fillna(0)
df_indicador_["trend"] = ""
for item, count in enumerate(df_indicador_["pct_change"].values):
    if count >= 0:
        df_indicador_["trend"][item] = 1 + count
    else:
        df_indicador_["trend"][item] = 1 * 0.9


# In[122]:


df_indicador_["crime_rate"] = df_indicador_.counts * df_indicador_.trend


# In[123]:


df_indicador_.head()


# In[124]:


df_indicador_["indicator"] = ""
for i, ind in enumerate(df_indicador_["crime_rate"].values):
    if ind <= mean:
        df_indicador_["indicator"][i] = "A"
    elif (ind > mean) & (ind <= q95):
        df_indicador_["indicator"][i] = "B"
    elif (ind > q95) & (ind <= 1500):
        df_indicador_["indicator"][i] = "C"
    elif ind > 1500:
        df_indicador_["indicator"][i] = "D"


# Tras el analisis realizado, que puede replicarse para colonias, calles, etc., se ha definido la siguiente distribucion:
# 

# In[125]:


print(
    "Nivel bajo de delincuencia:",
    df_indicador_.indicator.value_counts()["A"],
    "alcaldias",
)
print(
    "Nivel medio de delincuencia:",
    df_indicador_.indicator.value_counts()["B"],
    "alcaldias",
)
print(
    "Nivel medio-alto de delincuencia:",
    df_indicador_.indicator.value_counts()["C"],
    "alcaldias",
)
print(
    "Nivel alto de delincuencia:",
    df_indicador_.indicator.value_counts()["D"],
    "alcaldias",
)


# Como se menciono anteriormente el indicador pretende mostrar el nivel de inseguridad de CDMX, y se define por la suma del % que representan las alcaldias con nivel delincuencial medio-alto + la suma del % que representan las alcaldias con nivel delincuencial alto, del total de alcaldias analizadas.
# 
# En este caso, la suma del porcentaje delincuencial de las alcaldias con nivel medio-alto (26) y alto (16) representan el 7.5% de total del alcaldias para el anio 2021. 
# En el año 2020 se categorizaron con nivel delincuencial medio-alto a 38 alcaldias, y con nivel delincuencial alto a 16 alcaldias, lo que representa el 9.67%. Este indicador nos permite demostrar que con respecto al anio 2020, en el anio 2021 ha habido una reduccion de alcaldias con niveles de delincuencia medio-altos y altos, de esta manera y de forma general el nivel de delincuencia en CDMX ha bajado un 2.17% del anio 2020 al 2021.

# In[106]:


# report = pandas_profiling.ProfileReport(df)
# report

