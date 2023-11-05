####################################################################
"""
    ARCHIVO: MODEL-PLOTING
    AUTOR: DEP- INTELIGENCIA DE LA INFOMACIÓN - ÁREA: PROYECTOS
    FECHA DE CREACIÓN: 04-09-2023
    
    Descripción: el archivo contiene laas funciones para graficar 
    los reusltados obtenidos 

    Outputs: -
"""
####################################################################

#Definimos los modulos necesarios
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from pathlib import Path
from os.path import join
import json

utils_ = join(Path(__file__).parents[0].resolve(), "utils_.json")
with open(utils_, "r") as file:
    fonts_ = json.load(file)


#--------------------------------------------------------------------------------------------------------------------
def report_multilabale_prediction(base,
                                  split_x, split_y,
                                  shape_figure, title,
                                  class_condicion = None,
                                  porcentaje = True):
    
    ###############
    # CATEGORIAS DESERTORES ABSOLUTOS
    splits = np.unique(split_x)
    splits = np.delete(splits, [0, len(splits) - 1])
    bn = np.insert(np.insert(splits, len(splits), 1e8), 0, -1e8)
    labels = ["ABSOLUTO P" + str(i + 1) for i in range(len(bn) - 1)]
    base["Probablidad Desercion Absoluta"] = pd.cut(base["Probablidad Desercion Absoluta"], bins= bn, labels = labels)

    ###############
    # CATEGORIAS DESERTORES GENERALES
    splits = np.unique(split_y)
    splits = np.delete(splits, [0, len(splits) - 1])
    bn = np.insert(np.insert(splits, len(splits), 1e8), 0, -1e8)
    labels = ["GENERAL P" + str(i + 1) for i in range(len(bn) - 1)]
    base["Probablidad Desercion General"] = pd.cut(base["Probablidad Desercion General"], bins= bn, labels = labels)
    base["Values"] = 1

    if not not class_condicion:
        con = (base["Desercion Absoluta"] == class_condicion)
        base = base.loc[con, :]

    pivot_table = pd.pivot_table(base, 
                                 index='Probablidad Desercion General', 
                                 columns='Probablidad Desercion Absoluta', 
                                 values='Values', aggfunc='sum')
    cmatrix = pivot_table.to_numpy()
    if porcentaje:
        cmatrix = cmatrix/pivot_table.sum().sum()
    
    # Crear la figura y el subplot con tamaño personalizado
    fig, ax1 = plt.subplots(1,1,figsize=shape_figure, gridspec_kw={'hspace': 0.001})
    #-----------------------CREAMOS LA FIGURA DE LA MATRIZ DE CONFUSION
    ax1.set_title(title, fontdict=fonts_["to_title"])
    im_1 = ax1.imshow(cmatrix, interpolation='nearest', 
                      cmap=plt.cm.Blues)
    ax1.grid(False)
    # Agregar una barra de color
    cbar = ax1.figure.colorbar(im_1, ax = ax1)
    cbar.ax.set_ylabel('Porcentaje', rotation=-90, va="bottom")
    
    # Añadir las etiquetas de los ejes
    ax1.set(xticks=np.arange(cmatrix.shape[1]),
           yticks=np.arange(cmatrix.shape[0]),
           xticklabels=pivot_table.columns.tolist(), 
           yticklabels=pivot_table.index.tolist())
    plt.tick_params(axis='x', labeltop=True, labelbottom=False)
    for i in range(cmatrix.shape[0]):
        for j in range(cmatrix.shape[1]):
            ax1.text(j, i, "{:.2%}".format(cmatrix[i, j]),
                    ha="center", va="center", 
                    color="white" if cmatrix[i, j] > 0.15 else "black")
            
    plt.show()
#--------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------
def mhist(data, n_bins, hist_range, bin_scale, bool_porc, title, bin_format = "{:.2f}",
          labels = ("Valor", "Frecuencia"), shape_figure = (13, 6)):
    fig, ax = plt.subplots(1,1,figsize=shape_figure)
    counts, bins, patches = ax.hist(data, bins=n_bins, range=hist_range, edgecolor='black')

    sum_label = np.sum(counts)
    cmap = plt.get_cmap('jet')
    low = cmap(0.5)
    handles = [Rectangle((0,0),1,1,color=cmap(0.25),ec="k") for i in range(0,n_bins)]
    name_bin_format = "[" + bin_format + ", " +  bin_format + ")"
    bin_ranges = [name_bin_format.format(bins[i]/bin_scale, bins[i+1]/bin_scale) for i in range(len(bins)-1)]
    name_label = "{:.2f}%"
    bin_labels = [f"{bin_ranges[i]}: {name_label.format(100*(counts[i]/sum_label)) if bool_porc else counts[i]}" for i in range(len(counts))]

    ax.legend(handles, bin_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Agregamos etiquetas y título al gráfico
    x_name, y_name = labels
    ax.set_xlabel(x_name, fontdict= fonts_["to_labels"])
    ax.set_ylabel(y_name, fontdict= fonts_["to_labels"])
    ax.set_title(title, fontdict=fonts_["to_title"])

    # Mostramos el gráfico
    plt.show()
#--------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------
def report_multilabale_prediction__(ytrue, ypred, clases, 
                                  shape_figure = (15, 6),
                                  axis_1_title = "MATRIZ DE CONFUSIÓN", 
                                  axis_2_title = "METRICAS POR CLASE"):
    
    cmatrix = confusion_matrix(ytrue.astype(str), ypred.astype(str))
    cmatrix = cmatrix.astype('float')/cmatrix.sum(axis=1)[:, np.newaxis]

    
    # Crear la figura y el subplot con tamaño personalizado
    fig, (ax1 , ax2) = plt.subplots(1,2,figsize=shape_figure, gridspec_kw={'hspace': 0.001})
    #-----------------------CREAMOS LA FIGURA DE LA MATRIZ DE CONFUSION
    ax1.set_title(axis_1_title, fontdict=fonts_["to_title"])
    im_1 = ax1.imshow(cmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Agregar una barra de color
    cbar = ax1.figure.colorbar(im_1, ax = ax1)
    cbar.ax.set_ylabel('Porcentaje', rotation=-90, va="bottom")
    
    # Añadir las etiquetas de los ejes
    ax1.set(xticks=np.arange(cmatrix.shape[1]),
           yticks=np.arange(cmatrix.shape[0]),
           xticklabels=clases, yticklabels=clases,
           ylabel='TRUE CLASS',
           xlabel='PREDICTED CLASS')
    ax1.invert_yaxis()
    
    # Rotar las etiquetas del eje x
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    for i in range(cmatrix.shape[0]):
        for j in range(cmatrix.shape[1]):
            ax1.text(j, i, "{:.2%}".format(cmatrix[i, j]),
                    ha="center", va="center", color="white" if cmatrix[i, j] > 0.5 else "black")
    
    #-----------------------CREAMOS LA FIGURA DE LAS METRICAS

    ax2.set_title(axis_2_title, fontdict=fonts_["to_title"])
    precision, recall, fscore, support = precision_recall_fscore_support(ytrue, ypred)
    report_metrics = np.stack((precision, recall, fscore), axis = 1)
    im_2 = ax2.imshow(report_metrics, interpolation='nearest', cmap=plt.cm.Purples)

    cbar = ax2.figure.colorbar(im_2, ax=ax2)
    cbar.ax.set_ylabel('Porcentaje', rotation=-90, va="bottom")

    # Añadir las etiquetas de los ejes
    ax2.set(xticks=np.arange(report_metrics.shape[1]),
            yticks=np.arange(report_metrics.shape[0]),
            yticklabels=clases, xticklabels=['Precision', 'Recall', 'F1-score'])
    
    ax2.yaxis.set_ticks_position('both')

    # Añadir los valores de porcentaje en cada celda
    for i in range(report_metrics.shape[0]):
        for j in range(report_metrics.shape[1]):
            ax2.text(j, i, "{:.2%}".format(report_metrics[i, j]),
                    ha="center", va="center", color="white" if report_metrics[i, j] > 0.75 else "black")

    plt.show()
#--------------------------------------------------------------------------------------------------------------------   


#-------------------------------------------------------------------------------------------------------------------- 
def m_line_plot(df: pd.DataFrame,
                to_plot = None,
                show_values = False,
                names = None, 
                axis_y = None,
                shape_figure = (15, 6)):
    fig, ax1 = plt.subplots(1,1,figsize=shape_figure, gridspec_kw={'hspace': 0.001})
    
    #Definimoslos ejes a graficar
    x_axis = np.array(df.index.tolist())
    to_plot = df.columns.tolist() if to_plot is None else to_plot
    
    
    #Definimos los parametros del texto   
    
    for column in to_plot:
        try:
            column_ = df[column].to_numpy(np.float32)
        except(ValueError):
            return f"La columna {column} no puede convertirse en un np.float32, revise porfavor"
        except(KeyError):
            return f"La columna {column} no puede existe, revise porfavor"
        ax1.plot(x_axis, column_, label = column)
        
        if show_values:
            for i, valor in enumerate(column_):
                to_print = f'{valor:.1f}%'
                plt.text(x_axis[i], valor, to_print, **fonts_["to_text"])

    
    #Añadimos los objetos al plot para que se vea entendible
    if axis_y is not None:
        inf, sup, num_ticks = axis_y
        plt.ylim(inf, sup)
        plt.yticks(np.arange(inf, sup, step=num_ticks))
        
    if names is not None:
        x_name, y_name, title_ = names
        plt.xlabel(x_name, fontdict= fonts_["to_labels"])
        plt.ylabel(y_name, fontdict= fonts_["to_labels"])
        plt.title(title_, fontdict= fonts_["to_title"])
        
    
    plt.grid(alpha = 0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.legend()
    plt.show()
#--------------------------------------------------------------------------------------------------------------------  