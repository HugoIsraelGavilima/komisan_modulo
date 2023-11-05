####################################################################
"""
    ARCHIVO: MODEL-UTILS
    AUTOR: DEP- INTELIGENCIA DE LA INFOMACIÓN - ÁREA: PROYECTOS
    FECHA DE CREACIÓN: 11-09-2023
    
    Descripción: el archivo contiene la definicion de clases para 
    los modelos usados en el proyecto Wallace. 

    Outputs: -
"""
####################################################################
import pandas as pd
import numpy as np
import statsmodels.api as sm

from optbinning import ContinuousOptimalBinning, OptimalBinning
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


#------------------------------------------------------------------------------------------
class models_():
    """
        El objeto esta creado para obtener las caracteristicas importantes de nuestros
        modelos de machine learning, como su almacenamiento, lectura de variables, y adicional
    """
    def __init__(self, smoti_ = "oversample"):
        self.cortes = {}
        self.smoti_ = smoti_
        
    def get_var_info(self, datos):
        self.var_cat_split, self.var_cat_no_split, self.var_num_split, self.var_num_no_split_quant, self.var_num_no_split_num, self.var_num_no_split_quant_dict =  type_variables(datos)

    def make_variables(self, base, Y):
        #Grupo de variables categoricas 
        self.cortes = percentil_bining(base, self.var_num_split, Y, "numerical", self.cortes)
        self.cortes = percentil_bining(base, self.var_cat_split, Y, "categorical", self.cortes)
        self.cortes = split_var_cat(base, self.var_cat_no_split, self.cortes)
        self.cortes = percentil_quantile(base, self.var_num_no_split_quant_dict, self.cortes)

    def make_variables_eval(self, base, coef_res):    
        #Variables Categoricas
        for variable in self.var_cat_split + self.var_cat_no_split:
            labels = np.array(self.cortes[variable]['bin_names'])
            bn = self.cortes[variable]["bin"]
            base["bin_" + variable] = base[variable].apply(lambda x: find_label_cat(x, coef_res, labels, bn, variable))


        #Variables Numericas
        for variable in self.var_num_no_split_quant + self.var_num_split:
            bn = np.array(self.cortes[variable]["bin"])
            labels = self.cortes[variable]["bin_names"]
            base["bin_"+ variable] = pd.cut(base[variable], bins= bn, labels = labels)

        names_endog_split = self.var_num_split +  self.var_cat_no_split + self.var_cat_split + self.var_num_no_split_quant
        self.names_endog_split = ["bin_" + name for name in names_endog_split]
        
        
        mdict = {name_var: "Grupo " + name_var for name_var in self.var_num_no_split_num}
        base = base.rename(columns = mdict)
        self.var_num_no_split_num = list(mdict.values())
        
        return base
    
    def resample_(self, base, endog):
        smoti = RandomOverSampler(random_state=42)
        if self.smoti_ == "undersample":
            smoti = RandomUnderSampler(random_state=42)
        
        base, endog = smoti.fit_resample(base, endog)
        return base, endog.astype(int)
#------------------------------------------------------------------------------------------   

#------------------------------------------------------------------------------------------ 
def find_label_cat(x, coef_res, labels, bn, variable):
    lab = labels[[x in arr for arr in bn]]
    if not lab:
        min_label = coef_res.loc[((coef_res["Var"] == variable) & 
                                  (coef_res["coef"].astype(float) == 0)), "var_group"].values[0]
        return min_label   
    else:
        return lab[0]
#------------------------------------------------------------------------------------------ 

#------------------------------------------------------------------------------------------  
def type_variables(variables):  
    var_cat_split = [key for key, item in variables.items() if ((item["type"] == "cat") & (item["dummy_split"]))]
    var_cat_no_split = [key for key, item in variables.items() if ((item["type"] == "cat") & ~(item["dummy_split"]))]
    var_num_split = [key for key, item in variables.items() if ((item["type"] == "num") & (item["dummy_split"]))]
    var_num_no_split = [key for key, item in variables.items() if ((item["type"] == "num") & ~(item["dummy_split"]))]
    
    var_num_no_split_quant = [key for key in var_num_no_split if variables[key]["Num_Split"] != "None"]
    var_num_no_split_num = list(set(var_num_no_split) - set(var_num_no_split_quant))
    var_num_no_split_quant_dict = {key: variables[key]["Num_Split"] for key in var_num_no_split_quant}

    return var_cat_split, var_cat_no_split, var_num_split, var_num_no_split_quant, var_num_no_split_num, var_num_no_split_quant_dict         
#------------------------------------------------------------------------------------------  

#------------------------------------------------------------------------------------------  
def replace_var(base, dict_):
    if dict_ is None:
        return base
    for var, map_replace in dict_.items():
        base["bin_" + var] = base["bin_" + var].replace(map_replace)
    return base
#------------------------------------------------------------------------------------------  

#------------------------------------------------------------------------------------------     
def percentil_bining(base, names, endog, var_type, cortes):
    for variable in names:
        data_astype = np.float64 if var_type == "numerical" else object
        optb = ContinuousOptimalBinning(name=variable, dtype=var_type)
    
        if (endog.dtype == bool):
            optb = OptimalBinning(name=variable, dtype=var_type)
        
        optb.fit(base[variable].astype(data_astype).values, endog.values).status
    
        if var_type == "numerical":
            bn = np.insert(np.insert(optb.splits, len(optb.splits), 1e8), 0, -1e8).tolist()
            labels = ["Grupo " + variable + " N" + str(k + 1) for k in range(len(bn) -1)]
            
        if var_type == "categorical":
            bn = [arr.tolist() for arr in optb.splits]
            labels = ["Grupo " + variable + " N" + str(k + 1) for k in range(len(bn))]
            
        cortes[variable] = {"bin_names": labels, "bin": bn}
          
    return cortes
#------------------------------------------------------------------------------------------  

#------------------------------------------------------------------------------------------  
def split_var_cat(base, names, cortes):
    for var in names:
        bn = [[a] for a in base[var].unique().tolist()]
        labels = ["Grupo " + str(var) + " " + str(names_var_) for names_var_ in base[var].unique().tolist()]
        cortes[var] = {"bin_names": labels, "bin": bn}
    
    return cortes
#------------------------------------------------------------------------------------------  
 
#------------------------------------------------------------------------------------------         
def percentil_quantile(base, dict, cortes):
    for variable, quantiles_cut in dict.items():
        splits = base[variable].quantile(np.linspace(0,1,quantiles_cut + 1)).to_numpy()
        splits = np.unique(splits)
        splits = np.delete(splits, [0, len(splits) - 1])
        bn = np.insert(np.insert(splits, len(splits), 1e8), 0, -1e8)
        labels = ["Grupo_" + str(i) for i in range(len(bn) - 1)]
        cortes[variable] = {"bin_names": labels, "bin": bn}
    
    return cortes
#------------------------------------------------------------------------------------------  