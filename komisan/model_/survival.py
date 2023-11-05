from ..utils.models__ import*
from ..validation.model_eval import logit_eval
from lifelines import CoxPHFitter
import os 
import pickle


class survival(models_):
    """
    El objeto survival esta diseñado para crear un modelo de machine learning, 
    asi como almancenar propiamente sus resultados
        
    Definimos las variables de entrada del modelo
        - name: 
    """
    def __init__(self, name):
        self.name = name
        super().__init__()
        
    def definition(self, var_map, var_def):
        self.var_def = var_def
        self.var_map = var_map
        super().get_var_info(var_def)
        
    def fit(self, base, 
            name_event, name_duration, 
            name_entry_col = None,
            print_ = False):
        
        #Preparamos los datos pre-entreanmiento
        duration_col = base[name_duration].astype(int)
        event_col = base[name_event].astype(bool)
        if name_entry_col is not None:
            entry_col = base[name_entry_col].astype(int) 
        
        super().make_variables(base.copy(), duration_col)
        
        #Clusterizamos las variables
        base = self.make_variables_eval(base.copy(), coef_res = None) 
        base = replace_var(base, self.var_map)
        
        #Binarizamos las variables
        base = self.prepare_base(base.copy())
        self.fit_bin(base)
        base = self.get_bin(base)
        
        #Unimos el dataframe para el modelo
        names, drop_names = self.get_names()
        base = pd.DataFrame(base, columns= names + self.var_num_no_split_num)
        base[name_event] = event_col.values
        base[name_duration] = duration_col.values
        
        if name_entry_col is not None:
            base[name_entry_col] = entry_col.values
            base = base[base[[name_duration, name_entry_col]].apply(lambda x: x[1] < x[0], axis=1)]
            
        #Empezamos con el modelo
        self.result = CoxPHFitter()
        self.result.fit(base, duration_col = name_duration, event_col = name_event, entry_col = name_entry_col)
        
        #Guardamos los resultados
        self.result_train = {
            "train": {
                "base": base
            }
        }
        
        self.coef_names = coef_names(self.result, drop_names)
        if print_:
            None
    def predict(self, base):
        
        base_result = base.copy()
        #Clusterizamos las variables
        base = self.make_variables_eval(base.copy(), coef_res = self.coef_names) 
        base = replace_var(base, self.var_map)
        
        #Preparamos losdatos
        base = self.prepare_base(base.copy())
        base_relevant = base.copy()
        
        base = self.get_bin(base)
        
        #Hacemos la prediccion
        p_ = 0.90
        predict = pd.DataFrame(self.result.predict_expectation(base)).rename(columns={0: "predict"})
        median_ = predict["predict"].quantile(q = p_) 
        predict.loc[predict["predict"] >= 1e20, "predict"] = median_
        base_result["prediccion_survival"] = predict["predict"].astype(int).values
        base_result["variables_relevantes_survival"] = base_relevant.apply(lambda x: relevant(x, self.coef_names), axis=1)
        
        return base_result
    
    
    def fit_bin(self, base):
        self.sort_bin_var = OneHotEncoder(dtype = int, drop = 'first').fit(base[self.names_endog_split])
    
    def get_bin(self, base):
        base_endog_cat_bin = self.sort_bin_var.transform(base[self.names_endog_split]).toarray()
        base_endog_num_bin = base[self.var_num_no_split_num].values 
        return np.concatenate((base_endog_cat_bin, base_endog_num_bin), axis = 1)
         
    def prepare_base(self, base):
        #Make base
        names_endog = self.names_endog_split + self.var_num_no_split_num
        base = base[names_endog]
        
        base[self.names_endog_split] = base[self.names_endog_split].astype(object)
        base[self.var_num_no_split_num] = base[self.var_num_no_split_num].astype(float)
        return base
    
    def get_names(self):
        names = []
        drop_names = []
        for var_group in self.sort_bin_var.categories_:
            if var_group.shape[0] > 1:
                names.extend(var_group[1:])
                drop_names.append(var_group[0])
            else:
                drop_names.extend(var_group)
                
        return names, drop_names
    
    ############################################################################
    # Definiciones para su construccion
    def save(self, 
             to_save_in = False,
             path_folder = None):
        res = {
            "var_def": self.var_def,
            "var_map": self.var_map,
            "cortes": self.cortes,
            "bin_var": self.sort_bin_var,
            "model": self.result,
            "coef_names": self.coef_names,
            "result_train": self.result_train
        }
        
        if to_save_in:
            path = os.path.join(path_folder, f"{self.name}.pkl")
            
            if not os.path.exists(path_folder):
                os.mkdir(path_folder)
            
            with open(path, 'wb') as f:
                pickle.dump(res, f)
        return res
    
    def load(self, name):
        
        if type(name) is dict:
            try:
                self.var_def = name["var_def"]
                self.var_map = name["var_map"]
                self.cortes = name["cortes"]
                self.sort_bin_var = name["bin_var"]
                self.result = name["model"]
                self.coef_names = name["coef_names"]
                self.result_train = name["result_train"]
                
            except:
                return "El diccionario no esta estructurado correctamente"
        
        elif type(name) is str:
            try:
                with open(name, 'rb') as f:
                    res = pickle.load(f)
                
                self.var_def = res["var_def"]
                self.var_map = res["var_map"]
                self.cortes = res["cortes"]
                self.sort_bin_var = res["bin_var"]
                self.result = res["model"]
                self.coef_names = res["coef_names"]
                self.result_train = res["result_train"]
                
            except:
                return f"{name} no es un direccion correcta"
            
        #print("Carga Exitosa!!")  
        super().get_var_info(self.var_def)
        
    
#------------------------------------------------------------------------------------------------------------  
def coef_names(mod, drop_names):
       
    coef_names = mod.summary[["coef", "p"]].reset_index().rename(columns={"covariate": "var_group",
                                                                          "p": "p_values"})
    coef_drop_names = pd.DataFrame({"var_group": drop_names, 
                                    "coef": np.zeros(len(drop_names)),
                                    "p_values": np.zeros(len(drop_names))})

    coef_names = pd.concat([coef_names, coef_drop_names])
    coef_names["p_values"] = coef_names["p_values"].apply(lambda x: "{:.2f}".format(x))
    coef_names["Var"] = coef_names["var_group"].apply(lambda x: x.split()[1] if "Grupo" in x else "Constant")
    coef_names["coef_norm"] = coef_names["coef"].apply(lambda x: x - min(coef_names["coef"]))
    coef_names["coef"] = coef_names["coef"].apply(lambda x: "{:.2f}".format(x))
    coef_names["coef_var_group"] = (coef_names["coef_norm"]/np.sum(coef_names["coef_norm"].values))
    df = coef_names.groupby(by = "Var")["coef_var_group"].apply(np.sum).reset_index().rename(columns={"coef_var_group": "coef_var"})
    coef_names = pd.merge(coef_names, df,how="inner", on="Var")
    coef_names["coef_var_group"] = coef_names["coef_var_group"].apply(lambda x: "{:.2f}".format(x*100) + "%")
    coef_names["coef_var"] = coef_names["coef_var"].apply(lambda x: "{:.2f}".format(x*100) + "%")
    coef_names["coef_norm"] = coef_names["coef_norm"].apply(lambda x: "{:.2f}".format(x))
    
    return coef_names 
#------------------------------------------------------------------------------------------------------------  

#------------------------------------------------------------------------------------------------------------  
def relevant(series, coef_names):
    info = series.to_numpy()
    con = coef_names["var_group"].isin(info)
    rel = coef_names.loc[con, ["coef", "Var", "coef_var"]].sort_values('coef', ascending=False).drop(["coef"], axis=1).drop_duplicates()
    res = rel.iloc[0:3, :].agg(lambda x: x.values.tolist(), axis=1).tolist()
    return res
#------------------------------------------------------------------------------------------------------------  