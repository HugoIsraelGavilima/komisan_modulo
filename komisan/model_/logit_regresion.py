from ..utils.models__ import*
from ..validation.model_eval import logit_eval
import os 
import pickle

    
class Logistic_Regresion(models_):
    """
    El objeto regresion logistica esta diseñado para crear un modelo de machine learning, 
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
    
    def fit(self, base, name_ob, print_ = False):
        
        #Preparamos los datos pre-entreanmiento
        exog = base[name_ob].astype(bool)
        super().make_variables(base.copy(), exog)
        
        #Clusterizamos las variables
        base = self.make_variables_eval(base.copy(), coef_res = None) 
        base = replace_var(base, self.var_map)
        
        #Binarizamos las variables
        base = self.prepare_base(base.copy())
        base, exog = super().resample_(base, exog)
        self.fit_bin(base)
        base = self.get_bin(base)
        
        #Empezamos con el modelo
        x_train, x_test, y_train, y_test = train_test_split(base, exog, test_size=0.20)
        endog_train = y_train
        exog_train = sm.add_constant(x_train)
        exog_test = sm.add_constant(x_test)
        
        #Almaenamos los modelos
        self.mod = sm.Logit(endog_train, exog_train)
        self.result = self.mod.fit(full_output = True, disp = True)
        
        #Guardamos los resultados
        self.result_train = {
            "train": {
                "obs": endog_train,
                "predict": self.result.predict(exog_train)
            },
            "test": {
                "obs": y_test,
                "predict": self.result.predict(exog_test)
            }
        }
        self.coef_names = coef_names(self.result, self.sort_bin_var, self.var_num_no_split_num)
        
        if print_:
            logit_eval(self.result_train)
        
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
        base_result["prediction_logit"] = self.result.predict(np.c_[np.ones(base.shape[0]), base])
        base_result["variables_relevantes_logit"] = base_relevant.apply(lambda x: relevant(x, self.coef_names), axis=1)
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
    ############################################################################
        
        
    
    
#-----------------------------------------------------------------------------------------------------------
def coef_names(mod, sort_bin_var, var_num):
    names = ["Constant"]
    drop_names = []
    for var_group in sort_bin_var.categories_:
        if var_group.shape[0] > 1:
            names.extend(var_group[1:])
            drop_names.append(var_group[0])
        else:
            drop_names.extend(var_group)
            
    names = names + var_num
    coef_names = pd.DataFrame({"var_group": names, 
                               "coef": mod.params.to_numpy(),
                               "p_values": mod.pvalues.to_numpy()})
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

def relevant(series, coef_names):
    info = series.to_numpy()
    con = coef_names["var_group"].isin(info)
    rel = coef_names.loc[con, ["coef", "Var", "coef_var"]].sort_values('coef', ascending=False).drop(["coef"], axis=1).drop_duplicates()
    res = rel.iloc[0:3, :].agg(lambda x: x.values.tolist(), axis=1).tolist()
    return res

