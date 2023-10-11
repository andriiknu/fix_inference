# %%
import ROOT
import os
include = os.path.join("./fastforest", 'include')
lib = os.path.join("./fastforest", 'lib')
ROOT.gSystem.AddIncludePath(f"-I{include}")
ROOT.gSystem.AddLinkedLibs(f"-L{lib} -lfastforest")
ROOT.gSystem.Load(f"{lib}/libfastforest.so.1")
ROOT.gSystem.CompileMacro("ml_helpers.cpp", "kO")

ff_models = ROOT.get_fastforests("models/", 20)
ff_even = ff_models['even']
ff_odd = ff_models['odd']
from xgboost import XGBClassifier
xgb_even = XGBClassifier()
xgb_even.load_model('models/model_even.json')
xgb_odd = XGBClassifier()
xgb_odd.load_model('models/model_odd.json')

import numpy as np
from features import ml_features_config
# %%
def rdf2np(arrays, dimensions=3):
    if dimensions == 2:
        result = np.empty((len(arrays), len(arrays[0])))
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                result[x,y] = arrays[x][y]
    elif dimensions == 3:
        result = np.empty((len(arrays), len(arrays[0]), len(arrays[0][0])))
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                for z in range(result.shape[2]):
                    result[x, y, z] = arrays[x][y][z]
    else: raise NotImplementedError

    return result

def get_input_features():
    
    feature_names = [feature.name for feature in ml_features_config]
    feature_names = feature_names.__str__().replace("[","{").replace("]","}").replace("'","")

    df = ROOT.RDataFrame("features", "features/ttbar_nominal.root")
    df = df.Define("features", f"ROOT::VecOps::RVec<ROOT::RVecF>({feature_names})")
    input_features = df.AsNumpy(["features"])["features"]

    return rdf2np(input_features).transpose(0,2,1)

# %%
def rdf_predict_proba(ff_model, features):
    inputs = features.transpose(0,2,1).tolist()
    rdf_scores = [ 
        ROOT.inference(
                ROOT.VecOps.RVec[ROOT.RVecD](input), 
                ff_model,
        ) for input in inputs 
    ]
    return rdf2np(rdf_scores, dimensions=2)

def xgb_predict_proba(xgb_model, features):
    return np.array(
        [xgb_model.predict_proba(features_i)[:,1] for features_i in features]
    )

def predict_proba(xgb_model, ff_model, features):
    rdf_scores = rdf_predict_proba(ff_model, features)
    xgb_scores = xgb_predict_proba(xgb_model, features)
    return rdf_scores, xgb_scores

# %%
input_features = get_input_features()
scores_rdf, scores_xgb = predict_proba(xgb_odd, ff_odd, input_features)
print(
    f"maximum probability scores deviation with odd models: {np.abs(scores_rdf-scores_xgb).max()}"
)
# %%
def apply_argmax(proba_scores, features):
    res = np.empty((features.shape[0],features.shape[-1]))
    for event in range(proba_scores.shape[0]):
        res[event,:] = features[ event, proba_scores[event].argmax() , : ]
    return res

features_rdf = apply_argmax(scores_rdf, input_features)
features_xgb = apply_argmax(scores_xgb, input_features)
print(
    f"maximum features deviation with odd models: {np.abs(features_rdf-features_xgb).max()}"
)