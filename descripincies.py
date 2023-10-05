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
    from features import ml_features_config
    feature_names = [feature.name for feature in ml_features_config]
    feature_names = feature_names.__str__().replace("[","{").replace("]","}").replace("'","")

    df = ROOT.RDataFrame("features", "features/ttbar_nominal.root")
    df = df.Define("features", f"ROOT::VecOps::RVec<ROOT::RVecF>({feature_names})")
    input_features = df.AsNumpy(["features"])["features"]

    return rdf2np(input_features).transpose(0,2,1)

# %%
def predict_proba(xgb_model, ff_model, features, uplim=1000):

    rdf_scores = np.array(
        [ROOT.inference(
        ROOT.VecOps.RVec[ROOT.RVecD]( features_i ), ff_model
        ) for features_i in features[:uplim].transpose(0,2,1).tolist()]
    )

    xgb_scores = np.array(
        [xgb_model.predict_proba(features_i)[:,1] for features_i in features[:uplim]]
    )

    return rdf_scores, xgb_scores

# %%
input = get_input_features()
scores1, scores2 = predict_proba(xgb_odd, ff_odd, input)
print(
    f"maximum deviation with odd models: {(scores2-scores1).max()}"
)

# %%
scores1, scores2 = predict_proba(xgb_even, ff_even, input)
print(
    f"maximum deviation with even models: {(scores2-scores1).max()}"
)

# %%
input = np.random.rand(input.shape[0], input.shape[1], input.shape[2])
scores1, scores2 = predict_proba(xgb_even, ff_even, input)
print(
    f"max dev on random input: {(scores2-scores1).max()}"
)

# %%



