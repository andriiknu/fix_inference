import ROOT
import os
from dataclasses import dataclass

# histogram bin lower limit to use for each ML input feature
bin_low = [0, 0, 0, 0, 50, 50, 50, 50, 25, 25, 25, 25, 0, 0, 0, 0, -1, -1, -1, -1]

# histogram bin upper limit to use for each ML input feature
bin_high = [6, 6, 6, 6, 300, 300, 550, 550, 300, 300, 300, 300, 1, 1, 1, 1, 1, 1, 1, 1]

# names of each ML input feature (used when creating histograms)
feature_names = [
    "deltar_leptonbtoplep", "deltar_w1w2", "deltar_w1btophad", "deltar_w2btophad",
    "mass_leptonbtoplep",   "mass_w1w2",   "mass_w1w2btophad", "pt_w1w2btophad",
    "pt_w1",                "pt_w2",       "pt_btophad",       "pt_btoplep", 
    "btag_w1",              "btag_w2",     "btag_btophad",     "btag_btoplep", 
    "qgl_w1",               "qgl_w2",      "qgl_btophad",      "qgl_btoplep",
]

# labels for each ML input feature (used for plotting)
feature_labels = [
    "Delta R between b_{top-lep} Jet and Lepton",
    "Delta R between the two W Jets",
    "Delta R between first W Jet and b_{top-had} Jet",
    "Delta R between second W Jet and b_{top-had} Jet",
    "Combined Mass of b_{top-lep} Jet and Lepton [GeV]",
    "Combined Mass of the two W Jets [GeV]",
    "Combined Mass of b_{top-had} Jet and the two W Jets [GeV]",
    "Combined p_T of b_{top-had} Jet and the two W Jets [GeV]",
    "p_T of the first W Jet [GeV]",
    "p_T of the second W Jet [GeV]",
    "p_T of the b_{top-had} Jet [GeV]",
    "p_T of the b_{top-lep} Jet [GeV]",
    "btagCSVV2 of the first W Jet",
    "btagCSVV2 of the second W Jet",
    "btagCSVV2 of the b_{top-had} Jet",
    "btagCSVV2 of the b_{top-lep} Jet",
    "Quark vs Gluon likelihood discriminator of the first W Jet",
    "Quark vs Gluon likelihood discriminator of the second W Jet",
    "Quark vs Gluon likelihood discriminator of the b_{top-had} Jet",
    "Quark vs Gluon likelihood discriminator of the b_{top-lep} Jet",
]

@dataclass
class MLHistoConf:
  name: str
  title: str
  binning: (int, float, float) # nbins, low, high

ml_features_config: list[MLHistoConf] = [
    MLHistoConf(name = feature_names[i], title = feature_labels[i], binning = (25, bin_low[i], bin_high[i])) for i in range(len(feature_names))
]