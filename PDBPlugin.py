from dataset import PISToN_dataset
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
def get_processed(ppi_list, GRID_DIR):
    """
    Output processed protein complexes.
    For a small number of the models, interaction maps failed to preprocess, possibly due to clashes.
    The PIsToN scores for those models will be assigned to "incorrect"
    """
    processed_ppis = []
    for ppi in ppi_list:
        pid, ch1, ch2 = ppi.split('_')
        if os.path.exists(GRID_DIR + '/' + ppi + '.npy'):
            processed_ppis.append(ppi)
    return processed_ppis


import PyPluMA
import PyIO
import pickle
class PDBPlugin:
 def input(self, inputfile):
        self.parameters = PyIO.readParameters(inputfile)
 def run(self):
        pass
 def output(self, outputfile):
  PDB_DIR=PyPluMA.prefix()+"/"+self.parameters["PDB_DIR"]
  GRID_DIR=PyPluMA.prefix()+"/"+self.parameters["GRID_DIR"]
  out_scores=PyPluMA.prefix()+"/"+self.parameters["out_scores"]
  complexes_2023 = [x.strip('\n') for x in open(PyPluMA.prefix()+"/"+self.parameters["complexes"]).readlines()]

  # construct a list of docking models
  all_ppi = []
  with open(PyPluMA.prefix()+"/"+self.parameters["dockingmodels"], 'w') as out:
    for ppi in complexes_2023:
        pid, ch1, ch2 = ppi.split('_')
        for i in range(100):
            all_ppi.append(f'{pid}-model-{i+1}_A_Z')
            out.write(f'{pid}-model-{i+1}_A_Z\n')


  test_list_updated = get_processed(all_ppi, GRID_DIR)

  print(f"{len(test_list_updated)}/{len(all_ppi)} complexes were processed.")
  unprocessed_complexes = set(all_ppi) - set(test_list_updated)

  print(f"Unprocessed complexes: {unprocessed_complexes}")

  unique_pids = list(set([x.split('-')[0] for x in all_ppi]))

  print(f"Unique targets: {unique_pids}")

  test_dataset = PISToN_dataset(GRID_DIR, test_list_updated)

  outComplex = open(outputfile+".complexes.pkl", "wb")
  pickle.dump(complexes_2023, outComplex)
  test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=False)
  modelfile = open(PyPluMA.prefix()+"/"+self.parameters["modelfile"], 'rb')
  model = pickle.load(modelfile)
  from tqdm import tqdm
  #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  device=torch.device("cpu")

  # Infer in batches
  all_outputs = []
  with torch.no_grad():
    for grid, all_energies in tqdm(test_loader):
        grid = grid.to(device)
        all_energies = all_energies.float().to(device)
        model = model.to(device)
        output, attn = model(grid, all_energies)
        all_outputs.append(output)
  output = torch.cat(all_outputs, axis=0)
  torchfile = open(outputfile+".torches.pkl", "wb")
  pickle.dump(output, torchfile)


