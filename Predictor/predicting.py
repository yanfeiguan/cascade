#This is the script running predictions at backend using redis

import json
import redis
import time,os
import shutil
import tarfile

import pandas as pd
import numpy as np
from rdkit import Chem
from nfp.preprocessing import MolAPreprocessor, GraphSequence

import keras
import keras.backend as K

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from keras.layers import (Input, Embedding, Dense, BatchNormalization,
                                 Concatenate, Multiply, Add)

from keras.models import Model, load_model

from nfp.layers import (MessageLayer, GRUStep, Squeeze, EdgeNetwork,
                               ReduceBondToPro, ReduceBondToAtom,
                               GatherAtomToBond, ReduceAtomToPro)
from nfp.models import GraphModel
from cascade.apply import predict_NMR

redis_client = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)

#Always check for new inpur from views.py if there is any then run the calculation

modelpath = os.path.join('cascade', 'trained_model', 'best_model.hdf5')
batch_size = 32
atom_means = pd.Series(np.array([0,0,97.74193,0,0,0,0,0,0,0]).astype(np.float64), name='shift')
NMR_model = load_model(modelpath, custom_objects={'GraphModel': GraphModel,
                                             'ReduceAtomToPro': ReduceAtomToPro,
                                             'Squeeze': Squeeze,
                                             'GatherAtomToBond': GatherAtomToBond,
                                             'ReduceBondToAtom': ReduceBondToAtom})
NMR_model.summary()

atoms = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F', 15:'P', 16:'S', 17:'Cl'}

def handle_task_id(task_id):
    task_detail_key = "task_detail_{}".format(task_id)
    detail = redis_client.get("task_detail_{}".format(task_id))

    data = json.loads(detail)
    smiles = data.get("smiles", None)

    #predicting NMR
    mols, weightedPrediction, spreadShift = predict_NMR(smiles, NMR_model)

    #writing results into cascade/results folder
    folder = os.path.join("cascade", "results", task_id)
    os.mkdir(folder)
    writer = Chem.SDWriter(os.path.join(folder, "conformers.sdf"))
    for m in mols:
        writer.write(m)
    writer.close()
    weightedPrediction.to_csv(os.path.join(folder, "weighted_shift.csv"))
    spreadShift.to_csv(os.path.join(folder, "conformers_shift.csv"))
    with tarfile.open("{}.tar.gz".format(folder), "w:gz") as tar_handle:
        for root, dirs, files in os.walk(folder):
            for file in files:
                tar_handle.add(os.path.join(root, file), arcname=file)
    shutil.rmtree(folder)
    #Convert results into json
    weightedShiftTxt = ''
    for _,r in weightedPrediction.iterrows():
        weightedShiftTxt += '%s,%s;' % (int(r['atom_index']), r['Shift'])

    jsmol_command = ''
    for m in mols:
        coords = ''
        for i,a in enumerate(m.GetAtoms()):
            ixyz = m.GetConformer().GetAtomPosition(i)
            coords += "{} {} {} {}|".format(atoms[a.GetAtomicNum()], *ixyz)

        jsmol_command += "data \"model example\"|{}|testing|{}end \"model example\";show data!".format(m.GetNumAtoms(), coords)

    confShiftTxt = ''
    relative_E = ''
    group_spreadShift = spreadShift.groupby(['mol_id', 'cf_id'], sort=False)
    B_sum = 0
    for _,df in group_spreadShift:
        B_sum += df.iloc[0]['b_weight']

    for _,df in group_spreadShift:
        for _,r in df.iterrows():
            confShiftTxt += "{},{};".format(int(r['atom_index']), r['predicted'])
        relative_E += "{0},{1:.1f}!".format(df.iloc[0]['relative_E'], df.iloc[0]['b_weight']*100/B_sum)
        confShiftTxt += "!"

    task_result_key = "task_result_{}".format(task_id)

    redis_client.set(task_result_key, json.dumps({
        'jsmol_command': jsmol_command,
        'weightedShiftTxt': weightedShiftTxt,
        'confShiftTxt': confShiftTxt,
        'relative_E': relative_E,
        'smiles': smiles,
    }))

while True:
    result = redis_client.blpop("task_queue", 2)
    if result:
        _,task_id = result
        print("get result: {}".format(task_id))
        #try:
        handle_task_id(task_id)
        #except Exception as e:
        #    print(e)
    else:
        print("No task found, check after 0.3s")
        time.sleep(0.3)
