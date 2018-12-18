import json
import pandas as pd
def createIrisPopulationCollection(path,db, name=None):
    """
    Create collection from excel file
    
    path : str
        the path to excel file (must be of the same format as Population en 2015 - IRIS https://www.insee.fr/fr/statistiques/fichier/3627376/base-ic-evol-struct-pop-2015.zip)
    
    db: MongoClient.database
        data base where to create the collection
    
    name : str or None
        the name of the collection, if None the name is infered from the path
    
    """
    if not name :
        name=path.split('/')[-1].split('.')[-2]
    irisPop = db[name]
    records= pd.read_excel(path, encoding="utf8", skiprows=5, header=0).iloc[:,:13].apply(lambda x : json.loads(x.to_json(force_ascii=False)),axis=1).tolist()
    irisPop.insert_many(records)