# Projet PRIM : analyse de la congestion routière


## Commandes utiles:

Se connecter à Lame10 :
```
ssh mbourliatoux@ssh.enst.fr
ssh mbourliatoux@lame10
```

Faire un tunnel :
```
en local : ssh -N -L localhost:8888:localhost:8889 mbourliatoux@ssh.enst.fr
sur ssh.enst.fr : ssh -N -L localhost:8889:localhost:8890 mbourliatoux@lame10
```


Lancer un notebook jupyter :
```
.local/bin/jupyter notebook --no-browser --port=8890
```

Faire des screens:
```
dans lame10: screen -S jupyterNotebook

puis lancement du notebook sans navigateur sur le port 8890 :
.local/bin/jupyter notebook --no-browser --port=8890

détacher le "screen" :
ctrl+A+D

pour l'arreter :
retourner sur le screen avec: screen -R jupyterNotebook 
puis ctrl + C pour l'arreter
```
tuto screens : <https://www.linuxtricks.fr/wiki/screen-plusieurs-terminaux-virtuels-dans-un-seul>

## Zone étudiée : Rennes

nord est lat=48.1440&lon=-1.6140
sud ouest lat=48.0500&lon=-1.7550

```
[[48.1440,-1.6140],[48.0500,-1.7550]]
```

Pour compter le nombre de points à Rennes
```
db.getCollection('coyote_data').find({loc:{$geoWithin: {$geometry: {type: "Polygon" ,coordinates: [[[-1.6140, 48.1440], [-1.6140, 48.0500], [-1.7550, 48.0500], [-1.7550,48.1440], [-1.6140, 48.1440]]]}}}}).count()
```

Pour extraire les points de Rennes
```
mongoexport -h <server> -d <dbmane> -c <collection> -q '{"loc":{"$geoWithin": {"$geometry": {"type": "Polygon", "coordinates": [[[-1.6140, 48.1440], [-1.6140, 48.0500], [-1.7550, 48.0500], [-1.7550, 48.1440], [-1.6140, 48.1440]]]}}}}' -out <file>.json
```


Pour créer une mini collection de points de Rennes
```
db.coyote_data_rennes.aggregate([
   {
     $geoNear: {
        near: {'type': 'Point', 'coordinates': [-1.67928, 48.11176]},
        distanceField: "dist.calculated",
        maxDistance: 100,
        key: "loc",
        includeLocs: "dist.location",
        num: 1000,
        spherical: true
     }
   }, { $out: "subset_rennes" }
])
```

## Importer les données OSM dans mongodb

Utiliser le script `insert_osm_data.py`

```
python insert_osm_data.py <input-file>.osm <dbname>
```

Cela va créer 3 collections : 
- nodes
- ways
- relations

Pour obtenir une collection avec seulement les routes sur lesquelles passent les voitures:

```
db.ways.aggregate([
    {
      $match: {
        key:"highway",
        "tag.v": {
          $nin: [
            "footway", 
            "path", 
            "pedestrian",
            "track", 
            "bus_guideway", 
            "raceway", 
            "bridleway", 
            "steps", 
            "cycleway", 
            "construction", 
            "proposed"
          ]
        }
      }
    },
    {
      $out: "roads"
    }
])
```
Source: <https://wiki.openstreetmap.org/wiki/Key%3Ahighway>

{key:"highway", "tag.v": {$nin: ["footway", "path", "pedestrian","track", "bus_guideway", "raceway", "bridleway", "steps", "cycleway", "construction", "proposed"]}}

## Pour visualiser dans QGis:

Installer le plugin `qgis-mongodb-plugin-1.5.0.zip` (dans ce repo).
Le code source est dispo ici : <https://github.com/bourliam/qgis-mongodb-plugin>
C'est un fork adapté à QGIS 3 de ce plugin : <https://github.com/adrianaksan/qgis-mongodb-plugin>








10270362 points modified / 10270362 in 10314.604 seconds  => 2h52

