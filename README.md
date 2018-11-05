
visualistaion : QGis

pb=!1m14!1m12!1m3!1d   37012.19549673365!2d  -1.6840613378197213!3d   48.10885402717424!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!5e0!3m2!1sfr!2sfr!4v1540891451682"


nord est lat=48.1440&lon=-1.6140
sud ouest lat=48.0500&lon=-1.7550


[[48.1440,-1.6140],[48.0500,-1.7550]]


db.getCollection('coyote_data').find({loc:{$geoWithin: {$geometry: {type: "Polygon" ,coordinates: [[[-1.6140, 48.1440], [-1.6140, 48.0500], [-1.7550, 48.0500], [-1.7550,48.1440], [-1.6140, 48.1440]]]}}}}).count()


({"loc":{"$geoWithin": {"$geometry": {"type": "Polygon", "coordinates": [[[-1.6140, 48.1440], [-1.6140, 48.0500], [-1.7550, 48.0500], [-1.7550, 48.1440], [-1.6140, 48.1440]]]}}}})

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


db.ways.aggregate([
    {$match: { key: {$in: ["highway"]}}},
    {$out: "roads"}
])

https://wiki.openstreetmap.org/wiki/Key%3Ahighway
{key:"highway", "tag.v": {$nin: ["footway", "path", "pedestrian","track", "bus_guideway", "raceway", "bridleway", "steps", "cycleway", "construction", "proposed"]}}


{"tag": {"$nin":["highway","footway"]}}
