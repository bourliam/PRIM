"""This program parses an OSM XML file and inserts the data in a
MongoDB database"""

import sys
import os
import time
import pymongo
from datetime import datetime
#from xml.sax import make_parser
#from xml.sax.handler import ContentHandler
from pymongo import MongoClient
from xml.etree.cElementTree import iterparse

class OsmHandler(object):
    """Base class for parsing OSM XML data"""
    def __init__(self, client, db):
        self.client = client
        self.db = client[db]

        #self.db.nodes.ensure_index([('loc', pymongo.GEO2D)])
        self.db.nodes.ensure_index([('id', pymongo.ASCENDING),
                                            ('version', pymongo.DESCENDING)])

        #self.db.ways.ensure_index([('loc', pymongo.GEO2D)])
        self.db.ways.ensure_index([('id', pymongo.ASCENDING),
                                           ('version', pymongo.DESCENDING)])

        self.db.relations.ensure_index([('id', pymongo.ASCENDING),
                                                ('version', pymongo.DESCENDING)])
        self.stat_nodes = 0
        self.stat_ways = 0
        self.stat_relations = 0
        self.lastStatString = ""
        self.statsCount = 0

    def writeStatsToScreen(self):
        for char in self.lastStatString:
            sys.stdout.write('\b')
        self.lastStatString = "%dk nodes, %dk ways, %d relations" % (self.stat_nodes / 1000,
                                                                     self.stat_ways / 1000,
                                                                     self.stat_relations)
        sys.stdout.write(self.lastStatString)

    def fillDefault(self, attrs):
        ts = None
        """Fill in default record values"""
        record = dict(_id=int(attrs['id']),
                      #ts=self.isoToTimestamp(attrs['timestamp']),
                      timestamp=attrs['timestamp'] if 'timestamp' in attrs else None,
                      tag=[],
                      key=[])
        #record['_id'] = int(attrs['id'])
        #record['timestamp'] = self.isoToTimestamp(attrs['timestamp'])
        #record['tags'] = [] 
        #record['keys'] = []
        if 'user' in attrs:
            record['user'] = attrs['user']
        if 'uid' in attrs:
            record['uid'] = int(attrs['uid'])
        if 'version'in attrs:
            record['version'] = int(attrs['version'])
        if 'changeset' in attrs:
            record['changeset'] = int(attrs['changeset'])
        return record

    def isoToTimestamp(self, isotime):
        """Parse a date and return a time tuple"""
        t = datetime.strptime(isotime, "%Y-%m-%dT%H:%M:%SZ")
        return time.mktime(t.timetuple())

    def parse(self, file_obj):
        nodes = []
        ways = []
        
        context = iter(iterparse(file_obj, events=('start', 'end')))
        event, root = context.__next__()
        
        for (event, elem) in context:
            name = elem.tag
            attrs = elem.attrib
            
            if 'start' == event:
                """Parse the XML element at the start"""
                if name == 'node':
                    record = self.fillDefault(attrs)
                    loc = [float(attrs['lon']), float(attrs['lat'])]
                    record['loc'] = {'type':'Point', 'coordinates': loc} 
                elif name == 'tag':
                    k = attrs['k']
                    v = attrs['v']
                    # MongoDB doesn't let us have dots in the key names.
                    #k = k.replace('.', ',,')
                    record['tag'].append(dict(k=k, v=v))
                    record['key'].append(k)
                elif name == 'way':
                    # Insert remaining nodes
                    if len(nodes) > 0:
                        self.db.nodes.insert(nodes)
                        nodes = []

                    record = self.fillDefault(attrs)
                    record['nodes'] = []
                elif name == 'relation':
                    # Insert remaining ways
                    if len(ways) > 0:
                        self.db.ways.insert(ways)
                        ways = []

                    record = self.fillDefault(attrs)
                    record['members'] = []
                elif name == 'nd':
                    ref = int(attrs['ref'])
                    record['nodes'].append(ref)
                elif name == 'member':
                    ref=int(attrs['ref'])
                    record['members'].append(dict(type=attrs['type'],
                                                  ref=ref,
                                                  role=attrs['role']))
                    
                    if attrs['type'] == 'way':
                        ways2relations = self.db.ways.find_one({ '_id' : ref})
                        if ways2relations:
                            if 'relations' not in ways2relations:
                                ways2relations['relations'] = []
                            ways2relations['relations'].append(record['_id'])
                            self.db.ways.save(ways2relations)
                    elif attrs['type'] == 'node':
                        nodes2relations = self.db.nodes.find_one({ '_id' : ref})
                        if nodes2relations:
                            if 'relations' not in nodes2relations:
                                nodes2relations['relations'] = []
                            nodes2relations['relations'].append(record['_id'])
                            self.db.nodes.save(nodes2relations)
            elif 'end' == event:
                """Finish parsing an element
                (only really used with nodes, ways and relations)"""
                if name == 'node':
                    if len(record['tag']) == 0:
                        del record['tag']
                    if len(record['key']) == 0:
                        del record['key']
                    nodes.append(record)
                    if len(nodes) > 2500:
                        self.db.nodes.insert(nodes)
                        nodes = []
                        self.writeStatsToScreen()

                    record = {}
                    self.stat_nodes = self.stat_nodes + 1
                elif name == 'way':
                    if len(record['tag']) == 0:
                        del record['tag']
                    if len(record['key']) == 0:
                        del record['key']
                    nds = dict((rec['_id'], rec) for rec in self.db.nodes.find({ '_id': { '$in': record['nodes'] } }, { 'loc': 1, '_id': 1 }))
                    record['loc'] = dict()
                    locs = []
                    for node in record['nodes']:
                        if node in nds:
                            locs.append(nds[node]['loc']['coordinates'])
                        else:
                            print('node not found: ' + str(node))
                    record['loc'] = {'type': 'LineString', 'coordinates': locs}

                    ways.append(record)
                    if len(ways) > 2000:
                        self.db.ways.insert(ways)
                        ways = []

                    record = {}
                    self.statsCount = self.statsCount + 1
                    if self.statsCount > 1000:
                        self.writeStatsToScreen()
                        self.statsCount = 0
                    self.stat_ways = self.stat_ways + 1
                elif name == 'relation':
                    if len(record['tag']) == 0:
                        del record['tag']
                    if len(record['key']) == 0:
                        del record['key']
                    self.db.relations.save(record)
                    record = {}
                    self.statsCount = self.statsCount + 1
                    if self.statsCount > 10:
                        self.writeStatsToScreen()
                        self.statsCount = 0
                    self.stat_relations = self.stat_relations + 1
            elem.clear()
            root.clear()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: %s <OSM filename> <dbname>" % (sys.argv[0]))
        sys.exit(-1)

    filename = sys.argv[1]

    if not os.path.exists(filename):
        print("Path %s doesn't exist." % (filename))
        sys.exit(-1)

    client = MongoClient()
    db = sys.argv[2]
    #parser = make_parser()
    handler = OsmHandler(client, db)
    #parser.setContentHandler(handler)
    #parser.parse(open(filename))
    handler.parse(open(filename))
