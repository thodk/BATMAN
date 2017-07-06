# -*- coding: utf-8 -*-

import pymongo


db_name = "Prokaryotes"

client = pymongo.MongoClient()
db = client[db_name]
collection = db["ec_numbers"]

ecs = []
res = collection.find()
for i in res:
    ecs.append(i["ec_number"])


ecs = sorted(list(set(ecs)))

print len(ecs)
F = open("ec_numbers_"+db_name, 'w')
F.write("\n".join(ecs))
F.close()