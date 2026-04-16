import pinocchio as pin

import os
file_path = os.path.abspath(".")

mesh_path = file_path + '/model/bifrank_robot.urdf'
urdf = os.path.join(file_path, "model", "bifrank_robot.urdf")

model = pin.buildModelFromUrdf(urdf)
data = model.createData()

print("DoF:", model.nq, model.nv)

# geometry model
geom_model = pin.buildGeomFromUrdf(model, urdf, pin.GeometryType.COLLISION, package_dirs=[file_path]) # or pin.GeometryType.COLLISION ??
geom_model.addAllCollisionPairs()
geom_data = geom_model.createData()

# remove adjacent collision pairs
srdf_path = os.path.join(file_path, "model", "bifrank_robot.srdf")
pin.removeCollisionPairs(model, geom_model, srdf_path)
geom_data = geom_model.createData()

file_out = open("attributes.txt", "w")
file_out.write("Attributes of geom_model:\n")
for attr in dir(geom_model):
    file_out.write(f"  {attr}\n")
file_out.write("\nAttributes of geom_model.collisionPairs:\n")
for attr in dir(geom_model.collisionPairs):
    file_out.write(f"  {attr}\n")
file_out.write("\nDocumentation of collisionPairs[i]:\n")
file_out.write(f"{geom_model.collisionPairs[0].__doc__}\n")
file_out.write("\nAttributes of geom_model.collisionPairs[i]:\n")
for attr in dir(geom_model.collisionPairs[0]):
    file_out.write(f"  {attr}\n")
file_out.write("\nAttributes of geom_model.findCollisionPair:\n")
for attr in dir(geom_model.findCollisionPair):
    file_out.write(f"  {attr}\n")
file_out.write("\nAttributes of geom_model.geometryObjects[i]:\n")
for attr in dir(geom_model.geometryObjects[0]):
    file_out.write(f"  {attr}\n")
file_out.write("\nDocumentation of geom_model.geometryObjects[i]: \n")
file_out.write(f"{geom_model.geometryObjects[0].__doc__}\n")
file_out.write("\nAttributes of geom_data:\n")
for attr in dir(geom_data):
    file_out.write(f"  {attr}\n")
# file_out.write("\nAttributes of geom_data.collisionResults:\n")
# for attr in dir(geom_data.collisionResults):
#     file_out.write(f"  {attr}\n")
file_out.write("\nAttributes of geom_data.collisionResults[i]:\n")
for attr in dir(geom_data.collisionResults[0]):
    file_out.write(f"  {attr}\n")
file_out.write("\nAttributes of geom_data.distanceResults[i]:\n")
for attr in dir(geom_data.distanceResults[0]):
    file_out.write(f"  {attr}\n")
# file_out.write("\nDocumentation of geom_data.distanceResults[i].normal: \n")
# file_out.write(f"{geom_data.distanceResults[0].normal.__doc__}\n")
# file_out.write("\nAttributes of geom_data.oMg:\n")
# for attr in dir(geom_data.oMg):
#     file_out.write(f"  {attr}\n")
file_out.write("\nAttributes of geom_data.oMg[i]:\n")
for attr in dir(geom_data.oMg[0]):
    file_out.write(f"  {attr}\n")
file_out.write("\nPinocchio: \n")
for attr in dir(pin):
    file_out.write(f"  {attr}\n")
file_out.write("\nDocumentation of pin.getJointJacobian: \n")
file_out.write(f"{pin.getJointJacobian.__doc__}\n")
file_out.write("\nDocumentation of pin.computeDistances: \n")
file_out.write(f"{pin.computeDistances.__doc__}\n")
file_out.close()
