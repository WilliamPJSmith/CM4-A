import vtk
from vtk import *
from scipy.spatial import ConvexHull
import numpy
import math
import copy


class Capsule:

	def __init__(self, n):
	
		self.res = n 
		self.SetCanonicalPoints()
		self.SetCanonicalTri()

		
	def SetCanonicalPoints(self):
	
		n = self.res
		pi = numpy.pi
		thetas = numpy.linspace(-pi,+pi,n)
		phisL = numpy.linspace(-0.5*pi,0.0,int(math.floor(0.5*n)+1))
		phisU = numpy.linspace(0.0,+0.5*pi,int(math.floor(0.5*n)+1))
		phis = numpy.concatenate((phisL,phisU),axis=0)
		cosphis = numpy.cos(phis); cosphis[0] = 0.0; cosphis[-1] = 0.0
		sinthetas = numpy.sin(thetas); sinthetas[0] = 0.0; sinthetas[-1] = 0.0
		x = numpy.outer(cosphis,numpy.cos(thetas));
		y = numpy.outer(cosphis,sinthetas);
		z = numpy.outer(numpy.sin(phis),numpy.ones([n,],numpy.float))
		self.x = x
		self.y = y
		self.z = z
	
	
	def SetCanonicalTri(self):
	
		zz = copy.copy(self.z)
		m = numpy.floor(0.5*self.res)+1
		zz[0:m,:] -= 1.0; zz[m:,:] += 1.0
		ccloud = numpy.vstack((self.x.flatten('F'), \
							   self.y.flatten('F'), \
							   zz.flatten('F'))).T
		hull = ConvexHull(ccloud)
		self.tris = hull.simplices
	
	
	def Transform(self,l,R,p,a):
	
		n = self.res

		# copy canonical capsule and scale it to correct radius
		xx = R*copy.copy(self.x); yy = R*copy.copy(self.y); zz = R*copy.copy(self.z)
		
		# split the caps
		m = numpy.floor(0.5*n)+1
		zz[0:m,:] -= 0.5*l; zz[m:,:] += 0.5*l
		
		# flatten the 3 arrays into a single N x 3 matrix
		X = xx.flatten('F'); Y = yy.flatten('F'); Z = zz.flatten('F')	
		A = numpy.vstack((X,Y,Z)).T
		
		# compute rotation quaternion and apply
		k_hat = numpy.array([0,0,1]); axis = numpy.array(a)
		q = self.GetRotationQuaternion(k_hat, axis)
		Tcloud = self.QuatRotate(q, A)
		
		# translate point cloud by p
		Tcloud[:,0] += p[0]; Tcloud[:,1] += p[1]; Tcloud[:,2] += p[2]
							  
		return Tcloud
		
		
	def GetRotationQuaternion(self, u1, u2):
	
		u1n = u1 / numpy.linalg.norm(u1)
		u2n = u2 / numpy.linalg.norm(u2)
		q = numpy.zeros((4,));
		q[0] = 1 + numpy.dot(u2n, u1n);          
		q[1:] = numpy.cross(u2n, u1n); 
		
		return q / numpy.linalg.norm(q)  
	

	def QuatRotate(self, q, r):
	
		qin = q / numpy.linalg.norm(q)
		dcm = numpy.zeros((3,3));
		dcm[0][0] = numpy.power(qin[0],2) + numpy.power(qin[1],2) - numpy.power(qin[2],2) - numpy.power(qin[3],2)
		dcm[0][1] = 2*(numpy.multiply(qin[1], qin[2]) + numpy.multiply(qin[0], qin[3]))
		dcm[0][2] = 2*(numpy.multiply(qin[1], qin[3]) - numpy.multiply(qin[0], qin[2]))
		dcm[1][0] = 2*(numpy.multiply(qin[1], qin[2]) - numpy.multiply(qin[0], qin[3]))
		dcm[1][1] = numpy.power(qin[0],2) - numpy.power(qin[1],2) + numpy.power(qin[2],2) - numpy.power(qin[3],2)
		dcm[1][2] = 2*(numpy.multiply(qin[2], qin[3]) + numpy.multiply(qin[0], qin[1]))
		dcm[2][0] = 2*(numpy.multiply(qin[1], qin[3]) + numpy.multiply(qin[0], qin[2]))
		dcm[2][1] = 2*(numpy.multiply(qin[2], qin[3]) - numpy.multiply(qin[0], qin[1]))
		dcm[2][2] = numpy.power(qin[0],2) - numpy.power(qin[1],2) - numpy.power(qin[2],2) + numpy.power(qin[3],2)
		
		return numpy.dot(dcm,r.T).T
		
		
class CapsuleWriter:


	def __init__(self, n, file):
		self.cap = Capsule(n)
		self.FileName = file
		self.init_data()
		
		
	def init_data(self):
	
		# Setup points, vertices and any additional cell data
		self.Points = vtk.vtkPoints()
		self.Triangles = vtk.vtkCellArray()
		self.CellData = vtk.vtkUnsignedCharArray()
		self.CellData.SetNumberOfComponents(3);
		self.CellData.SetName("CellData");
		self.gap = 0
		
		
	def writeConfiguration(self, cellStates):
	
		# loop through cells and transform canonical capsule for each
		for cs in cellStates.itervalues():
			cloud = self.cap.Transform(cs.length, cs.radius, cs.pos, cs.dir)
			self.writeCapsule(cloud, cs.growthRate, cs.cellType, cs.deathFlag)

		# Create a polydata object
		trianglePolyData = vtk.vtkPolyData()

		# Add the geometry and topology to the polydata
		trianglePolyData.SetPoints(self.Points)
		trianglePolyData.GetPointData().SetScalars(self.CellData)
		trianglePolyData.SetPolys(self.Triangles)

		# Clean the polydata so that the edges are shared. Is this needed?
		#cleanPolyData = vtk.vtkCleanPolyData()
		#cleanPolyData.SetInput(trianglePolyData)
		
		trianglePolyData.Modified()

		if vtk.VTK_MAJOR_VERSION <= 5:
			trianglePolyData.Update()
	
		writer = vtk.vtkXMLPolyDataWriter();
		writer.SetFileName(self.FileName);
		if vtk.VTK_MAJOR_VERSION <= 5:
			writer.SetInput(trianglePolyData)
		else:
			writer.SetInputData(polydata)
		writer.Write()
		
		
	def writeCapsule(self,cloud,type,growthRate,deathFlag):
	
		gap = self.gap
	
		for point in cloud:
			self.Points.InsertNextPoint(point[0],point[1],point[2]) 
			self.CellData.InsertNextTupleValue([type,growthRate,deathFlag]); 
			
		for simplex in self.cap.tris:
			tri = vtk.vtkTriangle();
			tri.GetPointIds().SetId(0, simplex[0]+gap);
			tri.GetPointIds().SetId(1, simplex[1]+gap);
			tri.GetPointIds().SetId(2, simplex[2]+gap);
			self.Triangles.InsertNextCell(tri);
		
		self.gap += len(cloud)
		