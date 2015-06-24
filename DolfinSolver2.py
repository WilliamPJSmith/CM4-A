"""
DolfinSolver2.py
==================
Replaces DolfinPDESolver class. Designed to use scaled variables (see separate document).
All extraneous functions and tests have been removed for now. I created this with the specific function of creating 

W. P. J. S. 11.05.15
"""

try:
    from dolfin import *
except ImportError:
    print "Error: could not import dolfin library."
    print "Try calling $ source /Applications/FEniCS.app/Contents/Resources/share/fenics/TestFenicsPath.conf "
import numpy
import math
from pyopencl.array import vec
 
 

class DolfinSolver2:

# =======================================================================
# ============================== SETUP ==================================
# =======================================================================

	def __init__(self, solverParams):
		""" 
		Initialise the dolfin solver using a dictionary of params.
		"""
		# domain paramters
		self.h = solverParams['h']
		self.delta = solverParams['delta']
		self.origin = solverParams['origin']
		self.N_x = int(solverParams['N_x'])
		self.N_y = int(solverParams['N_y'])
		self.L_x = solverParams['L_x']
		self.L_y = solverParams['L_y']
		
		# kinetic parameter clusters
		self.Da = solverParams['Da']
		self.Eta = solverParams['Eta']

		# some parameters we will have to calculate on the fly: set them to 0 for now
		self.N_z = 0		# number of *canonical* elements in z
		self.Lz_b = 0.0 	# height at which to apply bulk boundary condition
		self.W = 0.0		# thickness of mesh buffer layer
		
		# some attributes that we'll update on the fly: set them to None for now
		self.boundaryCondition = None
		self.mesh = None
		self.V = None
		self.solution = None
	
	
# =======================================================================
# ============================== SOLVERS ================================
# =======================================================================
	
	def SolvePDE(self, centers, vols, filename, dir, stepNum=0, printEvery=10):
		"""
		High-level function to be called during the function.
		"""

		# get height of highest cell in domain
		max_height = 0.0
		for center in centers:
			hz = center[2] 
			if hz > max_height:
				max_height = hz
		print 'max height is %f' % max_height
		
		# update mesh, function space and BCs
		self.mesh = self.DoubleBufferedMesh(max_height)
		self.V = FunctionSpace(self.mesh, "CG", 1)
		self.set_scaled_bcs()
		
		# Use cell centres to evaluate volume occupancy of mesh
		self.GetVolFracs(centers, vols)
		
		# call a solver and save the solution
		self.NewtonIterator()
		if stepNum % printEvery == 0:
			self.WriteFieldToFile(dir+filename+'.pvd', self.solution)

		# interpolate solution to cell centres
		u_local = self.InterpolateToCenters(centers)
	
		return u_local
		
		
	def NewtonIterator(self):
		"""
		A Newton iterator for solving non-linear problems.
		/!\ Assumes that function space (V), boundaryCondition, vol_fracs are up-to-date.
		"""
		# Define variational problem
		u = Function(self.V, name = "Nutrient")
		v = TestFunction(self.V)
		F = dot(grad(u), grad(v))*dx - self.NDMonodNutrientSink(u)*v*dx

		# Call built-in Newton solver
		set_log_level(WARNING)
		solve(F == 0, u, self.boundaryCondition, solver_parameters = {"newton_solver":
																	  {"relative_tolerance": 1e-6}})															 
		self.solution = u													 		


	def set_scaled_bcs(self):
		"""
		Initialise scaled boundary conditions on the mesh.
		/!\ Assumes that global variable Lz_b is up-to-date.
		"""
		dbc = TopDirichletBoundary()
		self.boundaryCondition = DirichletBC(self.V, Constant(1.0), dbc)
		

# =======================================================================
# ========================== REACTION TERM ==============================
# =======================================================================

	def NDMonodNutrientSink(self, u):
		""" 
		Scaled Monod function with which to build RHS.
		"""
		a = Constant(self.Da)
		b = Constant(self.Eta)
		return -1 * a * u * VolumeFraction() / (b + u)
		
		
# =======================================================================
# ========================== MESH BUILDING ==============================
# =======================================================================
		
	def DoubleBufferedMesh(self, max_height):
		"""
		Given a boundary height Lz_b, returns a FEniCS mesh object with
			- canonical elements in the bottom of the cell domain (cells are always counted onto the same mesh)
			- buffer elements at the top of the cell domain (so upper boundary can have an arbitrary height.
		Having a double buffer layer avoids generating low-volume elements if Lz_b ~ n*h, but adds the constraint that 
		delta >= 2*h.
		/!\ Exports boundary height Lz_b as a global variable, to be used by TopDirichletBoundary.
		"""
		global Lz_b
		
		# Read off fixed parameters
		L_x = self.L_x; N_x = self.N_x
		L_y = self.L_y; N_y = self.N_y
		delta = self.delta
		h = self.h
		
		# Calculate DBM dimensions
		Lz_b = max_height + delta				 # height in um at which to apply bulk BC 
		A = int(Lz_b // h)		 				 # number of whole cubes that fit under Lz_b
		B = Lz_b % h							 # remainder for this division
		W = B + h								 # thickness of buffer layer in um
		
		# Update mesh variables
		self.Lz_b = Lz_b
		self.W = W
		self.N_z = A-1	
		self.L_z = (A-1)*h					

		# Create the node cloud and connectivity
		P = N_x+1; Q = N_y+1; R = A+2;
		cloud = self.GetDBLNodeCloud(P, Q, R)
		TRI = self.GetNodeConnectivity(P, Q, R)

		# Reformat the arrays to a datatype that FEniCS likes
		cells = numpy.array(TRI, dtype=numpy.uintp)
		nodes = numpy.array(cloud, dtype=numpy.double)

		# Pass the node and cell arrays to the FEniCS mesh editor
		mesh = Mesh(); editor = MeshEditor()
		editor.open(mesh, 3, 3); editor.init_vertices(nodes.shape[0]); editor.init_cells(cells.shape[0])
		[editor.add_vertex(i,n) for i,n in enumerate(nodes)]
		[editor.add_cell(i,n) for i,n in enumerate(cells)]
		editor.close()
		
		return mesh
		
		
	def GetDBLNodeCloud(self, P, Q, R):
		"""
		Compute node locations for a double-buffer-layer mesh
		"""
		x = numpy.linspace(0.0, self.L_x, num = P)
		y = numpy.linspace(0.0, self.L_y, num = Q)
		z = numpy.linspace(0.0, (self.N_z+2)*self.h, num = R)
		(X, Y, Z) = numpy.meshgrid(x, y, z, indexing ='ij')

		# Move the top two layers to make the buffer layer
		Z[:,:,-1] = self.Lz_b;              			           
		Z[:,:,-2] = self.Lz_b - 0.5*self.W;  

		# Flatten into a 3-by-(Num_nodes) array
		cloud = numpy.vstack((X.flatten('F'), Y.flatten('F'), Z.flatten('F'))).T
		
		return cloud
		
		
	def GetNodeConnectivity(self, P, Q, R):
		"""
		Compute the connectivity TRI of a regular grid of points	
		"""
		# Create an P-by-Q-by-R array of integers, numbering along x then y then z
		(pp,qq,rr) = numpy.meshgrid(range(0,P),range(0,Q),range(0,R),indexing='ij');
		inds = numpy.vstack((pp.flatten('F'), qq.flatten('F'), rr.flatten('F'))).T

		# In each direction, remove the last set of nodes (non-origin nodes)
		mask = ((inds[:,0]==self.N_x) + (inds[:,1]==self.N_y) + (inds[:,2]==self.N_z+2) == False)
		inds_p = inds[mask]
		nods_p = inds_p[:,0] + P*inds_p[:,1] + P*Q*inds_p[:,2]

		# Compute the stencil defining the 6 tetrahedra associated with a given origin
		stencil = self.GetConnectivityStencil(P, Q)
					   
		# For each origin node, define the 6 associated elements; compile to list TRI
		K = numpy.tile(nods_p.T, (6, 1))
		TRI = (numpy.tile(K.flatten('F'), (4,1)) + numpy.tile(stencil, (len(nods_p),1) ).T).T
		
		return TRI
	
	
	def GetConnectivityStencil(self, P, Q):
		"""
		Given the vertices of a cube, group these points into 6 identical tetrahedra
		"""
		stencil = numpy.array([[0, 1,   P+1,     P*(Q+1)+1], \
							   [0, 1,   P*Q+1,   P*(Q+1)+1], \
							   [0, P*Q, P*Q+1,   P*(Q+1)+1], \
							   [0, P,   P+1,     P*(Q+1)+1], \
							   [0, P*Q, P*(Q+1), P*(Q+1)+1], \
							   [0, P,   P*(Q+1), P*(Q+1)+1]])
		return stencil

# =======================================================================
# ========================= VOLUME FRACTION A ===========================
# =======================================================================

	def GetTetrahedronIndex(self, Point, cubeOrigin):
		"""
		Given mesh cube, assign which tetrahedron a point is in.
		/!\ Assumes tetrahedron is part of a cube.
		"""
		Origin = cubeOrigin
		p_x = Point[0]; p_y = Point[1]; p_z = Point[2]
		a_x = Origin[0]; a_y = Origin[1]; a_z = Origin[2]
		dx = p_x - a_x
		dy = p_y - a_y
		dz = p_z - a_z
		t = 1*(dy - dz > 0) + 2*(dz - dx > 0) + 4*(dx - dy > 0)
		conv_vec = [3,4,5,1,0,2]
		
		return conv_vec[t-1]


	def GetCubeIndex(self, Point):
		"""
		Given mesh dimensions, assign which cube a point is in.
		"""
		p_x = Point[0]; p_y = Point[1]; p_z = Point[2]
		p = int(numpy.floor(p_x*self.N_x / float(self.L_x)))  		# index along x
		q = int(numpy.floor(p_y*self.N_y / float(self.L_y)))   		# index along y
		r = int(numpy.floor(p_z*self.N_z / float(self.L_z)))   		# index along z
		c = p + q*self.N_x + r*self.N_x*self.N_y					# global index of this cube
		cubeOrigin = [p*self.L_x/float(self.N_x),\
					  q*self.L_y/float(self.N_y),\
					  r*self.L_z/float(self.N_z)]							# coordinates of this cube's origin
		
		return int(c), cubeOrigin 


	def GetElementIndex(self, point):
		"""
		Get tetrahedron and cube indices and calculate global element index.
		"""
		[c, cubeOrigin] = self.GetCubeIndex(point)
		t = self.GetTetrahedronIndex(point,cubeOrigin)
		
		return t + 6*c

	
	def AssignElementsToData(self, centers):
		"""
		Sort cell centres into their respective mesh elements.
		"""
		N = centers.shape[0]
		elements = numpy.zeros((N), numpy.int32)
		for i in range(0,N):
			point = centers[i]
			elements[i] = self.GetElementIndex(point)
			
		return elements
		
		
	def GetVolFracs(self, centers, vols):
		"""
		Create a global list of the cell volume fractions in mesh elements.
		Assumes that self.mesh and self.h are up-to-date.
		/!\ Exports the array vol_fracs as a global array, for use by VolumeFraction.
		"""
		global vol_fracs

		# assign elements of cells
		elements = self.AssignElementsToData(centers)
		
		# need to define volume fraction for every element in the mesh
		# (not just the canonical elements for counting)
		num_elements = self.mesh.num_cells()

		# sum cell volumes over each element
		v = math.pow(self.h, 3) / 6.0
		vol_fracs = numpy.bincount(elements,vols,num_elements) / v


# =======================================================================
# ========================== INTERPOLATION ==============================
# =======================================================================


	def InterpolateToCenters(self, centers):
		"""
		Interpolate a solution object u onto a list of cell coordinates
		"""
		u = self.solution
		data_t = tuple(map(tuple, centers)) 	   		     # Convert to tuple format
		u_local = numpy.zeros((len(data_t),),numpy.float64)  # preallocate solution array
		for i in range(0,len(data_t)):				  		 # loop over all cells
			u_local[i] = u(data_t[i][0:3])					 # extrapolate solution value at cell centre
		
		return u_local
		
# =======================================================================
# ============================= FILE I/O ================================
# =======================================================================
		
	def WriteFieldToFile(self, filename, u):
		"""
		Export the PDE solution as a pvd mesh.
		"""
		print "Writing fields..."
		File(filename) << u
		print 'Done.'


	def VolumeFractionOnElements(self):
		""" 
		Monod function with which to build RHS.
		"""
		return VolumeFraction()
		
# =======================================================================
# ============================= BOUNDARIES ==============================
# =======================================================================

class TopDirichletBoundary(SubDomain):

    def inside(self, x, on_boundary):
		"""
		Determine whether point x lies on the Dirchlet Boundary subdomain.
		/!\ Assumes Lz_b is supplied as a global variable.
		"""
		global Lz_b
		
		return bool(near(x[2],Lz_b) and on_boundary)


class XYPeriodicDomain(SubDomain):

    def inside(self, x, on_boundary):
    	"""
    	Return true if we are on either of the two master boundaries.
    	/!\ Assumes that origin = [0,0,z]!
    	"""
        return bool((near(x[0], 0.0) or near(x[1], 0.0)) and on_boundary)

    def map(self, x, y):
		""" 
		Map points on the slave boundaries to the corresponding master boundaries.
		/!\ Takes L_x and L_y as global variables.
		"""
		global L_x
		global L_y		
		
		if near(x[0], L_x) and near(x[1], L_y):
			y[0] = x[0] - L_x    
			y[1] = x[1] - L_y         
			y[2] = x[2]         
		elif near(x[0], L_x):
			y[0] = x[0] - L_x
			y[1] = x[1]       
			y[2] = x[2]       
		elif near(x[1], L_y):
			y[0] = x[0] 
			y[1] = x[1] - L_y    
			y[2] = x[2] 
		else:
			y[0] = x[0]
			y[1] = x[1]
			y[2] = x[2]	
			

# =======================================================================
# ========================= VOLUME FRACTION B ===========================
# =======================================================================
		
class VolumeFraction(Expression):

	  def eval_cell(self, value, x, ufc_cell):
		  """
		  Evaluate the cell volume fraction for this mesh element.
		  /!\ Assumes vol_fracs is being supplied as a global variable.
		  """
		  global vol_fracs
		  value[0] = vol_fracs[ufc_cell.index]