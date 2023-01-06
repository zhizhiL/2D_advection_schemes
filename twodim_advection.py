import numpy as np

class advection_2D(object):

    def __init__(self, xmin, xmax, ymin, ymax, NX, NY, dt):
        """
        create a 2D-field object with desired domain and grid cells
        arg: NX, NY are the number of grid POINTS, INCL. GHOST POINTS
        """
        self.NX = NX
        self.NY = NY
        self.dx = (xmax - xmin) / (self.NX - 2)
        self.dy = (ymax - ymin) / (self.NY - 2)
        self.dt = dt

        ## physical domain
        self.x = np.linspace(xmin + self.dx/2, xmax - self.dx/2, (self.NX - 2))
        self.y = np.linspace(ymin + self.dy/2, ymax - self.dy/2, (self.NY - 2))
   

    def meshgrid_2D(self):
        """
        ret: (xxi, yyi) a 2D meshgrid based on the required grid points
             can be used to generate initial scalar/vector field
             shape: (ny, nx), NOT including ghost points!
        """
        xxi, yyi = np.meshgrid(self.x, self.y)
        return xxi, yyi


    def LaxFriedrich_vect(self, qn, U, V):
        """
        implements Lax Friedrich scheme in VECTOR operations
        the validity of this scheme is subjects to the CFL condition, i.e. c*dt/dx <= 1, to avoid instability
        Here, NO B.C. has been imposed on ghost points, will need extra constraint for the complete implementation
        **NOTE**: this scheme is subjected to severe dispersion effect when c*dt/dx < 1 by introducing a numerical diffusion term
                  should only be used with caution!!!

        arg: 
        qn - quantity to be advected, ndarray (NY, NX)
        U, V - velocity field, ndarray (NY, NX)
        """

        cfl_x = np.max(U) * self.dt / self.dx
        cfl_y = np.max(V) * self.dt / self.dy

        if cfl_x > 1 or cfl_y > 1 :
            raise ValueError('The CFL condition has failed, this scheme no longer stable')
        
        else:
            qa = np.zeros_like(qn)
            qa[1:-1, 1:-1] = (qn[:-2, 1:-1] + qn[2:, 1:-1] + qn[1:-1,:-2] + qn[1:-1, 2:]) / 4   \
                        - U[1:-1, 1:-1] * self.dt * (qn[1:-1, 2:] - qn[1:-1, :-2]) / (2*self.dx)  \
                        - V[1:-1, 1:-1] * self.dt * (qn[2:, 1:-1] - qn[:-2, 1:-1]) / (2*self.dy)
            
            return qa


    def LaxWendroff_vect(self, qn, U, V):
        """
        implements Lax Friedrich scheme in VECTOR operations
        the validity of this scheme is subjects to the CFL condition, i.e. c*dt/dx \leq 1, to avoid instability
        Here, NO B.C. has been imposed on ghost points, will need extra constraint for the complete implementation
        arg: 
        qn - quantity to be advected, ndarray (NY, NX)
        U, V - velocity field, ndarray (NY, NX)
        """
        
        cfl_x = np.max(U) * self.dt / self.dx
        cfl_y = np.max(V) * self.dt / self.dy

        if cfl_x > 1 or cfl_y > 1 :
            raise ValueError('The CFL condition has failed, this scheme no longer stable')
        
        else:
            vel_sq = U**2 + V**2
            qa = np.zeros_like(qn)
            qa[1:-1, 1:-1] = qn[1:-1, 1:-1] - U[1:-1, 1:-1] * self.dt * (qn[1:-1, 2:] - qn[1:-1, :-2]) / (2*self.dx) \
                                   - V[1:-1, 1:-1] * self.dt * (qn[2:, 1:-1] - qn[:-2, 1:-1]) / (2*self.dy) \
                                   + (vel_sq[1:-1, 1:-1] * (self.dt**2) / 2) * ( (qn[1:-1, 2:] - 2*qn[1:-1, 1:-1] + qn[1:-1, :-2]) / (self.dx**2) \
                                   + (qn[2:, 1:-1] - 2*qn[1:-1, 1:-1] + qn[:-2, 1:-1]) / (self.dy**2) )
            return qa




    def Semilag(self, u, v, q):
        """
        1st order semi-Lagrangian advection
        """ 
        ADVq = np.zeros_like(q)
        
    # Matrices where 1 is right, 0 is left or center
        Mx2 = np.sign(np.sign(u[1:-1,1:-1]) + 1.)
        Mx1 = 1. - Mx2

    # Matrices where 1 is up, 0 is down or center
        My2 = np.sign(np.sign(v[1:-1,1:-1]) + 1.)
        My1 = 1. - My2

    # Matrices of absolute values for u and v
        au = abs(u[1:-1,1:-1])
        av = abs(v[1:-1,1:-1]) 

    # Matrices of coefficients respectively central, external, same x, same y
        Cc = (self.dx - au*self.dt) * (self.dy - av*self.dt)/self.dx/self.dy 
        Ce = self.dt*self.dt*au * av/self.dx/self.dy
        Cmx = (self.dx - au*self.dt) * av*self.dt/self.dx/self.dy
        Cmy =  self.dt*au*(self.dy - self.dt*av) /self.dx/self.dy


    # Computes the advected quantity
        ADVq[1:-1,1:-1] = (Cc * q[1:-1, 1:-1] +            
                        Ce * (Mx1*My1 * q[2:, 2:] + 
                                Mx2*My1 * q[2:, :-2] +
                                Mx1*My2 * q[:-2, 2:] +
                                Mx2*My2 * q[:-2, :-2]) +  
                        Cmx * (My1 * q[2:, 1:-1] +
                                My2 * q[:-2, 1:-1])+
                        Cmy * (Mx1 * q[1:-1, 2:] +
                                Mx2 * q[1:-1, :-2]))

        return ADVq

   
    def Semilag2(self, u,v,q):
        """
        Second order semi-Lagrangian advection
        """
    
        qstar=self.Semilag(u, v, q)
        qtilde=self.Semilag(-u,-v,qstar)    
        qstar  = q + (q-qtilde)/2;
        ADVq=self.Semilag(u,v,qstar)

        return ADVq