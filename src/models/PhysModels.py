import torch
import torch.nn as nn

class Damped_oscillation(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([init_phys*0.1], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.5], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        self.beta = nn.Parameter(self.beta )

        self.order = 2

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z[:,1:2]
      y0 = z[:,0:1]

      dt = dt
      y_hat = y1 +(y1 - y0) - dt*(self.beta*(y1-y0) +dt*self.alpha*y1 )

      return  y_hat
    
class dyn_1storder(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([init_phys], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.5], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        #self.beta = nn.Parameter(self.beta )

        self.order = 1

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z

      dt = dt

      y_hat= y1 - dt*self.alpha*y1

      return  y_hat
    
class lineal(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([init_phys], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.5], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        self.beta = nn.Parameter(self.beta )

        self.order = 1

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z

      dt = dt

      y_hat= y1 - dt*self.alpha

      return  y_hat
   
class Oscillation(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.alpha = torch.tensor([-0.5], requires_grad=True).float()
        self.alpha = nn.Parameter(self.alpha )

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z[:,1:2]
      y0 = z[:,0:1]

      dt = dt

      y_hat = y1+ (y1-y0) -dt*dt *self.alpha* y1


      return  y_hat

class ODE_2ObjectsSpring(nn.Module):
    def __init__(self, k, eq_distance):
        super().__init__()        
               
        self.eq_distance = torch.tensor([eq_distance], requires_grad=True).float()
        self.eq_distance = nn.Parameter(self.eq_distance)

        
        self.k = torch.tensor([k], requires_grad=True).float()
        self.k = nn.Parameter(self.k)

        self.relu = nn.ReLU()

    def force_eq(self, p1, p2):
        diff = (p2 - p1)
        euclidean_distance = torch.norm(diff, dim=1, keepdim=True)
        direction = (p2 - p1)/euclidean_distance
        #Force = self.k*(euclidean_distance - self.eq_distance )*direction
        #Force = torch.exp(self.k)*diff - torch.exp(self.k)*torch.exp(self.eq_distance)*direction
        Force = self.k*(euclidean_distance - 2*torch.abs(self.eq_distance) )*direction
        return Force
    def vel_eq(self, v):
        
        return v
    
    def runge_kutta_force(self,f, p1, p2, dt):
        k1 = f(p1, p2)
        k2 = f(p1 + dt/2, p2 + dt/2)
        k3 = f(p1 + dt/2, p2 + dt/2)
        k4 = f(p1 + dt, p2 + dt)
        return (k1 + 2*k2 + 2*k3 + k4)/6
    def runge_kutta_vel(self,f, v, dt):
        k1 = f(v)
        k2 = f(v + dt/2)
        k3 = f(v + dt/2)
        k4 = f(v + dt)
        return (k1 + 2*k2 + 2*k3 + k4)/6

    def forward(self, x, dt):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.tensor([dt], requires_grad=False).float().to(device)
    
        
        p1 = x[:,1,0:2]
        p2 = x[:,1,2:4]

        p1_0 = x[:,0,0:2]
        p2_0 = x[:,0,2:4]

        v1 = (p1 - p1_0)/dt
        v2 = (p2 - p2_0)/dt
            

        diff = (p1 - p2)

        
        #Force = self.runge_kutta_force(self.force_eq, p1, p2, dt)

        Force = self.force_eq(p1, p2)
        p1_new = 2*p1 -p1_0 + Force*dt*dt
        p2_new = 2*p2 -p2_0 - Force*dt*dt

        #Force = self.runge_kutta_force(self.force_eq, p1, p2, dt)

        #v11 = 
        #p1_new = p1 + self.runge_kutta_vel(self.vel_eq, v1 + Force , dt)
        #p2_new = p2 + self.runge_kutta_vel(self.vel_eq, v2 - Force , dt)



        

            #p1_0 = p1
            #p2_0 = p2
            #p1 = p1_new
            #p2 = p2_new

        
        z_hat = torch.cat((p1_new.unsqueeze(1) ,p2_new.unsqueeze(1)),dim=2)

        #print("z",x)

        #print("z_hat",z_hat)

        return z_hat

class Pendulum(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([init_phys*0.1], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.5], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        self.beta = nn.Parameter(self.beta )

        self.order = 2

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z[:,1:2]
      y0 = z[:,0:1]

      dt = dt

      y_hat = y1 +(y1 - y0) - dt*(self.beta*(y1-y0) +dt*(9.80665/ (torch.abs(self.alpha)+1e-5)  )*torch.sin(y1) )

      return  y_hat

class Sliding_block(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([init_phys], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.5], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        self.beta = nn.Parameter(self.beta )

        self.order = 2

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z[:,1:2]
      y0 = z[:,0:1]

      dt = dt
      #theta = torch.abs(self.alpha)*torch.pi/180
      #theta = torch.abs(self.alpha)

      y_hat = y1 +(y1 - y0) + dt*dt*self.alpha*10

      return  y_hat

class free_fall(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        print("Here! init_phys for free fall",init_phys)

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([init_phys*0.1], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.5], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        #self.beta = nn.Parameter(self.beta )

        self.order = 2

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z[:,1:2]
      y0 = z[:,0:1]    

      y_hat = 2*y1-y0 - self.alpha*dt*dt

      return  y_hat  

class led(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([0.0], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.0], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        #self.beta = nn.Parameter(self.beta )

        self.order = 1

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z

      dt = dt

      y_hat= y1 - dt*self.alpha*y1

      return  y_hat

class torricelli(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([0.0], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.0], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        #self.beta = nn.Parameter(self.beta )

        self.order = 1

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z

      dt = dt

      # make sure y1 is positive

      y1 = torch.abs(y1)

      y_hat= y1 - dt*self.alpha*torch.sqrt(y1)

      return  y_hat

def getModel(name, init_phys = None):

    if name == "IntegratedFire":

        return IntegratedFire()
    if name == "dyn_1storder":
        return dyn_1storder(init_phys)
    if name == "Clifford_Attractor":    
        return Clifford_Attractor()
    if name == "ODE_2ObjectsSpring":
        return ODE_2ObjectsSpring(init_phys[0], init_phys[1])
    if name == "Damped_oscillation":
        return Damped_oscillation(init_phys)
    elif name == "Oscillation":
        return Oscillation()
    elif name == "Sprin_ode":
        return Sprin_ode()
    elif name == "gravity_ode":
        return gravity_ode()
    elif name == "double_pendulum":
        return double_pendulum()
    elif name == "lineal":
        return lineal()
    elif name == "pendulum":
        return Pendulum(init_phys)
    elif name == "sliding_block"  :
        return Sliding_block(init_phys)
    elif name == "bouncing_ball" or name == "dropped_ball" or name == "free_fall":
        return free_fall(init_phys)
    elif name == "led":
        return led(init_phys)
    elif name == "torricelli":
        return torricelli(init_phys)
    else:
        return None
    
