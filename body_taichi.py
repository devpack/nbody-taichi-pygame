import taichi as ti
import numpy as np
# -----------------------------------------------------------------------------------------------------------

@ti.dataclass
class FieldPoint:

    pos: ti.types.vector(3, ti.f32)
    #col: ti.types.vector(3, float)

# -----------------------------------------------------------------------------------------------------------

@ti.dataclass
class Body:

    pos: ti.types.vector(3, float)
    vel: ti.types.vector(3, float)
    acc: ti.types.vector(3, float)

    mass: float

# -----------------------------------------------------------------------------------------------------------

@ti.data_oriented
class NBodySystem:

    def __init__(self, p1_mass=1.0, p2_mass=1.0, p3_mass=1.0, px_mass=1.0 , nb_body=8, field_size=10):
        self.nb_body = nb_body

        self.p1_mass = p1_mass
        self.p2_mass = p2_mass
        self.p3_mass = p3_mass
        self.px_mass = px_mass
        self.bodies = Body.field(shape=self.nb_body)

        self.field_size = field_size
        self.field_points = FieldPoint.field(shape=self.field_size*self.field_size)

    @ti.kernel
    #def init_field(self, min:ti.f32, max:ti.f32, ppl:ti.i32):
    def init_field(self, arr: ti.types.ndarray(ti.types.vector(3, ti.f32))):

        for i in range(0, self.field_size*self.field_size):
            fp = FieldPoint()
            #print(arr[i])
            fp.pos = arr[i]
            self.field_points[i] = fp

    @ti.kernel
    def init_bodies_2(self):

        p = Body()
        p.pos = [0.0, 0.0, 0.0]
        p.vel = [0.0, 0.0, 0.0]
        p.acc = [0.0, 0.0, 0.0]
        p.mass = self.p1_mass
        self.bodies[0] = p

        p = Body()
        p.pos = [1.0, 0.0, 0.0]
        p.vel = [0.0, 0.8, 0.0]
        p.acc = [0.0, 0.0, 0.0]
        p.mass = self.p2_mass
        self.bodies[1] = p

    @ti.kernel
    def init_bodies(self):

        for i in range(self.nb_body):

            p = Body()
            
            #p.pos = [ (ti.random(float)*2.)-1., (ti.random(float)*2.)-1, (ti.random(float)*2.)-1] # [-1, 1] coords
            p.pos = [ (ti.random(float)*2.)-1., (ti.random(float)*2.)-1, 0.0] # [-1, 1] coords
            p.vel = [0.0, 0.0, 0.0]
            p.acc = [0.0, 0.0, 0.0]

            if i==0:
                p.mass = self.p1_mass
            elif i==1:
                p.mass = self.p2_mass
            elif i==2:
                p.mass = self.p3_mass
            else:
                p.mass = self.px_mass

            self.bodies[i] = p

    @ti.kernel
    def update(self, dt: ti.f32, eps: ti.f32):

        for i in range(self.nb_body):
            #if i != 0:
                self.bodies[i].vel += self.bodies[i].acc * 0.5 * dt
                self.bodies[i].pos += self.bodies[i].vel * dt

        # only the outer loop is optimized 
            
        # For tne next for loop set the number of threads in a block on GPU
        # ti.loop_config(block_dim=8)

        # For tne next for loop set the number of threads in a block on CPU
        #ti.loop_config(parallelize=8)

        eps2 = eps*eps

        #for i, j in ti.ndrange(self.nb_body, self.nb_body): # does not work on vulkan / opengl ?
        for i in range(self.nb_body):
            for j in range(self.nb_body):
                if i != j:
                    DR = self.bodies[j].pos - self.bodies[i].pos
                    DR2 = ti.math.dot(DR, DR)
                    DR2 += eps2

                    PHI = self.bodies[j].mass / (ti.sqrt(DR2) * DR2)

                    self.bodies[i].acc += DR * PHI

        for i in range(self.nb_body):
            self.bodies[i].vel += self.bodies[i].acc * 0.5 * dt
            self.bodies[i].acc = [0.0, 0.0, 0.0]  