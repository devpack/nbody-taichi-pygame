import taichi as ti
import numpy as np

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

    def __init__(self, screen_width: ti.f32, screen_height: ti.f32, fov: ti.f32, near: ti.f32, far: ti.f32,
                 p1_mass=1.0, p2_mass=1.0, p3_mass=1.0, px_mass=1.0 , nb_body=5, trace_len=1000):
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.aspect_ratio = screen_width / screen_height
        self.fov = fov
        self.near = near
        self.far = far

        self.nb_body = nb_body

        self.p1_mass = p1_mass
        self.p2_mass = p2_mass
        self.p3_mass = p3_mass
        self.px_mass = px_mass

        self.bodies = Body.field(shape=self.nb_body)
        #self.bodies = Body.field()
        #ti.root.dense(ti.i, self.nb_body).place(self.bodies)

        #self.projected_points = ti.types.ndarray(dtype=ti.math.vec2, ndim=1) # not working: AttributeError: 'NdarrayType' object has no attribute '__getitem__'
        #self.projected_points = ti.field(ti.math.vec2, shape=1) # not working
        self.projected_points = ti.Vector.field(2, ti.f32, self.nb_body) # working
        #self.projected_points = ti.Vector.field(2, ti.f32)
        #ti.root.dense(ti.i, self.nb_body).place(self.projected_points)

        self._projected_points = ti.Vector.field(3, ti.f32, self.nb_body * trace_len) # working

    # not used
    @ti.kernel
    def project_point(self, m_proj: ti.math.mat4, m_view: ti.math.mat4, m_model: ti.math.mat4, ndc_hmg_vec4: ti.math.vec4) -> ti.math.vec2:

        #pt_proj_vec4 = m_proj @ m_view @ m_model @ ndc_hmg_vec4
        pt_proj_vec4 = ndc_hmg_vec4 @ m_model @ m_view @ m_proj # inversed
        
        x_proj = (self.screen_width/2) +  (pt_proj_vec4.x/pt_proj_vec4.w)*(self.screen_width/2)
        y_proj = (self.screen_height/2) - (pt_proj_vec4.y/pt_proj_vec4.w)*(self.screen_height/2)

        return ti.math.vec2(x_proj, y_proj)

    @ti.kernel
    def project_points_from_ti_bodies(self, m_proj: ti.math.mat4, m_view: ti.math.mat4, m_model: ti.math.mat4):

        for i in range(self.nb_body):

            pt_proj_vec4 = ti.math.vec4(self.bodies[i].pos, 1) @ m_model @ m_view @ m_proj

            x_proj = (self.screen_width/2)  + (pt_proj_vec4.x/pt_proj_vec4.w)*(self.screen_width/2)
            y_proj = (self.screen_height/2) - (pt_proj_vec4.y/pt_proj_vec4.w)*(self.screen_height/2)

            self.projected_points[i] = ti.math.vec2(x_proj, y_proj)

    @ti.kernel
    def project_points_from_deque(self, m_proj: ti.math.mat4, m_view: ti.math.mat4, m_model: ti.math.mat4, in_points: ti.types.ndarray(ti.math.vec3, ndim=1)):

        for i in in_points:
            pt_proj_vec4 = ti.math.vec4(in_points[i], 1) @ m_model @ m_view @ m_proj

            x_proj = (self.screen_width/2)  + (pt_proj_vec4.x/pt_proj_vec4.w)*(self.screen_width/2)
            y_proj = (self.screen_height/2) - (pt_proj_vec4.y/pt_proj_vec4.w)*(self.screen_height/2)

            self._projected_points[i] = ti.math.vec3(x_proj, y_proj, pt_proj_vec4.w)

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
        p.vel = [0.0, 0.7, 0.0]
        p.acc = [0.0, 0.0, 0.0]
        p.mass = self.p2_mass
        self.bodies[1] = p

    # Stables 3 bodies orbit: https://observablehq.com/@rreusser/periodic-planar-three-body-orbits
    @ti.kernel
    def init_bodies_3(self):

        p = Body()
        p.pos = [-1.0, 0.0, 0.0]
        p.vel = [0.306893, 0.125507, 0.0]
        p.acc = [0.0, 0.0, 0.0]
        p.mass = self.p1_mass
        self.bodies[0] = p

        p = Body()
        p.pos = [1.0, 0.0, 0.0]
        p.vel = [0.306893, 0.125507, 0.0]
        p.acc = [0.0, 0.0, 0.0]
        p.mass = self.p2_mass
        self.bodies[1] = p

        p = Body()
        p.pos = [0.0, 0.0, 0.0]
        p.vel = [-0.613786, -0.251014, 0.0]
        p.acc = [0.0, 0.0, 0.0]
        p.mass = self.p2_mass
        self.bodies[2] = p

    @ti.kernel
    def init_bodies(self):

        for i in range(self.nb_body):

            p = Body()
            
            p.pos = [ (ti.random(float)*2.)-1., (ti.random(float)*2.)-1, (ti.random(float)*2.)-1] # [-1, 1] coords
            #p.pos = [ (ti.random(float)*2.)-1., (ti.random(float)*2.)-1, 0.0] # [-1, 1] coords
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

    # somehow the optimized version is slower
    @ti.kernel
    def update(self, dt: ti.f32, eps: ti.f32):
        """ Sym opt: calc i=>j and j=>i at the same time """

        for i in range(self.nb_body):
            self.bodies[i].vel += self.bodies[i].acc * 0.5 * dt
            self.bodies[i].pos += self.bodies[i].vel * dt

        eps2 = eps*eps

        for i in range(0, self.nb_body):
            for j in range(i+1, self.nb_body):
                if i != j:
                    DR = self.bodies[j].pos - self.bodies[i].pos
                    DR2 = ti.math.dot(DR, DR)
                    DR2 += eps2

                    PHI = DR / (ti.sqrt(DR2) * DR2)
                    self.bodies[i].acc += self.bodies[j].mass * PHI            
                    self.bodies[j].acc -= self.bodies[i].mass * PHI      

        for i in range(self.nb_body):
            self.bodies[i].vel += self.bodies[i].acc * 0.5 * dt
            self.bodies[i].acc = [0.0, 0.0, 0.0]  

    @ti.kernel
    def update_O2(self, dt: ti.f32, eps: ti.f32):
        """ O2 loop, no opt """

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